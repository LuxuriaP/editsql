"""Contains a main function for training and/or evaluating a model."""

import os
import sys

import numpy as np
import random
import shutil
import copy

from parse_args import interpret_args

import data_util
from data_util import atis_data
from model.schema_interaction_model import SchemaInteractionATISModel
from logger import Logger
from model.model import ATISModel
from model_util import Metrics, evaluate_utterance_sample, evaluate_interaction_sample, \
    train_epoch_with_utterances, train_epoch_with_interactions, evaluate_using_predicted_queries, \
    generate_samples, dis_train_epoch, dis_eval_epoch, get_progressbar
import progressbar
from data_util.atis_vocab import EOS_TOK

from data_util.dis_data_iter import DisDataIter

import torch
from torch.autograd import Variable

from model.discriminator import Discriminator
import torch.optim as optim
import torch.nn as nn

np.random.seed(0)
random.seed(0)

VALID_EVAL_METRICS = [Metrics.LOSS, Metrics.TOKEN_ACCURACY, Metrics.STRING_ACCURACY]
TRAIN_EVAL_METRICS = [Metrics.LOSS, Metrics.TOKEN_ACCURACY, Metrics.STRING_ACCURACY]
FINAL_EVAL_METRICS = [Metrics.STRING_ACCURACY, Metrics.TOKEN_ACCURACY]


def train(model, data, params, start_epoch=0):
    """ Trains a model.

    Inputs:
        model (ATISModel): The model to train.
        data (ATISData): The data that is used to train.
        params (namespace): Training parameters.
    """
    # Get the training batches.
    log = Logger(os.path.join(params.logdir, params.logfile), "w")
    num_train_original = atis_data.num_utterances(data.train_data)
    log.put("Original number of training utterances:\t"
            + str(num_train_original))

    eval_fn = evaluate_utterance_sample
    trainbatch_fn = data.get_utterance_batches
    trainsample_fn = data.get_random_utterances
    validsample_fn = data.get_all_utterances
    batch_size = params.batch_size
    if params.interaction_level:
        batch_size = 1
        eval_fn = evaluate_interaction_sample
        trainbatch_fn = data.get_interaction_batches
        trainsample_fn = data.get_random_interactions
        validsample_fn = data.get_all_interactions

    maximum_output_length = params.train_maximum_sql_length
    train_batches = trainbatch_fn(batch_size,
                                  max_output_length=maximum_output_length,
                                  randomize=not params.deterministic)

    if params.num_train >= 0:
        train_batches = train_batches[:params.num_train]

    training_sample = trainsample_fn(params.train_evaluation_size,
                                     max_output_length=maximum_output_length)
    valid_examples = validsample_fn(data.valid_data,
                                    max_output_length=maximum_output_length)

    num_train_examples = sum([len(batch) for batch in train_batches])
    num_steps_per_epoch = len(train_batches)

    log.put(
        "Actual number of used training examples:\t" +
        str(num_train_examples))
    log.put("(Shortened by output limit of " +
            str(maximum_output_length) +
            ")")
    log.put("Number of steps per epoch:\t" + str(num_steps_per_epoch))
    log.put("Batch size:\t" + str(batch_size))

    print(
        "Kept " +
        str(num_train_examples) +
        "/" +
        str(num_train_original) +
        " examples")
    print(
        "Batch size of " +
        str(batch_size) +
        " gives " +
        str(num_steps_per_epoch) +
        " steps per epoch")

    # Keeping track of things during training.
    epochs = start_epoch
    patience = params.initial_patience
    learning_rate_coefficient = 1.
    previous_epoch_loss = float('inf')
    maximum_validation_accuracy = 0.
    maximum_string_accuracy = 0.

    countdown = int(patience)

    if params.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model.trainer, mode='min', )

    keep_training = True
    while keep_training:
        log.put("Epoch:\t" + str(epochs))
        model.set_dropout(params.dropout_amount)

        if not params.scheduler:
            model.set_learning_rate(learning_rate_coefficient * params.initial_learning_rate)

        # Run a training step.
        if params.interaction_level:
            epoch_loss = train_epoch_with_interactions(
                train_batches,
                params,
                model,
                randomize=not params.deterministic,
                sampling=params.is_sampling
            )
        else:
            epoch_loss = train_epoch_with_utterances(
                train_batches,
                model,
                randomize=not params.deterministic)

        log.put("train epoch loss:\t" + str(epoch_loss))

        model.set_dropout(0.)

        # Run an evaluation step on a sample of the training data.
        train_eval_results = eval_fn(training_sample,
                                     model,
                                     params.train_maximum_sql_length,
                                     name=os.path.join(params.logdir, "train-eval"),
                                     write_results=True,
                                     gold_forcing=True,
                                     metrics=TRAIN_EVAL_METRICS)[0]

        for name, value in train_eval_results.items():
            log.put(
                "train final gold-passing " +
                name.name +
                ":\t" +
                "%.2f" %
                value)

        # Run an evaluation step on the validation set.
        valid_eval_results = eval_fn(valid_examples,
                                     model,
                                     params.eval_maximum_sql_length,
                                     name=os.path.join(params.logdir, "valid-eval"),
                                     write_results=True,
                                     gold_forcing=True,
                                     metrics=VALID_EVAL_METRICS)[0]
        for name, value in valid_eval_results.items():
            log.put("valid gold-passing " + name.name + ":\t" + "%.2f" % value)

        valid_loss = valid_eval_results[Metrics.LOSS]
        valid_token_accuracy = valid_eval_results[Metrics.TOKEN_ACCURACY]
        string_accuracy = valid_eval_results[Metrics.STRING_ACCURACY]

        if train_eval_results[Metrics.STRING_ACCURACY] >= params.gen_acc_threshold:
            keep_training = False

        if params.scheduler:
            scheduler.step(valid_loss)

        if valid_loss > previous_epoch_loss:
            learning_rate_coefficient *= params.learning_rate_ratio
            log.put(
                "learning rate coefficient:\t" +
                str(learning_rate_coefficient))

        previous_epoch_loss = valid_loss

        if string_accuracy > maximum_string_accuracy:
            maximum_string_accuracy = string_accuracy
            patience = patience * params.patience_ratio
            countdown = int(patience)

            log.put(
                "maximum string accuracy:\t" +
                str(maximum_string_accuracy))
            log.put("patience:\t" + str(patience))

        if countdown <= 0:
            keep_training = False

        countdown -= 1
        log.put("countdown:\t" + str(countdown))
        log.put("")

        epochs += 1

        if params.max_epoch and epochs >= params.max_epoch:
            keep_training = False

        # save checkpoint
        ckp = {
            'epoch': epochs,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': model.trainer.state_dict(),
            'bert_optimizer_state_dict': model.bert_trainer.state_dict()
        }
        save_ckp(ckp, params.logdir, params.gen_pretrain_ckp)

    log.put("Finished training!")
    log.close()


def evaluate(model, data, params, split):
    """Evaluates a pretrained model on a dataset.

    Inputs:
        model (ATISModel): Model class.
        data (ATISData): All of the data.
        params (namespace): Parameters for the model.
    """
    filename = split

    if filename == 'dev':
        split = data.dev_data
    elif filename == 'train':
        split = data.train_data
    elif filename == 'test':
        split = data.test_data
    elif filename == 'valid':
        split = data.valid_data
    else:
        raise ValueError("Split not recognized: " + str(params.evaluate_split))

    if params.use_predicted_queries:
        filename += "_use_predicted_queries"
    else:
        filename += "_use_gold_queries"

    full_name = os.path.join(params.logdir, filename) + params.results_note

    if params.interaction_level or params.use_predicted_queries:
        examples = data.get_all_interactions(split)
        if params.interaction_level:
            valid_eval_results = evaluate_interaction_sample(
                examples,
                model,
                name=full_name,
                metrics=FINAL_EVAL_METRICS,
                total_num=atis_data.num_utterances(split),
                database_username=params.database_username,
                database_password=params.database_password,
                database_timeout=params.database_timeout,
                use_predicted_queries=params.use_predicted_queries,
                max_generation_length=params.eval_maximum_sql_length,
                write_results=True,
                use_gpu=True,
                compute_metrics=params.compute_metrics)[0]
        else:
            valid_eval_results = evaluate_using_predicted_queries(
                examples,
                model,
                name=full_name,
                metrics=FINAL_EVAL_METRICS,
                total_num=atis_data.num_utterances(split),
                database_username=params.database_username,
                database_password=params.database_password,
                database_timeout=params.database_timeout)[0]
    else:
        examples = data.get_all_utterances(split)
        valid_eval_results = evaluate_utterance_sample(
            examples,
            model,
            name=full_name,
            gold_forcing=False,
            metrics=FINAL_EVAL_METRICS,
            total_num=atis_data.num_utterances(split),
            max_generation_length=params.eval_maximum_sql_length,
            database_username=params.database_username,
            database_password=params.database_password,
            database_timeout=params.database_timeout,
            write_results=True)[0]

    for name, value in valid_eval_results.items():
        print("valid gold-passing " + name.name + ":\t" + "%.2f" % value)

    valid_token_accuracy = valid_eval_results[Metrics.TOKEN_ACCURACY]
    string_accuracy = valid_eval_results[Metrics.STRING_ACCURACY]

    print("token accuracy:\t" + str(valid_token_accuracy))
    print("maximum string accuracy:\t" + str(string_accuracy))


def pretrain_discriminator(params, generator, discriminator,
                           dis_criterion, dis_optimizer, data,
                           start_epoch=0):
    log = Logger(os.path.join(params.logdir, params.dis_logfile), 'w')

    if params.interaction_level:
        get_data = data.get_all_interactions
    else:
        get_data = data.get_all_utterances

    train_data = get_data(
        data.train_data,
        max_output_length=params.train_maximum_sql_length
    )
    valid_data = get_data(
        data.valid_data,
        max_output_length=params.train_maximum_sql_length
    )

    real_train_path = os.path.join(params.samples_dir, params.real_train_file)
    fake_train_path = os.path.join(params.samples_dir, params.fake_train_file)
    real_valid_path = os.path.join(params.samples_dir, params.real_valid_file)
    fake_valid_path = os.path.join(params.samples_dir, params.fake_valid_file)

    generator.set_dropout(0.)

    if params.generated_train:
        print("Already generated training samples!")
    else:
        print("Generating training samples!")
        with torch.no_grad():
            if params.debug:
                generate_samples(generator, train_data,
                                 real_train_path, fake_train_path,
                                 params.train_maximum_sql_length,
                                 sampling=params.is_sampling,
                                 gen_num=100,
                                 train=True)
            else:
                generate_samples(generator, train_data,
                                 real_train_path, fake_train_path,
                                 params.train_maximum_sql_length,
                                 sampling=params.is_sampling,
                                 train=True)
        print("Finished generating training samples!")

    train_iter = DisDataIter(real_train_path,
                             fake_train_path,
                             params.dis_batch_size)

    log.put(
        "Number of training examples:\t" + str(train_iter.data_num)
    )
    log.put(
        "Number of steps per epoch:\t" + str(train_iter.num_batches)
    )
    log.put("Batch size:\t" + str(train_iter.batch_size))

    print(
        "Number of training examples: " + str(train_iter.data_num)
    )
    print("Batch size of " + str(train_iter.batch_size) + " gives "
          + str(train_iter.num_batches) + " steps per epoch")

    if params.generated_valid:
        print("Already generated validation samples!")
    else:
        print("Generating validation samples!")
        with torch.no_grad():
            if params.debug:
                generate_samples(generator, valid_data,
                                 real_valid_path, fake_valid_path,
                                 params.train_maximum_sql_length,
                                 sampling=params.is_sampling,
                                 gen_num=100)
            else:
                generate_samples(generator, valid_data,
                                 real_valid_path, fake_valid_path,
                                 params.train_maximum_sql_length,
                                 sampling=params.is_sampling)
        print("Finished generating validation samples!")

    valid_iter = DisDataIter(real_valid_path,
                             fake_valid_path,
                             params.dis_batch_size)

    print("Begin pre-training!")

    for epoch in range(start_epoch, params.num_dis_epoch):
        log.put("Epoch:\t" + str(epoch))
        print("Epoch: " + str(epoch))
        t_metrics = dis_train_epoch(
            discriminator,
            train_iter,
            dis_criterion,
            dis_optimizer)
        log.put("Train loss:\t" + str(t_metrics["loss"]))
        log.put("Train accuracy:\t" + str(t_metrics["acc"]))
        log.put("Train real accuracy:\t" + str(t_metrics["real_acc"]))
        log.put("Train fake accuracy:\t" + str(t_metrics["fake_acc"]))
        log.put("Train confidence:\t" + str(t_metrics["con"]))
        log.put("Train real confidence:\t" + str(t_metrics["real_con"]))
        log.put("Train fake confidence:\t" + str(t_metrics["fake_con"]))

        print("Train loss: " + str(t_metrics["loss"]))
        print("Train accuracy: " + str(t_metrics["acc"]))
        print("Train real accuracy: " + str(t_metrics["real_acc"]))
        print("Train fake accuracy: " + str(t_metrics["fake_acc"]))
        print("Train confidence: " + str(t_metrics["con"]))
        print("Train real confidence: " + str(t_metrics["real_con"]))
        print("Train fake confidence: " + str(t_metrics["fake_con"]))

        with torch.no_grad():
            v_metrics = dis_eval_epoch(
                discriminator,
                valid_iter,
                dis_criterion)
            log.put("Valid loss:\t" + str(v_metrics["loss"]))
            log.put("Valid accuracy:\t" + str(v_metrics["acc"]))
            log.put("Valid real accuracy:\t" + str(v_metrics["real_acc"]))
            log.put("Valid fake accuracy:\t" + str(v_metrics["fake_acc"]))
            log.put("Valid confidence:\t" + str(v_metrics["con"]))
            log.put("Valid real confidence:\t" + str(v_metrics["real_con"]))
            log.put("Valid fake confidence:\t" + str(v_metrics["fake_con"]))

            print("Valid loss: " + str(v_metrics["loss"]))
            print("Valid accuracy: " + str(v_metrics["acc"]))
            print("Valid real accuracy: " + str(v_metrics["real_acc"]))
            print("Valid fake accuracy: " + str(v_metrics["fake_acc"]))
            print("Valid confidence: " + str(v_metrics["con"]))
            print("Valid real confidence: " + str(v_metrics["real_con"]))
            print("Valid fake confidence: " + str(v_metrics["fake_con"]))

        # save checkpoint
        ckp = {
            'epoch': epoch + 1,
            'state_dict': discriminator.state_dict(),
            'optimizer_state_dict': dis_optimizer.state_dict(),
        }
        save_ckp(ckp, params.logdir, params.dis_pretrain_ckp)

        if t_metrics["con"] >= params.train_accuracy_threshold:
            break

    log.put("Finished pre-training discriminator!")
    log.close()
    print("Finished pre-training discriminator!")


def adv_train(generator, discriminator, dis_criterion,
              dis_optimizer, data, params, start_epoch=0,
              start_batches=None, start_pos_in_batch=0):
    log = Logger(os.path.join(params.logdir, params.adv_logfile), 'w')

    if params.interaction_level:
        get_batch = data.get_interaction_batches
        get_data = data.get_all_interactions
        get_sample = data.get_random_interactions
        evaluate = evaluate_interaction_sample
    else:
        get_batch = data.get_utterance_batches
        get_data = data.get_all_utterances
        get_sample = data.get_random_utterances
        evaluate = evaluate_utterance_sample

    if start_batches:
        train_batch = start_batches
    else:
        train_batch = get_batch(
            params.gan_batch_size,
            max_output_length=params.train_maximum_sql_length
        )
    num_batch = len(train_batch)

    train_data = get_data(
        data.train_data,
        max_output_length=params.train_maximum_sql_length
    )
    train_sample = get_sample(
        params.train_evaluation_size,
        max_output_length=params.train_maximum_sql_length
    )
    valid_data = get_data(
        data.valid_data,
        max_output_length=params.train_maximum_sql_length
    )

    progbar = get_progressbar("adversarial training     ",
                              num_batch * params.adv_epoch)
    progbar.start()
    print("")

    real_path = os.path.join(params.samples_dir, params.adv_real_file)
    fake_path = os.path.join(params.samples_dir, params.adv_fake_file)
    real_valid = os.path.join(params.samples_dir, params.adv_real_valid)
    fake_valid = os.path.join(params.samples_dir, params.adv_fake_valid)

    generator.set_dropout(params.dropout_amount)

    for epoch in range(start_epoch, params.adv_epoch):
        log.put("Epoch:\t" + str(epoch))
        print("Epoch: " + str(epoch))

        for i in range(start_pos_in_batch, num_batch):
            batch = train_batch[i]
            gen_loss = 0.

            progbar2 = get_progressbar("generator     ",
                                       params.gan_batch_size)
            progbar2.start()

            for j, example in enumerate(batch.items):
                seq, _, prob, pred = \
                    generator(example, params.train_maximum_sql_length,
                              sampling=params.is_sampling)
                if seq[-1] == EOS_TOK:
                    seq = seq[:-1]
                    prob = prob[:-1]
                with torch.no_grad():
                    rewards = generator.get_reward(
                        seq, pred, example, params.roll_num,
                        params.max_gen_len, discriminator,
                        bias=params.bias, mle=params.mle
                    )

                # log.put("Generator reward:\t" + str(rewards.tolist()))
                # print("Generator reward: " + str(rewards.tolist()))

                rewards = torch.Tensor(rewards).cuda()
                loss = generator.update_gan_loss(prob, rewards)

                gen_loss += loss

                torch.cuda.empty_cache()

                progbar2.update(j)

            progbar2.finish()

            log.put("Generator mean loss:\t" + str(gen_loss/params.gan_batch_size))
            print("Generator mean loss: " + str(gen_loss/params.gan_batch_size))

            if params.teacher_forcing:
                forcing_loss = 0.

                progbar3 = get_progressbar("forcing     ",
                                           params.gan_batch_size)
                progbar3.start()

                for j, example in enumerate(batch.items):
                    seq, _, prob, _ = \
                        generator(example, params.train_maximum_sql_length,
                                  sampling=params.is_sampling,
                                  forcing=True)
                    if seq[-1] == EOS_TOK:
                        seq = seq[:-1]
                        prob = prob[:-1]

                    rewards = torch.Tensor(np.ones(len(seq))).cuda()
                    loss = generator.update_gan_loss(prob, rewards)

                    forcing_loss += loss

                    torch.cuda.empty_cache()

                    progbar3.update(j)

                progbar3.finish()

                log.put("Forcing mean loss:\t" + str(forcing_loss/params.gan_batch_size))
                print("Forcing mean loss: " + str(forcing_loss/params.gan_batch_size))

            # Run an evaluation step on a sample of the training data.
            train_eval_results = evaluate(train_sample,
                                          generator,
                                          params.train_maximum_sql_length,
                                          name=os.path.join(params.logdir, "train-eval"),
                                          write_results=True,
                                          gold_forcing=True,
                                          metrics=TRAIN_EVAL_METRICS)[0]

            for name, value in train_eval_results.items():
                log.put(
                    "train final gold-passing " +
                    name.name +
                    ":\t" +
                    "%.2f" %
                    value)
                print(
                    "train final gold-passing " +
                    name.name +
                    ":\t" +
                    "%.2f" %
                    value)

            valid_eval_results = evaluate(valid_data,
                                          generator,
                                          params.eval_maximum_sql_length,
                                          name=os.path.join(params.logdir, "valid-eval"),
                                          write_results=True,
                                          gold_forcing=True,
                                          metrics=VALID_EVAL_METRICS)[0]
            for name, value in valid_eval_results.items():
                log.put("valid gold-passing " + name.name + ":\t" + "%.2f" % value)
                print("valid gold-passing " + name.name + ":\t" + "%.2f" % value)

            print("Generating training samples!")
            with torch.no_grad():
                generate_samples(generator, train_data,
                                 real_path, fake_path,
                                 params.max_gen_len,
                                 sampling=params.is_sampling,
                                 # gen_num=params.gen_num / (1 - train_eval_results[Metrics.STRING_ACCURACY] / 100.),
                                 train=True)
            print("Finished generating training samples!")

            dis_data_iter = DisDataIter(real_path,
                                        fake_path,
                                        params.dis_batch_size)

            print("Finetuning discriminator!")
            for _ in range(params.dis_k_steps):
                metrics = dis_train_epoch(
                    discriminator,
                    dis_data_iter,
                    dis_criterion,
                    dis_optimizer)
                log.put("Discriminator loss:\t" + str(metrics["loss"]))
                log.put("Discriminator accuracy:\t" + str(metrics["acc"]))
                log.put("Discriminator real accuracy:\t" + str(metrics["real_acc"]))
                log.put("Discriminator fake accuracy:\t" + str(metrics["fake_acc"]))
                log.put("Discriminator confidence:\t" + str(metrics["con"]))
                log.put("Discriminator real confidence:\t" + str(metrics["real_con"]))
                log.put("Discriminator fake confidence:\t" + str(metrics["fake_con"]))

                print("Discriminator loss: " + str(metrics["loss"]))
                print("Discriminator accuracy: " + str(metrics["acc"]))
                print("Discriminator real accuracy: " + str(metrics["real_acc"]))
                print("Discriminator fake accuracy: " + str(metrics["fake_acc"]))
                print("Discriminator confidence: " + str(metrics["con"]))
                print("Discriminator real confidence: " + str(metrics["real_con"]))
                print("Discriminator fake confidence: " + str(metrics["fake_con"]))

            print("Finished finetuning discriminator!")

            # save checkpoint
            ckp = {
                'epoch': epoch,
                'batches': train_batch,
                'pos_in_batch': i+1,
                'gen_state_dict': generator.state_dict(),
                'dis_state_dict': discriminator.state_dict(),
                'gen_optimizer_state_dict': generator.trainer.state_dict(),
                'gen_bert_optimizer_state_dict': generator.bert_trainer.state_dict(),
                'dis_optimizer_state_dict': dis_optimizer.state_dict()
            }
            save_ckp(ckp, params.logdir, params.adv_ckp)

            progbar.update(i)
            print("")

        random.shuffle(train_batch)
        random.shuffle(train_data)

        start_pos_in_batch = 0

    progbar.finish()
    log.put("Finished adversarial training!")
    log.close()


def mixed_mle(generator, discriminator, dis_criterion,
              dis_optimizer, data, params, start_epoch=0,
              start_batches=None, start_pos_in_batch=0,
              start_clamp=0., start_len=0):
    log = Logger(os.path.join(params.logdir, params.adv_logfile), 'w')

    if params.interaction_level:
        get_batch = data.get_interaction_batches
        get_data = data.get_all_interactions
        get_sample = data.get_random_interactions
        evaluate = evaluate_interaction_sample
    else:
        get_batch = data.get_utterance_batches
        get_data = data.get_all_utterances
        get_sample = data.get_random_utterances
        evaluate = evaluate_utterance_sample

    if start_batches:
        train_batch = start_batches
    else:
        train_batch = get_batch(
            params.gan_batch_size,
            max_output_length=params.train_maximum_sql_length
        )
    num_batch = len(train_batch)

    train_data = get_data(
        data.train_data,
        max_output_length=params.train_maximum_sql_length
    )
    train_sample = get_sample(
        params.train_evaluation_size,
        max_output_length=params.train_maximum_sql_length
    )
    valid_data = get_data(
        data.valid_data,
        max_output_length=params.train_maximum_sql_length
    )

    # find max length gold_query
    max_len = 0
    for example in train_data:
        utterance, = example.gold_utterances()
        if len(utterance.gold_query()) > max_len:
            max_len = len(utterance.gold_query())

    progbar = get_progressbar("adversarial training     ",
                              num_batch * params.adv_epoch * max_len)
    progbar.start()
    print("")

    real_path = os.path.join(params.samples_dir, params.adv_real_file)
    fake_path = os.path.join(params.samples_dir, params.adv_fake_file)

    generator.set_dropout(params.dropout_amount)

    for epoch in range(start_epoch, params.adv_epoch):
        log.put("Epoch:\t" + str(epoch))
        print("Epoch: " + str(epoch))

        clamp = start_clamp

        for k in range(start_len, max_len - 1):
            clamp -= params.step_size

            for i in range(start_pos_in_batch, num_batch):
                batch = train_batch[i]
                gen_loss = 0.

                progbar2 = get_progressbar("generator     ",
                                           params.gan_batch_size)
                progbar2.start()

                for j, example in enumerate(batch.items):
                    seq, _, prob, pred = \
                        generator(example, params.train_maximum_sql_length,
                                  sampling=params.is_sampling,
                                  forcing=True)
                    with torch.no_grad:
                        l = clamp if len(seq) + clamp > 0 else -len(seq) + 1
                        rewards, probs = generator.get_reward_mm(
                            seq[:l], pred[:l], example,
                            params.roll_num, params.max_gen_len,
                            discriminator
                        )

                    # log.put("Generator reward:\t" + str(rewards.tolist()))
                    # print("Generator reward: " + str(rewards.tolist()))

                    rewards = torch.Tensor(rewards).cuda()
                    loss = generator.update_gan_loss_mm(prob, probs, rewards)

                    gen_loss += loss

                    torch.cuda.empty_cache()

                    progbar2.update(j)

                progbar2.finish()

                log.put("Generator mean loss:\t" + str(gen_loss/params.gan_batch_size))
                print("Generator mean loss: " + str(gen_loss/params.gan_batch_size))

                # Run an evaluation step on a sample of the training data.
                train_eval_results = evaluate(train_sample,
                                              generator,
                                              params.train_maximum_sql_length,
                                              name=os.path.join(params.logdir, "train-eval"),
                                              write_results=True,
                                              gold_forcing=True,
                                              metrics=TRAIN_EVAL_METRICS)[0]

                for name, value in train_eval_results.items():
                    log.put(
                        "train final gold-passing " +
                        name.name +
                        ":\t" +
                        "%.2f" %
                        value)
                    print(
                        "train final gold-passing " +
                        name.name +
                        ":\t" +
                        "%.2f" %
                        value)

                valid_eval_results = evaluate(valid_data,
                                              generator,
                                              params.eval_maximum_sql_length,
                                              name=os.path.join(params.logdir, "valid-eval"),
                                              write_results=True,
                                              gold_forcing=True,
                                              metrics=VALID_EVAL_METRICS)[0]
                for name, value in valid_eval_results.items():
                    log.put("valid gold-passing " + name.name + ":\t" + "%.2f" % value)
                    print("valid gold-passing " + name.name + ":\t" + "%.2f" % value)

                print("Generating training samples!")
                with torch.no_grad():
                    generate_samples(generator, train_data,
                                     real_path, fake_path,
                                     params.max_gen_len,
                                     sampling=params.is_sampling,
                                     gen_num=params.gen_num)
                print("Finished generating training samples!")

                dis_data_iter = DisDataIter(real_path,
                                            fake_path,
                                            params.dis_batch_size)

                print("Finetuning discriminator!")
                metrics = dis_train_epoch(
                    discriminator,
                    dis_data_iter,
                    dis_criterion,
                    dis_optimizer)
                log.put("Discriminator loss:\t" + str(metrics["loss"]))
                log.put("Discriminator accuracy:\t" + str(metrics["acc"]))
                log.put("Discriminator real accuracy:\t" + str(metrics["real_acc"]))
                log.put("Discriminator fake accuracy:\t" + str(metrics["fake_acc"]))
                log.put("Discriminator confidence:\t" + str(metrics["con"]))
                log.put("Discriminator real confidence:\t" + str(metrics["real_con"]))
                log.put("Discriminator fake confidence:\t" + str(metrics["fake_con"]))

                print("Discriminator loss: " + str(metrics["loss"]))
                print("Discriminator accuracy: " + str(metrics["acc"]))
                print("Discriminator real accuracy: " + str(metrics["real_acc"]))
                print("Discriminator fake accuracy: " + str(metrics["fake_acc"]))
                print("Discriminator confidence: " + str(metrics["con"]))
                print("Discriminator real confidence: " + str(metrics["real_con"]))
                print("Discriminator fake confidence: " + str(metrics["fake_con"]))

                print("Finished finetuning discriminator!")

                # save checkpoint
                ckp = {
                    'epoch': epoch,
                    'batches': train_batch,
                    'pos_in_batch': i+1,
                    'gen_state_dict': generator.state_dict(),
                    'dis_state_dict': discriminator.state_dict(),
                    'gen_optimizer_state_dict': generator.trainer.state_dict(),
                    'gen_bert_optimizer_state_dict': generator.bert_trainer.state_dict(),
                    'dis_optimizer_state_dict': dis_optimizer.state_dict(),
                    'clamp': clamp,
                    'length': k
                }
                save_ckp(ckp, params.logdir, params.adv_ckp)

                progbar.update(i)
                print("")

            random.shuffle(train_batch)
            random.shuffle(train_data)

            start_pos_in_batch = 0

    progbar.finish()
    log.put("Finished adversarial training!")
    log.close()


def save_ckp(ckp, ckp_dir, ckp_filename):
    f_path = os.path.join(ckp_dir, ckp_filename)
    torch.save(ckp, f_path)


def load_ckp(ckp_file_path, model, optimizer, bert_optimizer=None):
    ckp = torch.load(ckp_file_path)
    epoch = ckp['epoch']
    model.load_state_dict(ckp['state_dict'])
    optimizer.load_state_dict(ckp['optimizer_state_dict'])
    if bert_optimizer:
        bert_optimizer.load_state_dict(ckp['bert_optimizer_state_dict'])
    return epoch, model, optimizer, bert_optimizer


def load_adv_ckp(ckp_path, gen, dis, gen_optm, dis_optm, bert_optm=None, mle=False):
    ckp = torch.load(ckp_path)
    epoch = ckp['epoch']
    batches = ckp['batches']
    pos_in_batch = ckp['pos_in_batch']
    gen.load_state_dict(ckp['gen_state_dict'])
    dis.load_state_dict(ckp['dis_state_dict'])
    gen_optm.load_state_dict(ckp['gen_optimizer_state_dict'])
    dis_optm.load_state_dict(ckp['dis_optimizer_state_dict'])
    if bert_optm:
        bert_optm.load_state_dict(ckp['gen_bert_optimizer_state_dict'])
    if mle:
        clamp = ckp['clamp']
        length = ckp['length']
    else:
        clamp, length = 0, 0

    return epoch, batches, pos_in_batch, gen, dis, gen_optm, dis_optm, bert_optm, clamp, length


def main():
    """Main function that trains and/or evaluates a model."""
    params = interpret_args()

    if params.gan:
        assert params.max_gen_len == params.train_maximum_sql_length \
               == params.eval_maximum_sql_length
        data = atis_data.ATISDataset(params)

        generator = SchemaInteractionATISModel(
            params,
            data.input_vocabulary,
            data.output_vocabulary,
            data.output_vocabulary_schema,
            None
        )

        generator = generator.cuda()

        generator.build_optim()

        if params.gen_from_ckp:
            gen_ckp_path = os.path.join(params.logdir, params.gen_pretrain_ckp)
            if params.fine_tune_bert:
                gen_epoch, generator, generator.trainer, \
                    generator.bert_trainer = \
                    load_ckp(
                        gen_ckp_path,
                        generator,
                        generator.trainer,
                        generator.bert_trainer
                    )
            else:
                gen_epoch, generator, generator.trainer, _ = \
                    load_ckp(
                        gen_ckp_path,
                        generator,
                        generator.trainer
                    )
        else:
            gen_epoch = 0

        print('====================Model Parameters====================')
        print('=======================Generator========================')
        for name, param in generator.named_parameters():
            print(name, param.requires_grad, param.is_cuda, param.size())
            assert param.is_cuda

        print('==================Optimizer Parameters==================')
        print('=======================Generator========================')
        for param_group in generator.trainer.param_groups:
            print(param_group.keys())
            for param in param_group['params']:
                print(param.size())

        if params.fine_tune_bert:
            print('=========================BERT===========================')
            for param_group in generator.bert_trainer.param_groups:
                print(param_group.keys())
                for param in param_group['params']:
                    print(param.size())

        sys.stdout.flush()

        # Pre-train generator with MLE
        if params.train:
            print('=============== Pre-training generator! ================')
            train(generator, data, params, gen_epoch)
            print('=========== Pre-training generator complete! ===========')

        dis_filter_sizes = [i for i in range(1, params.max_gen_len, 4)]
        dis_num_filters = [(100 + i * 10)
                           for i in range(1, params.max_gen_len, 4)]

        discriminator = Discriminator(
            params,
            data.dis_src_vocab,
            data.dis_tgt_vocab,
            params.max_gen_len,
            params.num_dis_classes,
            dis_filter_sizes,
            dis_num_filters,
            params.max_pos_emb,
            params.num_tok_type,
            params.dis_dropout
        )

        discriminator = discriminator.cuda()

        dis_criterion = nn.NLLLoss(reduction='mean')
        dis_criterion = dis_criterion.cuda()
        dis_optimizer = optim.Adam(discriminator.parameters())

        if params.dis_from_ckp:
            dis_ckp_path = os.path.join(params.logdir, params.dis_pretrain_ckp)
            dis_epoch, discriminator, dis_optimizer, _ = load_ckp(
                dis_ckp_path,
                discriminator,
                dis_optimizer
            )
        else:
            dis_epoch = 0

        print('====================Model Parameters====================')
        print('=====================Discriminator======================')
        for name, param in discriminator.named_parameters():
            print(name, param.requires_grad, param.is_cuda, param.size())
            assert param.is_cuda

        print('==================Optimizer Parameters==================')
        print('=====================Discriminator======================')
        for param_group in dis_optimizer.param_groups:
            print(param_group.keys())
            for param in param_group['params']:
                print(param.size())

        sys.stdout.flush()

        # Pre-train discriminator
        if params.pretrain_discriminator:
            print('============= Pre-training discriminator! ==============')
            pretrain_discriminator(
                params,
                generator,
                discriminator,
                dis_criterion,
                dis_optimizer,
                data,
                start_epoch=dis_epoch
            )
            print('========= Pre-training discriminator complete! =========')

        # Adversarial Training
        if params.adversarial_training:
            print('================ Adversarial training! =================')
            generator.build_optim()
            dis_criterion = nn.NLLLoss(reduction='mean')
            dis_optimizer = optim.Adam(discriminator.parameters())
            dis_criterion = dis_criterion.cuda()

            if params.adv_from_ckp and params.mle is not "mixed_mle":
                adv_ckp_path = os.path.join(params.logdir, params.adv_ckp)
                if params.fine_tune_bert:
                    epoch, batches, pos_in_batch, generator, discriminator, \
                        generator.trainer, dis_optimizer, \
                        generator.bert_trainer, _, _ = \
                        load_adv_ckp(
                            adv_ckp_path,
                            generator,
                            discriminator,
                            generator.trainer,
                            dis_optimizer,
                            generator.bert_trainer)
                else:
                    epoch, batches, pos_in_batch, generator, discriminator, \
                        generator.trainer, dis_optimizer, _, _, _ = \
                        load_adv_ckp(
                            adv_ckp_path,
                            generator,
                            discriminator,
                            generator.trainer,
                            dis_optimizer)
                adv_train(
                    generator,
                    discriminator,
                    dis_criterion,
                    dis_optimizer,
                    data,
                    params,
                    start_epoch=epoch,
                    start_batches=batches,
                    start_pos_in_batch=pos_in_batch
                )

            elif params.adv_from_ckp and params.mle == "mixed_mle":
                adv_ckp_path = os.path.join(params.logdir, params.adv_ckp)
                if params.fine_tune_bert:
                    epoch, batches, pos_in_batch, generator, discriminator, \
                        generator.trainer, dis_optimizer, \
                        generator.bert_trainer, clamp, length = \
                        load_adv_ckp(
                            adv_ckp_path,
                            generator,
                            discriminator,
                            generator.trainer,
                            dis_optimizer,
                            generator.bert_trainer,
                            mle=True)
                else:
                    epoch, batches, pos_in_batch, generator, discriminator, \
                        generator.trainer, dis_optimizer, _, clamp, length = \
                        load_adv_ckp(
                            adv_ckp_path,
                            generator,
                            discriminator,
                            generator.trainer,
                            dis_optimizer,
                            mle=True)
                mixed_mle(
                    generator,
                    discriminator,
                    dis_criterion,
                    dis_optimizer,
                    data,
                    params,
                    start_epoch=epoch,
                    start_batches=batches,
                    start_pos_in_batch=pos_in_batch,
                    start_clamp=clamp,
                    start_len=length
                )
            else:
                if params.mle == 'mixed_mle':
                    mixed_mle(
                        generator,
                        discriminator,
                        dis_criterion,
                        dis_optimizer,
                        data,
                        params
                    )
                else:
                    adv_train(
                        generator,
                        discriminator,
                        dis_criterion,
                        dis_optimizer,
                        data,
                        params
                    )

        if params.evaluate and 'valid' in params.evaluate_split:
            print("================== Evaluating! ===================")
            evaluate(generator, data, params, split='valid')
            print("============= Evaluation finished! ===============")

    # else:
    #     # Prepare the dataset into the proper form.
    #     data = atis_data.ATISDataset(params)
    #
    #     # Construct the model object.
    #     if params.interaction_level:
    #         model_type = SchemaInteractionATISModel
    #     else:
    #         print('not implemented')
    #         exit()
    #
    #     model = model_type(
    #         params,
    #         data.input_vocabulary,
    #         data.output_vocabulary,
    #         data.output_vocabulary_schema,
    #         data.anonymizer if params.anonymize and params.anonymization_scoring else None)
    #
    #     model = model.cuda()
    #     print('=====================Model Parameters=====================')
    #     for name, param in model.named_parameters():
    #         print(name, param.requires_grad, param.is_cuda, param.size())
    #         assert param.is_cuda
    #
    #     model.build_optim()
    #
    #     print('=====================Parameters in Optimizer==============')
    #     for param_group in model.trainer.param_groups:
    #         print(param_group.keys())
    #         for param in param_group['params']:
    #             print(param.size())
    #
    #     if params.fine_tune_bert:
    #         print('=====================Parameters in BERT Optimizer==============')
    #         for param_group in model.bert_trainer.param_groups:
    #             print(param_group.keys())
    #             for param in param_group['params']:
    #                 print(param.size())
    #
    #     sys.stdout.flush()
    #
    #     last_save_file = ""
    #
    #     if params.train:
    #         last_save_file = train(model, data, params)
    #     if params.evaluate and 'valid' in params.evaluate_split:
    #         evaluate(model, data, params, last_save_file, split='valid')
    #     if params.evaluate and 'dev' in params.evaluate_split:
    #         evaluate(model, data, params, last_save_file, split='dev')
    #     if params.evaluate and 'test' in params.evaluate_split:
    #         evaluate(model, data, params, last_save_file, split='test')


if __name__ == "__main__":
    main()
