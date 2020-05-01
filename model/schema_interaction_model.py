""" Class for the Sequence to sequence model for ATIS."""

import torch
import torch.nn.functional as F
from . import torch_utils

import numpy as np
import os
import copy

import data_util.snippets as snippet_handler
from data_util import sql_util
import data_util.vocabulary as vocab
from data_util.vocabulary import EOS_TOK, UNK_TOK
import data_util.tokenizers

from .token_predictor import construct_token_predictor
from .attention import Attention
from .model import ATISModel, encode_snippets_with_states, get_token_indices
from data_util.utterance import ANON_INPUT_KEY

from .encoder import Encoder
from .decoder import SequencePredictorWithSchema

from . import utils_bert

import data_util.atis_batch


LIMITED_INTERACTIONS = {"raw/atis2/12-1.1/ATIS2/TEXT/TRAIN/SRI/QS0/1": 22,
                        "raw/atis3/17-1.1/ATIS3/SP_TRN/MIT/8K7/5": 14,
                        "raw/atis2/12-1.1/ATIS2/TEXT/TEST/NOV92/770/5": -1}

END_OF_INTERACTION = {"quit", "exit", "done"}


class SchemaInteractionATISModel(ATISModel):
    """ Interaction ATIS model, where an interaction is processed all at once.
    """

    def __init__(self,
                 params,
                 input_vocabulary,
                 output_vocabulary,
                 output_vocabulary_schema,
                 anonymizer):
        ATISModel.__init__(
            self,
            params,
            input_vocabulary,
            output_vocabulary,
            output_vocabulary_schema,
            anonymizer)

        if self.params.use_schema_encoder:
            # Create the schema encoder
            schema_encoder_num_layer = 1
            schema_encoder_input_size = params.input_embedding_size
            schema_encoder_state_size = params.encoder_state_size
            if params.use_bert:
                schema_encoder_input_size = self.bert_config.hidden_size

            self.schema_encoder = Encoder(schema_encoder_num_layer, schema_encoder_input_size, schema_encoder_state_size)

        # self-attention
        if self.params.use_schema_self_attention:
            self.schema2schema_attention_module = Attention(self.schema_attention_key_size, self.schema_attention_key_size, self.schema_attention_key_size)

        # utterance level attention
        if self.params.use_utterance_attention:
            self.utterance_attention_module = Attention(self.params.encoder_state_size, self.params.encoder_state_size, self.params.encoder_state_size)

        # Use attention module between input_hidden_states and schema_states
        # schema_states: self.schema_attention_key_size x len(schema)
        # input_hidden_states: self.utterance_attention_key_size x len(input)
        if params.use_encoder_attention:
            self.utterance2schema_attention_module = Attention(self.schema_attention_key_size, self.utterance_attention_key_size, self.utterance_attention_key_size)
            self.schema2utterance_attention_module = Attention(self.utterance_attention_key_size, self.schema_attention_key_size, self.schema_attention_key_size)

            new_attention_key_size = self.schema_attention_key_size + self.utterance_attention_key_size
            self.schema_attention_key_size = new_attention_key_size
            self.utterance_attention_key_size = new_attention_key_size

            if self.params.use_schema_encoder_2:
                self.schema_encoder_2 = Encoder(schema_encoder_num_layer, self.schema_attention_key_size, self.schema_attention_key_size)
                self.utterance_encoder_2 = Encoder(params.encoder_num_layers, self.utterance_attention_key_size, self.utterance_attention_key_size)

        self.token_predictor = construct_token_predictor(params,
                                                         output_vocabulary,
                                                         self.utterance_attention_key_size,
                                                         self.schema_attention_key_size,
                                                         self.final_snippet_size,
                                                         anonymizer)

        # Use schema_attention in decoder
        if params.use_schema_attention and params.use_query_attention:
            decoder_input_size = params.output_embedding_size + self.utterance_attention_key_size + self.schema_attention_key_size + params.encoder_state_size
        elif params.use_schema_attention:
            decoder_input_size = params.output_embedding_size + self.utterance_attention_key_size + self.schema_attention_key_size
        else:
            decoder_input_size = params.output_embedding_size + self.utterance_attention_key_size

        self.decoder = SequencePredictorWithSchema(params, decoder_input_size, self.output_embedder, self.column_name_token_embedder, self.token_predictor)

        self.path = os.path.join(params.samples_dir, "encodings.pt")
        self.q_enc = None
        self.q_mem = None
        self.s_mem = None

    def predict_turn(self,
                     utterance_final_state,
                     input_hidden_states,
                     schema_states,
                     max_generation_length,
                     gold_query=None,
                     snippets=None,
                     input_sequence=None,
                     previous_queries=None,
                     previous_query_states=None,
                     input_schema=None,
                     feed_gold_tokens=False,
                     training=False,
                     sampling=False):
        """ Gets a prediction for a single turn -- calls decoder and updates loss, etc.

        TODO:  this can probably be split into two methods, one that just predicts
            and another that computes the loss.
        """
        predicted_sequence = []
        fed_sequence = []
        loss = None
        token_accuracy = 0.

        if self.params.use_encoder_attention:
            schema_attention = self.utterance2schema_attention_module(torch.stack(schema_states,dim=0), input_hidden_states).vector # input_value_size x len(schema)
            utterance_attention = self.schema2utterance_attention_module(torch.stack(input_hidden_states,dim=0), schema_states).vector # schema_value_size x len(input)

            if schema_attention.dim() == 1:
                schema_attention = schema_attention.unsqueeze(1)
            if utterance_attention.dim() == 1:
                utterance_attention = utterance_attention.unsqueeze(1)

            new_schema_states = torch.cat([torch.stack(schema_states, dim=1), schema_attention], dim=0) # (input_value_size+schema_value_size) x len(schema)
            schema_states = list(torch.split(new_schema_states, split_size_or_sections=1, dim=1))
            schema_states = [schema_state.squeeze() for schema_state in schema_states]

            new_input_hidden_states = torch.cat([torch.stack(input_hidden_states, dim=1), utterance_attention], dim=0) # (input_value_size+schema_value_size) x len(input)
            input_hidden_states = list(torch.split(new_input_hidden_states, split_size_or_sections=1, dim=1))
            input_hidden_states = [input_hidden_state.squeeze() for input_hidden_state in input_hidden_states]

            # bi-lstm over schema_states and input_hidden_states (embedder is an identify function)
            if self.params.use_schema_encoder_2:
                final_schema_state, schema_states = self.schema_encoder_2(schema_states, lambda x: x, dropout_amount=self.dropout)
                final_utterance_state, input_hidden_states = self.utterance_encoder_2(input_hidden_states, lambda x: x, dropout_amount=self.dropout)

        if feed_gold_tokens:
            decoder_results = self.decoder(utterance_final_state,
                                           input_hidden_states,
                                           schema_states,
                                           max_generation_length,
                                           gold_sequence=gold_query,
                                           input_sequence=input_sequence,
                                           previous_queries=previous_queries,
                                           previous_query_states=previous_query_states,
                                           input_schema=input_schema,
                                           snippets=snippets,
                                           dropout_amount=self.dropout,
                                           sampling=sampling)

            all_scores = []
            all_alignments = []
            for prediction in decoder_results.predictions:
                scores = F.softmax(prediction.scores, dim=0)
                alignments = prediction.aligned_tokens
                if self.params.use_previous_query and self.params.use_copy_switch and len(previous_queries) > 0:
                    query_scores = F.softmax(prediction.query_scores, dim=0)
                    copy_switch = prediction.copy_switch
                    scores = torch.cat([scores * (1 - copy_switch), query_scores * copy_switch], dim=0)
                    alignments = alignments + prediction.query_tokens

                all_scores.append(scores)
                all_alignments.append(alignments)

            # Compute the loss
            gold_sequence = gold_query

            loss = torch_utils.compute_loss(gold_sequence, all_scores, all_alignments, get_token_indices)
            if not training:
                predicted_sequence = torch_utils.get_seq_from_scores(all_scores, all_alignments)
                token_accuracy = torch_utils.per_token_accuracy(gold_sequence, predicted_sequence)
            fed_sequence = gold_sequence
        else:
            decoder_results = self.decoder(utterance_final_state,
                                           input_hidden_states,
                                           schema_states,
                                           max_generation_length,
                                           input_sequence=input_sequence,
                                           previous_queries=previous_queries,
                                           previous_query_states=previous_query_states,
                                           input_schema=input_schema,
                                           snippets=snippets,
                                           dropout_amount=self.dropout,
                                           sampling=sampling)
            predicted_sequence = decoder_results.sequence
            fed_sequence = predicted_sequence

        decoder_states = [pred.decoder_state for pred in decoder_results.predictions]

        # fed_sequence contains EOS, which we don't need when encoding snippets.
        # also ignore the first state, as it contains the BEG encoding.

        for token, state in zip(fed_sequence[:-1], decoder_states[1:]):
            if snippet_handler.is_snippet(token):
                snippet_length = 0
                for snippet in snippets:
                    if snippet.name == token:
                        snippet_length = len(snippet.sequence)
                        break
                assert snippet_length > 0
                decoder_states.extend([state for _ in range(snippet_length)])
            else:
                decoder_states.append(state)

        return (predicted_sequence,
                loss,
                token_accuracy,
                decoder_states,
                decoder_results)

    def encode_schema_bow_simple(self, input_schema):
        schema_states = []
        for column_name in input_schema.column_names_embedder_input:
            schema_states.append(input_schema.column_name_embedder_bow(column_name, surface_form=False, column_name_token_embedder=self.column_name_token_embedder))
        input_schema.set_column_name_embeddings(schema_states)
        return schema_states

    def encode_schema_self_attention(self, schema_states):
        schema_self_attention = self.schema2schema_attention_module(torch.stack(schema_states,dim=0), schema_states).vector
        if schema_self_attention.dim() == 1:
            schema_self_attention = schema_self_attention.unsqueeze(1)
        residual_schema_states = list(torch.split(schema_self_attention, split_size_or_sections=1, dim=1))
        residual_schema_states = [schema_state.squeeze() for schema_state in residual_schema_states]

        new_schema_states = [schema_state+residual_schema_state for schema_state, residual_schema_state in zip(schema_states, residual_schema_states)]

        return new_schema_states

    def encode_schema(self, input_schema, dropout=False):
      schema_states = []
      for column_name_embedder_input in input_schema.column_names_embedder_input:
        tokens = column_name_embedder_input.split()

        if dropout:
          final_schema_state_one, schema_states_one = self.schema_encoder(tokens, self.column_name_token_embedder, dropout_amount=self.dropout)
        else:
          final_schema_state_one, schema_states_one = self.schema_encoder(tokens, self.column_name_token_embedder)

        # final_schema_state_one: 1 means hidden_states instead of cell_memories, -1 means last layer
        schema_states.append(final_schema_state_one[1][-1])

      input_schema.set_column_name_embeddings(schema_states)

      # self-attention over schema_states
      if self.params.use_schema_self_attention:
        schema_states = self.encode_schema_self_attention(schema_states)

      return schema_states

    def get_bert_encoding(self, input_sequence, input_schema, discourse_state, dropout):
        utterance_states, schema_token_states = utils_bert.get_bert_encoding(self.bert_config, self.model_bert, self.tokenizer, input_sequence, input_schema, bert_input_version=self.params.bert_input_version, num_out_layers_n=1, num_out_layers_h=1)

        if self.params.discourse_level_lstm:
            utterance_token_embedder = lambda x: torch.cat([x, discourse_state], dim=0)
        else:
            utterance_token_embedder = lambda x: x

        if dropout:
            final_utterance_state, utterance_states = self.utterance_encoder(
                utterance_states,
                utterance_token_embedder,
                dropout_amount=self.dropout)
        else:
            final_utterance_state, utterance_states = self.utterance_encoder(
                utterance_states,
                utterance_token_embedder)

        schema_states = []
        for schema_token_states1 in schema_token_states:
            if dropout:
                final_schema_state_one, schema_states_one = self.schema_encoder(schema_token_states1, lambda x: x, dropout_amount=self.dropout)
            else:
                final_schema_state_one, schema_states_one = self.schema_encoder(schema_token_states1, lambda x: x)

            # final_schema_state_one: 1 means hidden_states instead of cell_memories, -1 means last layer
            schema_states.append(final_schema_state_one[1][-1])

        input_schema.set_column_name_embeddings(schema_states)

        # self-attention over schema_states
        if self.params.use_schema_self_attention:
            schema_states = self.encode_schema_self_attention(schema_states)

        return final_utterance_state, utterance_states, schema_states

    def get_query_token_embedding(self, output_token, input_schema):
        if input_schema:
            if not (self.output_embedder.in_vocabulary(output_token) or input_schema.in_vocabulary(output_token, surface_form=True)):
                output_token = 'value'
            if self.output_embedder.in_vocabulary(output_token):
                output_token_embedding = self.output_embedder(output_token)
            else:
                output_token_embedding = input_schema.column_name_embedder(output_token, surface_form=True)
        else:
            output_token_embedding = self.output_embedder(output_token)
        return output_token_embedding

    def get_utterance_attention(self, final_utterance_states_c, final_utterance_states_h, final_utterance_state, num_utterances_to_keep):
        # self-attention between utterance_states            
        final_utterance_states_c.append(final_utterance_state[0][0])
        final_utterance_states_h.append(final_utterance_state[1][0])
        final_utterance_states_c = final_utterance_states_c[-num_utterances_to_keep:]
        final_utterance_states_h = final_utterance_states_h[-num_utterances_to_keep:]

        attention_result = self.utterance_attention_module(final_utterance_states_c[-1], final_utterance_states_c)
        final_utterance_state_attention_c = final_utterance_states_c[-1] + attention_result.vector.squeeze()

        attention_result = self.utterance_attention_module(final_utterance_states_h[-1], final_utterance_states_h)
        final_utterance_state_attention_h = final_utterance_states_h[-1] + attention_result.vector.squeeze()

        final_utterance_state = ([final_utterance_state_attention_c],[final_utterance_state_attention_h])

        return final_utterance_states_c, final_utterance_states_h, final_utterance_state

    def get_previous_queries(self, previous_queries, previous_query_states, previous_query, input_schema):
        previous_queries.append(previous_query)
        num_queries_to_keep = min(self.params.maximum_queries, len(previous_queries))
        previous_queries = previous_queries[-num_queries_to_keep:]

        query_token_embedder = lambda query_token: self.get_query_token_embedding(query_token, input_schema)
        _, previous_outputs = self.query_encoder(previous_query, query_token_embedder, dropout_amount=self.dropout)
        assert len(previous_outputs) == len(previous_query)
        previous_query_states.append(previous_outputs)
        previous_query_states = previous_query_states[-num_queries_to_keep:]

        return previous_queries, previous_query_states

    def forward(self, interaction, max_generation_length, sampling=False,
                forcing=False):
        """ Only for single utterance interaction (Spider). """
        input_schema = interaction.get_schema()
        if input_schema and not self.params.use_bert:
            schema_states = self.encode_schema_bow_simple(input_schema)

        try:
            utterance, = interaction.gold_utterances()
        except ValueError:
            utterance = interaction.gold_utterances()[0]
            print("Interaction contains multiple utterances.")

        input_sequence = utterance.input_sequence()

        if not self.params.use_bert:
            final_utterance_state, utterance_states = \
                self.utterance_encoder(input_sequence,
                                       self.input_embedder,
                                       dropout_amount=self.dropout)
        else:
            final_utterance_state, utterance_states, schema_states = \
                self.get_bert_encoding(input_sequence,
                                       input_schema,
                                       [],
                                       dropout=True)

        if self.params.use_encoder_attention:
            schema_attention = \
                self.utterance2schema_attention_module(
                    torch.stack(schema_states, dim=0),
                    utterance_states
                ).vector
            utterance_attention = \
                self.schema2utterance_attention_module(
                    torch.stack(utterance_states, dim=0),
                    schema_states
                ).vector

            if schema_attention.dim() == 1:
                schema_attention = schema_attention.unsqueeze(1)
            if utterance_attention.dim() == 1:
                utterance_attention = utterance_attention.unsqueeze(1)

            schema_states = torch.cat(
                [torch.stack(schema_states, dim=1),
                 schema_attention],
                dim=0)
            schema_states = list(
                torch.split(schema_states,
                            split_size_or_sections=1,
                            dim=1))
            schema_states = [schema_state.squeeze()
                             for schema_state in schema_states]

            utterance_states = torch.cat(
                [torch.stack(utterance_states, dim=1),
                 utterance_attention],
                dim=0)
            utterance_states = list(
                torch.split(utterance_states,
                            split_size_or_sections=1,
                            dim=1))
            utterance_states = [utterance_state.squeeze()
                                for utterance_state in utterance_states]

            if self.params.use_schema_encoder_2:
                _, schema_states = \
                    self.schema_encoder_2(schema_states,
                                          lambda x: x,
                                          dropout_amount=self.dropout)
                _, utterance_states = \
                    self.utterance_encoder_2(utterance_states,
                                             lambda x: x,
                                             dropout_amount=self.dropout)

        # encodings = {
        #     "q_enc": ([state.cpu().detach().numpy() for state in final_utterance_state[0]],
        #               [state.cpu().detach().numpy() for state in final_utterance_state[1]]),
        #     "q_mem": [state.cpu().detach().numpy() for state in utterance_states],
        #     "s_mem": [state.cpu().detach().numpy() for state in schema_states]
        # }

        # self.save_encodings(encodings)
        if forcing:
            sample = self.decoder(
                final_utterance_state,
                utterance_states,
                schema_states,
                max_generation_length,
                input_schema=input_schema,
                dropout_amount=self.dropout,
                gold_sequence=utterance.gold_query(),
                sampling=sampling
            )
        else:
            sample = self.decoder(
                final_utterance_state,
                utterance_states,
                schema_states,
                max_generation_length,
                input_schema=input_schema,
                dropout_amount=self.dropout,
                sampling=sampling
            )

        if self.params.consolidate:
            prob = []
            for i, prediction in enumerate(sample.predictions):
                prob_ = F.softmax(prediction.scores, dim=0)  # vocab_size
                multi_hot = torch.zeros(prob_.size(), dtype=torch.bool).cuda()
                multi_hot.scatter_(0, sample.indices[i], True)
                prob_ = torch.sum(torch.masked_select(prob_, multi_hot))
                prob.append(prob_)
            prob = torch.stack(prob).squeeze()
        else:
            scores = []
            for prediction in sample.predictions:
                scores.append(prediction.scores)
            scores = torch.stack(scores, dim=0)
            prob = F.softmax(scores, dim=1)     # seq_len x vocab_size
            one_hot = torch.zeros((scores.size()), dtype=torch.bool).cuda()
            one_hot.scatter_(1, sample.indices.data.view((-1, 1)), True)
            prob = torch.masked_select(prob, one_hot)

        # for i in range(1):
        #     for state in encodings['q_enc'][i]:
        #         del state
        # for state in encodings['q_mem']:
        #     del state
        # for state in encodings['s_mem']:
        #     del state

        torch.cuda.empty_cache()

        # prevent discriminator from classifying based on EOS
        # also, change surface form to embedder input
        sample_mod = []
        for tok in sample.sequence:
            if input_schema.in_vocabulary(tok, surface_form=True):
                sample_mod.extend(input_schema.sf_to_emb_in(tok).split())
            elif tok == EOS_TOK:
                break
            else:
                sample_mod.append(tok)

        return sample.sequence, sample_mod, prob, sample.predictions

    def save_encodings(self, encodings):
        torch.save(encodings, self.path)

    def load_encodings(self):
        encodings = torch.load(self.path)
        q_enc = encodings["q_enc"]
        q_mem = encodings["q_mem"]
        s_mem = encodings["s_mem"]

        q_enc = ([torch.Tensor(state).cuda() for state in q_enc[0]],
                 [torch.Tensor(state).cuda() for state in q_enc[1]])
        q_mem = [torch.Tensor(state).cuda() for state in q_mem]
        s_mem = [torch.Tensor(state).cuda() for state in s_mem]

        return q_enc, q_mem, s_mem

    def sample(self, max_generation_length, interaction,
               given=None, given_preds=None,
               # final_utterance_state=None,
               # utterance_states=None,
               # schema_states=None
               ):
        """ Only for single utterance interaction (Spider). """

        input_schema = interaction.get_schema()

        # if final_utterance_state is None \
        #         or utterance_states is None \
        #         or schema_states is None \
        #         or input_schema is None:
        if self.q_enc is None or self.q_mem is None or self.s_mem is None:
            if input_schema and not self.params.use_bert:
                schema_states = self.encode_schema_bow_simple(input_schema)

            try:
                utterance, = interaction.gold_utterances()
            except ValueError:
                utterance = interaction.gold_utterances()[0]
                print("Interaction contains multiple utterances.")

            input_sequence = utterance.input_sequence()

            if not self.params.use_bert:
                final_utterance_state, utterance_states = \
                    self.utterance_encoder(input_sequence,
                                           self.input_embedder,
                                           dropout_amount=self.dropout)
            else:
                final_utterance_state, utterance_states, schema_states = \
                    self.get_bert_encoding(input_sequence,
                                           input_schema,
                                           [],
                                           dropout=True)

            if self.params.use_encoder_attention:
                schema_attention = \
                    self.utterance2schema_attention_module(
                        torch.stack(schema_states, dim=0),
                        utterance_states
                    ).vector
                utterance_attention = \
                    self.schema2utterance_attention_module(
                        torch.stack(utterance_states, dim=0),
                        schema_states
                    ).vector

                if schema_attention.dim() == 1:
                    schema_attention = schema_attention.unsqueeze(1)
                if utterance_attention.dim() == 1:
                    utterance_attention = utterance_attention.unsqueeze(1)

                schema_states = torch.cat(
                    [torch.stack(schema_states, dim=1),
                     schema_attention],
                    dim=0)
                schema_states = list(
                    torch.split(schema_states,
                                split_size_or_sections=1,
                                dim=1))
                schema_states = [schema_state.squeeze()
                                 for schema_state in schema_states]

                utterance_states = torch.cat(
                    [torch.stack(utterance_states, dim=1),
                     utterance_attention],
                    dim=0)
                utterance_states = list(
                    torch.split(utterance_states,
                                split_size_or_sections=1,
                                dim=1))
                utterance_states = [utterance_state.squeeze()
                                    for utterance_state in utterance_states]

                if self.params.use_schema_encoder_2:
                    _, schema_states = \
                        self.schema_encoder_2(schema_states,
                                              lambda x: x,
                                              dropout_amount=self.dropout)
                    _, utterance_states = \
                        self.utterance_encoder_2(utterance_states,
                                                 lambda x: x,
                                                 dropout_amount=self.dropout)
            self.q_enc = ([state for state in final_utterance_state[0]],
                          [state for state in final_utterance_state[1]])
            self.q_mem = [state for state in utterance_states]
            self.s_mem = [state for state in schema_states]

        else:
            final_utterance_state = self.q_enc
            utterance_states = self.q_mem
            schema_states = self.s_mem

        # by now, we have all necessary arguments to call decoder.sample
        sample, _, prob = self.decoder.sample(
            final_utterance_state,
            utterance_states,
            schema_states,
            max_generation_length,
            given=given,
            given_preds=given_preds,
            input_schema=input_schema,
            dropout_amount=self.dropout
        )

        # del final_utterance_state
        # for state in utterance_states:
        #     del state
        # for state in schema_states:
        #     del state

        torch.cuda.empty_cache()

        # prevent discriminator from classifying based on EOS
        # also, change surface form to embedder input
        sample_mod = []
        for tok in sample:
            if input_schema.in_vocabulary(tok, surface_form=True):
                sample_mod.extend(input_schema.sf_to_emb_in(tok).split())
            elif tok == EOS_TOK:
                break
            else:
                sample_mod.append(tok)

        return sample, sample_mod, prob

    def get_reward(self, sequence, predictions, interaction,
                   roll_num, max_generation_length, discriminator,
                   bias=0., mle=False):
        try:
            utterance, = interaction.gold_utterances()
        except ValueError:
            utterance = interaction.gold_utterances()[0]
            print("Interaction contains multiple utterances.")

        utterance = utterance.dis_gold_query()

        # q_enc, q_mem, s_mem = self.load_encodings()

        # q_enc = ([torch.Tensor(state).cuda() for state in self.q_enc[0]],
        #          [torch.Tensor(state).cuda() for state in self.q_enc[1]])
        # q_mem = [torch.Tensor(state).cuda() for state in self.q_mem]
        # s_mem = [torch.Tensor(state).cuda() for state in self.s_mem]

        rewards = []
        seq_len = len(sequence)
        for i in range(roll_num):
            for l in range(1, seq_len):
                given = sequence[:l]
                given_preds = predictions[:l]
                _, sample_mod, _ = \
                    self.sample(
                        max_generation_length,
                        interaction,
                        given=given,
                        given_preds=given_preds,
                        # final_utterance_state=q_enc,
                        # utterance_states=q_mem,
                        # schema_states=s_mem
                    )
                pred = discriminator([utterance], [sample_mod]).squeeze()
                pred = torch.exp(pred).cpu().data[1].numpy()
                # if mle:
                #     pred = pred / (1. - pred)
                if i == 0:
                    rewards.append(pred)
                else:
                    rewards[l-1] += pred

            # for the last token
            pred = discriminator([utterance], [sequence]).squeeze()
            pred = torch.exp(pred).cpu().data[1].numpy()
            # if mle:
            #     pred = pred / (1. - pred)
            if i == 0:
                rewards.append(pred)
            else:
                rewards[-1] += pred

        rewards = np.array(rewards)

        if mle:
            rewards = rewards / (1.0 * roll_num)
            rewards = seq_len * rewards / np.sum(rewards)
        else:
            if bias > 0.:
                b = np.zeros_like(rewards)
                b += bias * roll_num
                rewards = (rewards - bias) / (1.0 * roll_num)
            else:
                rewards = rewards / (1.0 * roll_num)

        # del q_enc
        # for state in q_mem:
        #     del state
        # for state in s_mem:
        #     del state

        self.q_enc = None
        self.q_mem = None
        self.s_mem = None

        torch.cuda.empty_cache()
        return rewards

    def get_reward_mm(self, sequence, predictions, interaction,
                      roll_num, max_generation_length, discriminator):
        try:
            utterance, = interaction.gold_utterances()
        except ValueError:
            utterance = interaction.gold_utterances()[0]
            print("Interaction contains multiple utterances.")

        utterance = utterance.dis_gold_query()

        # q_enc, q_mem, s_mem = self.load_encodings()

        # q_enc = ([torch.Tensor(state).cuda() for state in self.q_enc[0]],
        #          [torch.Tensor(state).cuda() for state in self.q_enc[1]])
        # q_mem = [torch.Tensor(state).cuda() for state in self.q_mem]
        # s_mem = [torch.Tensor(state).cuda() for state in self.s_mem]

        rewards = []
        probs = []
        for i in range(roll_num):
            _, sample_mod, prob = \
                self.sample(
                    max_generation_length,
                    interaction,
                    given=sequence,
                    given_preds=predictions
                )
            pred = discriminator([utterance], [sample_mod]).squeeze()
            pred = torch.exp(pred).cpu().data[1].numpy()
            pred = pred / (1. - pred)
            rewards.append(pred)
            probs.append(prob)

        rewards = np.array(rewards)
        rewards = rewards / np.sum(rewards)
        rewards -= np.mean(rewards)
        probs = torch.cuda.FloatTensor(np.array(probs))

        # del q_enc
        # for state in q_mem:
        #     del state
        # for state in s_mem:
        #     del state

        self.q_enc = None
        self.q_mem = None
        self.s_mem = None

        torch.cuda.empty_cache()
        return rewards, probs

    def update_gan_loss(self, prob, rewards):
        assert prob.dim() == 1 and rewards.dim() == 1 \
               and prob.size()[0] == rewards.size()[0]
        loss = torch.log(prob) * rewards
        loss = -torch.sum(loss)

        self.trainer.zero_grad()
        if self.params.fine_tune_bert:
            self.bert_trainer.zero_grad()
        loss.backward()
        self.trainer.step()
        if self.params.fine_tune_bert:
            self.bert_trainer.step()

        loss = loss.item()

        torch.cuda.empty_cache()

        return loss

    def update_gan_loss_mm(self, prob_n, probs, rewards):
        loss = rewards * torch.log(probs)
        loss = -torch.sum(loss) + torch.sum(prob_n)

        self.trainer.zero_grad()
        if self.params.fine_tune_bert:
            self.bert_trainer.zero_grad()
        loss.backward()
        self.trainer.step()
        if self.params.fine_tune_bert:
            self.bert_trainer.step()

        loss = loss.item()

        torch.cuda.empty_cache()

        return loss

    def train_step(self, interaction, max_generation_length, snippet_alignment_probability=1.,
                   sampling=False):
        """ Trains the interaction-level model on a single interaction.

        Inputs:
            interaction (Interaction): The interaction to train on.
            learning_rate (float): Learning rate to use.
            snippet_keep_age (int): Age of oldest snippets to use.
            snippet_alignment_probability (float): The probability that a snippet will
                be used in constructing the gold sequence.
        """
        # assert self.params.discourse_level_lstm

        losses = []
        total_gold_tokens = 0

        input_hidden_states = []
        input_sequences = []

        final_utterance_states_c = []
        final_utterance_states_h = []

        previous_query_states = []
        previous_queries = []

        decoder_states = []

        discourse_state = None
        if self.params.discourse_level_lstm:
            discourse_state, discourse_lstm_states = self._initialize_discourse_states()
        discourse_states = []

        # Schema and schema embeddings
        input_schema = interaction.get_schema()
        schema_states = []

        if input_schema and not self.params.use_bert:
            schema_states = self.encode_schema_bow_simple(input_schema)

        for utterance_index, utterance in enumerate(interaction.gold_utterances()):
            if interaction.identifier in LIMITED_INTERACTIONS and utterance_index > LIMITED_INTERACTIONS[interaction.identifier]:
                break

            input_sequence = utterance.input_sequence()

            available_snippets = utterance.snippets()
            previous_query = utterance.previous_query()

            # Get the gold query: reconstruct if the alignment probability is less than one
            if snippet_alignment_probability < 1.:
                gold_query = sql_util.add_snippets_to_query(
                    available_snippets,
                    utterance.contained_entities(),
                    utterance.anonymized_gold_query(),
                    prob_align=snippet_alignment_probability) + [vocab.EOS_TOK]
            else:
                gold_query = utterance.gold_query()

            # Encode the utterance, and update the discourse-level states
            if not self.params.use_bert:
                if self.params.discourse_level_lstm:
                    utterance_token_embedder = lambda token: torch.cat([self.input_embedder(token), discourse_state], dim=0)
                else:
                    utterance_token_embedder = self.input_embedder
                final_utterance_state, utterance_states = self.utterance_encoder(
                    input_sequence,
                    utterance_token_embedder,
                    dropout_amount=self.dropout)
            else:
                final_utterance_state, utterance_states, schema_states = self.get_bert_encoding(input_sequence, input_schema, discourse_state, dropout=True)

            input_hidden_states.extend(utterance_states)
            input_sequences.append(input_sequence)

            num_utterances_to_keep = min(self.params.maximum_utterances, len(input_sequences))

            # final_utterance_state[1][0] is the first layer's hidden states at the last time step (concat forward lstm and backward lstm)
            if self.params.discourse_level_lstm:
                _, discourse_state, discourse_lstm_states = torch_utils.forward_one_multilayer(self.discourse_lstms, final_utterance_state[1][0], discourse_lstm_states, self.dropout)

            if self.params.use_utterance_attention:
                final_utterance_states_c, final_utterance_states_h, final_utterance_state = self.get_utterance_attention(final_utterance_states_c, final_utterance_states_h, final_utterance_state, num_utterances_to_keep)

            if self.params.state_positional_embeddings:
                utterance_states, flat_sequence = self._add_positional_embeddings(input_hidden_states, input_sequences)
            else:
                flat_sequence = []
                for utt in input_sequences[-num_utterances_to_keep:]:
                    flat_sequence.extend(utt)

            snippets = None
            if self.params.use_snippets:
                if self.params.previous_decoder_snippet_encoding:
                    snippets = encode_snippets_with_states(available_snippets, decoder_states)
                else:
                    snippets = self._encode_snippets(previous_query, available_snippets, input_schema)

            if self.params.use_previous_query and len(previous_query) > 0:
                previous_queries, previous_query_states = self.get_previous_queries(previous_queries, previous_query_states, previous_query, input_schema)

            if len(gold_query) <= max_generation_length and len(previous_query) <= max_generation_length:
                prediction = self.predict_turn(final_utterance_state,
                                               utterance_states,
                                               schema_states,
                                               max_generation_length,
                                               gold_query=gold_query,
                                               snippets=snippets,
                                               input_sequence=flat_sequence,
                                               previous_queries=previous_queries,
                                               previous_query_states=previous_query_states,
                                               input_schema=input_schema,
                                               feed_gold_tokens=True,
                                               training=True,
                                               sampling=sampling)
                loss = prediction[1]
                decoder_states = prediction[3]
                total_gold_tokens += len(gold_query)
                losses.append(loss)
            else:
                # Break if previous decoder snippet encoding -- because the previous
                # sequence was too long to run the decoder.
                if self.params.previous_decoder_snippet_encoding:
                    break
                continue

            torch.cuda.empty_cache()

        if losses:
            average_loss = torch.sum(torch.stack(losses)) / total_gold_tokens

            # Renormalize so the effect is normalized by the batch size.
            normalized_loss = average_loss
            if self.params.reweight_batch:
                normalized_loss = len(losses) * average_loss / float(self.params.batch_size)

            normalized_loss.backward()
            self.trainer.step()
            if self.params.fine_tune_bert:
                self.bert_trainer.step()
            self.zero_grad()

            loss_scalar = normalized_loss.item()
        else:
            loss_scalar = 0.

        return loss_scalar

    def predict_with_predicted_queries(self, interaction, max_generation_length, syntax_restrict=True):
        """ Predicts an interaction, using the predicted queries to get snippets."""
        # assert self.params.discourse_level_lstm

        syntax_restrict=False

        predictions = []

        input_hidden_states = []
        input_sequences = []

        final_utterance_states_c = []
        final_utterance_states_h = []

        previous_query_states = []
        previous_queries = []

        discourse_state = None
        if self.params.discourse_level_lstm:
            discourse_state, discourse_lstm_states = self._initialize_discourse_states()
        discourse_states = []

        # Schema and schema embeddings
        input_schema = interaction.get_schema()
        schema_states = []

        if input_schema and not self.params.use_bert:
            schema_states = self.encode_schema_bow_simple(input_schema)

        interaction.start_interaction()
        while not interaction.done():
            utterance = interaction.next_utterance()

            available_snippets = utterance.snippets()
            previous_query = utterance.previous_query()

            input_sequence = utterance.input_sequence()

            if not self.params.use_bert:
                if self.params.discourse_level_lstm:
                    utterance_token_embedder = lambda token: torch.cat([self.input_embedder(token), discourse_state], dim=0)
                else:
                    utterance_token_embedder = self.input_embedder
                final_utterance_state, utterance_states = self.utterance_encoder(
                    input_sequence,
                    utterance_token_embedder)
            else:
                final_utterance_state, utterance_states, schema_states = self.get_bert_encoding(input_sequence, input_schema, discourse_state, dropout=False)

            input_hidden_states.extend(utterance_states)
            input_sequences.append(input_sequence)

            num_utterances_to_keep = min(self.params.maximum_utterances, len(input_sequences))

            if self.params.discourse_level_lstm:
                _, discourse_state, discourse_lstm_states = torch_utils.forward_one_multilayer(self.discourse_lstms, final_utterance_state[1][0], discourse_lstm_states)

            if self.params.use_utterance_attention:
               final_utterance_states_c, final_utterance_states_h, final_utterance_state = self.get_utterance_attention(final_utterance_states_c, final_utterance_states_h, final_utterance_state, num_utterances_to_keep)

            if self.params.state_positional_embeddings:
                utterance_states, flat_sequence = self._add_positional_embeddings(input_hidden_states, input_sequences)
            else:
                flat_sequence = []
                for utt in input_sequences[-num_utterances_to_keep:]:
                    flat_sequence.extend(utt)

            snippets = None
            if self.params.use_snippets:
                snippets = self._encode_snippets(previous_query, available_snippets, input_schema)

            if self.params.use_previous_query and len(previous_query) > 0:
                previous_queries, previous_query_states = self.get_previous_queries(previous_queries, previous_query_states, previous_query, input_schema)

            results = self.predict_turn(final_utterance_state,
                                        utterance_states,
                                        schema_states,
                                        max_generation_length,
                                        input_sequence=flat_sequence,
                                        previous_queries=previous_queries,
                                        previous_query_states=previous_query_states,
                                        input_schema=input_schema,
                                        snippets=snippets)

            predicted_sequence = results[0]
            predictions.append(results)

            # Update things necessary for using predicted queries
            anonymized_sequence = utterance.remove_snippets(predicted_sequence)
            if EOS_TOK in anonymized_sequence:
                anonymized_sequence = anonymized_sequence[:-1] # Remove _EOS
            else:
                anonymized_sequence = ['select', '*', 'from', 't1']

            if not syntax_restrict:
                utterance.set_predicted_query(interaction.remove_snippets(predicted_sequence))
                if input_schema:
                    # on SParC
                    interaction.add_utterance(utterance, anonymized_sequence, previous_snippets=utterance.snippets(), simple=True)
                else:
                    # on ATIS
                    interaction.add_utterance(utterance, anonymized_sequence, previous_snippets=utterance.snippets(), simple=False)
            else:
                utterance.set_predicted_query(utterance.previous_query())
                interaction.add_utterance(utterance, utterance.previous_query(), previous_snippets=utterance.snippets())

        return predictions


    def predict_with_gold_queries(self, interaction, max_generation_length, feed_gold_query=False):
        """ Predicts SQL queries for an interaction.

        Inputs:
            interaction (Interaction): Interaction to predict for.
            feed_gold_query (bool): Whether or not to feed the gold token to the
                generation step.
        """
        # assert self.params.discourse_level_lstm

        predictions = []

        input_hidden_states = []
        input_sequences = []

        final_utterance_states_c = []
        final_utterance_states_h = []

        previous_query_states = []
        previous_queries = []

        decoder_states = []

        discourse_state = None
        if self.params.discourse_level_lstm:
            discourse_state, discourse_lstm_states = self._initialize_discourse_states()
        discourse_states = []

        # Schema and schema embeddings
        input_schema = interaction.get_schema()
        schema_states = []
        if input_schema and not self.params.use_bert:
            schema_states = self.encode_schema_bow_simple(input_schema)


        for utterance in interaction.gold_utterances():
            input_sequence = utterance.input_sequence()

            available_snippets = utterance.snippets()
            previous_query = utterance.previous_query()

            # Encode the utterance, and update the discourse-level states
            if not self.params.use_bert:
                if self.params.discourse_level_lstm:
                    utterance_token_embedder = lambda token: torch.cat([self.input_embedder(token), discourse_state], dim=0)
                else:
                    utterance_token_embedder = self.input_embedder
                final_utterance_state, utterance_states = self.utterance_encoder(
                    input_sequence,
                    utterance_token_embedder,
                    dropout_amount=self.dropout)
            else:
                final_utterance_state, utterance_states, schema_states = self.get_bert_encoding(input_sequence, input_schema, discourse_state, dropout=True)

            input_hidden_states.extend(utterance_states)
            input_sequences.append(input_sequence)

            num_utterances_to_keep = min(self.params.maximum_utterances, len(input_sequences))

            if self.params.discourse_level_lstm:
                _, discourse_state, discourse_lstm_states = torch_utils.forward_one_multilayer(self.discourse_lstms, final_utterance_state[1][0], discourse_lstm_states, self.dropout)

            if self.params.use_utterance_attention:
                final_utterance_states_c, final_utterance_states_h, final_utterance_state = self.get_utterance_attention(final_utterance_states_c, final_utterance_states_h, final_utterance_state, num_utterances_to_keep)

            if self.params.state_positional_embeddings:
                utterance_states, flat_sequence = self._add_positional_embeddings(input_hidden_states, input_sequences)
            else:
                flat_sequence = []
                for utt in input_sequences[-num_utterances_to_keep:]:
                    flat_sequence.extend(utt)

            snippets = None
            if self.params.use_snippets:
                if self.params.previous_decoder_snippet_encoding:
                    snippets = encode_snippets_with_states(available_snippets, decoder_states)
                else:
                    snippets = self._encode_snippets(previous_query, available_snippets, input_schema)

            if self.params.use_previous_query and len(previous_query) > 0:
                previous_queries, previous_query_states = self.get_previous_queries(previous_queries, previous_query_states, previous_query, input_schema)

            prediction = self.predict_turn(final_utterance_state,
                                           utterance_states,
                                           schema_states,
                                           max_generation_length,
                                           gold_query=utterance.gold_query(),
                                           snippets=snippets,
                                           input_sequence=flat_sequence,
                                           previous_queries=previous_queries,
                                           previous_query_states=previous_query_states,
                                           input_schema=input_schema,
                                           feed_gold_tokens=feed_gold_query)

            decoder_states = prediction[3]
            predictions.append(prediction)

        return predictions
