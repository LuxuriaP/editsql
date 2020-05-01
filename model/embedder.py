""" Embedder for tokens. """

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import data_util.snippets as snippet_handler
import data_util.vocabulary as vocabulary_handler


class LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).

        From BERT
        """
        super(LayerNorm, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(hidden_size))
        self.beta = torch.nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        # x = input to the neuron.
        # normalize each vector (each token).
        # regularize x.
        # If x follows Gaussian distribution, it becomes standard Normal distribution (i.e., mu=0, std=1).
        u = x.mean(-1, keepdim=True) # keepdim = keeprank of tensor.
        s = (x - u).pow(2).mean(-1, keepdim=True) # variance
        x = (x - u) / torch.sqrt(s + self.variance_epsilon) # standard

        # Gamma & Beta is trainable parameters.
        return self.gamma * x + self.beta


class Embedder(torch.nn.Module):
    """ Embeds tokens. """
    def __init__(self, embedding_size, name="", initializer=None,
                 vocabulary=None, num_tokens=-1, anonymizer=None,
                 freeze=False, use_unk=True):
        super().__init__()

        if vocabulary:
            assert num_tokens < 0, "Specified a vocabulary but also set number of tokens to " + \
                str(num_tokens)
            self.in_vocabulary = lambda token: token in vocabulary.tokens
            self.vocab_token_lookup = lambda token: vocabulary.token_to_id(token)
            if use_unk:
                self.unknown_token_id = vocabulary.token_to_id(vocabulary_handler.UNK_TOK)
            else:
                self.unknown_token_id = -1
            self.vocabulary_size = len(vocabulary)
        else:
            def check_vocab(index):
                """ Makes sure the index is in the vocabulary."""
                assert index < num_tokens, "Passed token ID " + \
                    str(index) + "; expecting something less than " + str(num_tokens)
                return index < num_tokens
            self.in_vocabulary = check_vocab
            self.vocab_token_lookup = lambda x: x
            self.unknown_token_id = num_tokens  # Deliberately throws an error here,
            # But should crash before this
            self.vocabulary_size = num_tokens

        self.anonymizer = anonymizer

        emb_name = name + "-tokens"
        print("Creating token embedder called " + emb_name + " of size " + str(self.vocabulary_size) + " x " + str(embedding_size))

        if initializer is not None:
            word_embeddings_tensor = torch.FloatTensor(initializer)
            self.token_embedding_matrix = torch.nn.Embedding.from_pretrained(word_embeddings_tensor, freeze=freeze)
        else:
            init_tensor = torch.empty(self.vocabulary_size, embedding_size).uniform_(-0.1, 0.1)
            self.token_embedding_matrix = torch.nn.Embedding.from_pretrained(init_tensor, freeze=False)

        if self.anonymizer:
            emb_name = name + "-entities"
            entity_size = len(self.anonymizer.entity_types)
            print("Creating entity embedder called " + emb_name + " of size " + str(entity_size) + " x " + str(embedding_size))
            init_tensor = torch.empty(entity_size, embedding_size).uniform_(-0.1, 0.1)
            self.entity_embedding_matrix = torch.nn.Embedding.from_pretrained(init_tensor, freeze=False)

    def forward(self, token):
        assert isinstance(token, int) or not snippet_handler.is_snippet(token), "embedder should only be called on flat tokens; use snippet_bow if you are trying to encode snippets"

        if self.in_vocabulary(token):
            index_list = torch.LongTensor([self.vocab_token_lookup(token)])
            if self.token_embedding_matrix.weight.is_cuda:
                index_list = index_list.cuda()
            return self.token_embedding_matrix(index_list).squeeze()
        elif self.anonymizer and self.anonymizer.is_anon_tok(token):
            index_list = torch.LongTensor([self.anonymizer.get_anon_id(token)])
            if self.token_embedding_matrix.weight.is_cuda:
                index_list = index_list.cuda()
            return self.entity_embedding_matrix(index_list).squeeze()
        else:
            index_list = torch.LongTensor([self.unknown_token_id])
            if self.token_embedding_matrix.weight.is_cuda:
                index_list = index_list.cuda()
            return self.token_embedding_matrix(index_list).squeeze()


class DiscriminatorEmbedding(torch.nn.Module):
    def __init__(self, embedding_size, name="", initializer=None,
                 vocabulary=None, num_tokens=-1, freeze=False,
                 use_unk=True, max_pos_emb=0, num_token_type=0):
        super().__init__()

        if vocabulary:
            assert num_tokens < 0, "Specified a vocabulary but also set number of tokens to " + \
                str(num_tokens)
            self.in_vocabulary = lambda token: token in vocabulary.tokens
            self.vocab_token_lookup = lambda token: vocabulary.token_to_id(token)
            if use_unk:
                self.unknown_token_id = vocabulary.token_to_id(vocabulary_handler.UNK_TOK)
            else:
                self.unknown_token_id = -1
            self.vocabulary_size = len(vocabulary)
        else:
            def check_vocab(index):
                """ Makes sure the index is in the vocabulary."""
                assert index < num_tokens, "Passed token ID " + \
                    str(index) + "; expecting something less than " + str(num_tokens)
                return index < num_tokens
            self.in_vocabulary = check_vocab
            self.vocab_token_lookup = lambda x: x
            self.unknown_token_id = num_tokens  # Deliberately throws an error here,
            # But should crash before this
            self.vocabulary_size = num_tokens

        emb_name = name + "-tokens"
        print("Creating token embedder called " + emb_name + " of size " + str(self.vocabulary_size) + " x " + str(embedding_size))

        if initializer is not None:
            word_embeddings_tensor = torch.FloatTensor(initializer)
            self.token_embedding_matrix = torch.nn.Embedding.from_pretrained(word_embeddings_tensor, freeze=freeze)
        else:
            init_tensor = torch.empty(self.vocabulary_size, embedding_size).uniform_(-0.1, 0.1)
            self.token_embedding_matrix = torch.nn.Embedding.from_pretrained(init_tensor, freeze=False)

        self.positional_embedding = torch.nn.Embedding(max_pos_emb, embedding_size) if max_pos_emb > 0 else None
        self.token_type_embedding = torch.nn.Embedding(num_token_type, embedding_size) if num_token_type > 0 else None
        self.layer_norm = LayerNorm(embedding_size) if max_pos_emb > 0 and num_token_type > 0 else None

    def forward(self, sequences, token_type_ids=None):
        input_ids = []
        max_len = 0

        for sequence in sequences:
            indices = []
            for token in sequence:
                indices.append(self.vocab_token_lookup(token) if self.in_vocabulary(token) else self.unknown_token_id)
            if len(indices) > max_len:
                max_len = len(indices)
            if self.token_embedding_matrix.weight.is_cuda:
                indices = torch.LongTensor(indices).cuda()
            else:
                input_ids = torch.LongTensor(indices)
            input_ids.append(indices)

        input_ids = pad_sequence(input_ids, batch_first=True)   # batch x seq_len

        embeddings = self.token_embedding_matrix(input_ids)     # batch x seq_len x emb_dim

        if self.positional_embedding:
            position_ids = torch.arange(input_ids.size(0), dtype=torch.long, device=input_ids.device)
            pos_emb = self.positional_embedding(position_ids)
            embeddings = embeddings + pos_emb

        if self.token_type_embedding:
            token_type_ids = torch.LongTensor(token_type_ids) if token_type_ids else torch.zeros_like(input_ids)
            tok_type_emb = self.token_type_embedding(token_type_ids)
            embeddings = embeddings + tok_type_emb

        return embeddings, max_len


def bow_snippets(token, snippets, output_embedder, input_schema):
    """ Bag of words embedding for snippets"""
    assert snippet_handler.is_snippet(token) and snippets

    snippet_sequence = []
    for snippet in snippets:
        if snippet.name == token:
            snippet_sequence = snippet.sequence
            break
    assert snippet_sequence

    if input_schema:
        snippet_embeddings = []
        for output_token in snippet_sequence:
            assert output_embedder.in_vocabulary(output_token) or input_schema.in_vocabulary(output_token, surface_form=True)
            if output_embedder.in_vocabulary(output_token):
                snippet_embeddings.append(output_embedder(output_token))
            else:
                snippet_embeddings.append(input_schema.column_name_embedder(output_token, surface_form=True))
    else:
        snippet_embeddings = [output_embedder(subtoken) for subtoken in snippet_sequence]

    snippet_embeddings = torch.stack(snippet_embeddings, dim=0) # len(snippet_sequence) x emb_size
    return torch.mean(snippet_embeddings, dim=0) # emb_size

