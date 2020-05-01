import torch
import torch.nn as nn
import torch.nn.functional as F

from data_util.vocabulary import CLS_TOK, SEP_TOK
from model.embedder import DiscriminatorEmbedding
from model.model import load_word_embeddings


class Discriminator(nn.Module):
    def __init__(self, params, src_vocab, tgt_vocab, max_len,
                 num_classes, filter_sizes, num_filters,
                 max_pos_emb=0, num_tok_type=0,
                 dropout_amount=0.):
        super(Discriminator, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

        src_emb, tgt_emb, _, emb_size = load_word_embeddings(
            src_vocab,
            tgt_vocab,
            None,
            params)

        self.emb_size = emb_size
        self.src_emb = DiscriminatorEmbedding(self.emb_size,
                                              initializer=src_emb,
                                              vocabulary=src_vocab,
                                              max_pos_emb=max_pos_emb,
                                              num_token_type=num_tok_type)
        self.tgt_emb = DiscriminatorEmbedding(self.emb_size,
                                              initializer=tgt_emb,
                                              vocabulary=tgt_vocab)

        # src convolutions
        self.src_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, n, (f, self.emb_size)),
                          nn.BatchNorm2d(n))
            for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.src_highway = nn.Linear(sum(num_filters), sum(num_filters))

        # tgt convolutions
        self.tgt_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, n, (f, self.emb_size)),
                          nn.BatchNorm2d(n))
            for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.tgt_highway = nn.Linear(sum(num_filters), sum(num_filters))

        self.dropout = nn.Dropout(p=dropout_amount)

        # prediction
        self.linear = nn.Linear(sum(num_filters) * 2, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, src, tgt):
        """
        Args:
            src (List[List[str]])
            tgt (List[List[str]])

        Returns:
            pred (FloatTensor): 2 dim

        """

        # # padding may induce nondeterministic behavior in its backward pass
        # src_pad_num = self.max_len - len(src)
        # tgt_pad_num = self.max_len - len(tgt)

        src_emb, src_max_len = self.src_emb(src)    # batch x src_len x emb_dim
        src_pad_num = self.max_len - src_max_len
        src_emb_pad = F.pad(src_emb.unsqueeze(1), [0, 0, 0, src_pad_num]) # batch x 1 x max_len x emb_dim
        src_convs = [F.relu(conv(src_emb_pad)).squeeze(3)
                     for conv in self.src_convs]    # batch x num_filter x src_len
        src_pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2)
                     for conv in src_convs]         # batch x num_filter
        src_pred = torch.cat(src_pools, dim=1)      # batch x sum(num_filter)
        src_highway = self.src_highway(src_pred)    # batch x sum(num_filter)
        src_pred = torch.sigmoid(src_highway) * F.relu(src_highway) \
            + (1. - torch.sigmoid(src_highway)) * src_pred

        tgt_emb, tgt_max_len = self.tgt_emb(tgt)
        tgt_pad_num = self.max_len - tgt_max_len
        tgt_emb_pad = F.pad(tgt_emb.unsqueeze(1), [0, 0, 0, tgt_pad_num])
        tgt_convs = [F.relu(conv(tgt_emb_pad)).squeeze(3)
                     for conv in self.tgt_convs]
        tgt_pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2)
                     for conv in tgt_convs]
        tgt_pred = torch.cat(tgt_pools, dim=1)
        tgt_highway = self.tgt_highway(tgt_pred)
        tgt_pred = torch.sigmoid(tgt_highway) * F.relu(tgt_highway) \
            + (1. - torch.sigmoid(tgt_highway)) * tgt_pred

        # input: batch x 2 * sum(num_filter)
        # output: batch x num_classes
        scores = self.linear(torch.cat([self.dropout(src_pred),
                                        self.dropout(tgt_pred)], dim=1))
        pred = self.softmax(scores)

        torch.cuda.empty_cache()

        return pred     # batch x num_classes


class SchemaDiscriminator(nn.Module):
    def __init__(self, params, src_vocab, sch_vocab, tgt_vocab, max_len,
                 num_classes, filter_sizes, num_filters,
                 max_pos_emb=0, num_tok_type=0,
                 dropout_amount=0.):
        super(SchemaDiscriminator, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.sch_vocab = sch_vocab
        self.max_len = max_len

        src_emb, tgt_emb, sch_emb, emb_size = load_word_embeddings(
            src_vocab,
            tgt_vocab,
            sch_vocab,
            params)

        self.emb_size = emb_size
        self.src_emb = DiscriminatorEmbedding(self.emb_size,
                                              initializer=src_emb,
                                              vocabulary=src_vocab,
                                              max_pos_emb=max_pos_emb,
                                              num_token_type=num_tok_type)
        self.sch_emb = DiscriminatorEmbedding(self.emb_size,
                                              initializer=sch_emb,
                                              vocabulary=sch_vocab)
        self.tgt_emb = DiscriminatorEmbedding(self.emb_size,
                                              initializer=tgt_emb,
                                              vocabulary=tgt_vocab)

        # src convolutions
        self.src_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, n, (f, self.emb_size)),
                          nn.BatchNorm2d(n))
            for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.src_highway = nn.Linear(sum(num_filters), sum(num_filters))

        # src convolutions
        self.sch_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, n, (f, self.emb_size)),
                          nn.BatchNorm2d(n))
            for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.sch_highway = nn.Linear(sum(num_filters), sum(num_filters))

        # tgt convolutions
        self.tgt_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, n, (f, self.emb_size)),
                          nn.BatchNorm2d(n))
            for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.tgt_highway = nn.Linear(sum(num_filters), sum(num_filters))

        self.dropout = nn.Dropout(p=dropout_amount)

        # prediction
        self.linear = nn.Linear(sum(num_filters) * 3, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, src, sch, tgt):
        """
        Args:
            src (List[List[str]])
            sch (List[List[str]])
            tgt (List[List[str]])

        Returns:
            pred (FloatTensor): 2 dim

        """

        # # padding may induce nondeterministic behavior in its backward pass
        # src_pad_num = self.max_len - len(src)
        # tgt_pad_num = self.max_len - len(tgt)

        src_emb, src_max_len = self.src_emb(src)    # batch x src_len x emb_dim
        src_pad_num = self.max_len - src_max_len
        src_emb_pad = F.pad(src_emb.unsqueeze(1), [0, 0, 0, src_pad_num]) # batch x 1 x max_len x emb_dim
        src_convs = [F.relu(conv(src_emb_pad)).squeeze(3)
                     for conv in self.src_convs]    # batch x num_filter x src_len
        src_pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2)
                     for conv in src_convs]         # batch x num_filter
        src_pred = torch.cat(src_pools, dim=1)      # batch x sum(num_filter)
        src_highway = self.src_highway(src_pred)    # batch x sum(num_filter)
        src_pred = torch.sigmoid(src_highway) * F.relu(src_highway) \
            + (1. - torch.sigmoid(src_highway)) * src_pred

        sch_emb, sch_max_len = self.sch_emb(sch)
        sch_pad_num = self.max_len - sch_max_len
        sch_emb_pad = F.pad(sch_emb.unsqueeze(1), [0, 0, 0, sch_pad_num])
        sch_convs = [F.relu(conv(sch_emb_pad)).squeeze(3)
                     for conv in self.sch_convs]
        sch_pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2)
                     for conv in sch_convs]
        sch_pred = torch.cat(sch_pools, dim=1)
        sch_highway = self.sch_highway(sch_pred)
        sch_pred = torch.sigmoid(sch_highway) * F.relu(sch_highway) \
            + (1. - torch.sigmoid(sch_highway)) * sch_pred

        tgt_emb, tgt_max_len = self.tgt_emb(tgt)
        tgt_pad_num = self.max_len - tgt_max_len
        tgt_emb_pad = F.pad(tgt_emb.unsqueeze(1), [0, 0, 0, tgt_pad_num])
        tgt_convs = [F.relu(conv(tgt_emb_pad)).squeeze(3)
                     for conv in self.tgt_convs]
        tgt_pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2)
                     for conv in tgt_convs]
        tgt_pred = torch.cat(tgt_pools, dim=1)
        tgt_highway = self.tgt_highway(tgt_pred)
        tgt_pred = torch.sigmoid(tgt_highway) * F.relu(tgt_highway) \
            + (1. - torch.sigmoid(tgt_highway)) * tgt_pred

        # input: batch x 2 * sum(num_filter)
        # output: batch x num_classes
        scores = self.linear(torch.cat([self.dropout(src_pred),
                                        self.dropout(sch_pred),
                                        self.dropout(tgt_pred)], dim=1))
        pred = self.softmax(scores)

        torch.cuda.empty_cache()

        return pred     # batch x num_classes

