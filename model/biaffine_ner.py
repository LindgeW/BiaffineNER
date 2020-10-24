import torch.nn as nn
from modules.rnn import LSTM
from model.biaffine_model import BiaffineScorer
from modules.layers import Embeddings, CharCNNEmbedding
from modules.BertModel_fixed import BertEmbedding
from modules.dropout import *


class NERParser(nn.Module):
    def __init__(self, num_wds, num_chars, num_tags,
                 wd_embed_dim, char_embed_dim, tag_embed_dim, bert_embed_dim,
                 hidden_size, num_rnn_layer, ffnn_size, num_lbl, bert_path, num_bert_layer=4,
                 ffnn_drop=0.0, dropout=0.0, embed_weight=None):
        super(NERParser, self).__init__()
        self.dropout = dropout
        self.word_embedding = Embeddings(num_embeddings=num_wds,
                                         embedding_dim=wd_embed_dim,
                                         embed_weight=embed_weight,
                                         pad_idx=0)

        self.bert_embedding = BertEmbedding(bert_path, num_bert_layer)
        self.bert_fc = nn.Linear(self.bert_embedding.hidden_size, bert_embed_dim, bias=False)
        self.bert_norm = nn.LayerNorm(bert_embed_dim, eps=1e-6)

        self.char_embedding = CharCNNEmbedding(num_chars, char_embed_dim)

        self.tag_embedding = Embeddings(num_embeddings=num_tags,
                                        embedding_dim=tag_embed_dim,
                                        pad_idx=0)

        inp_size = wd_embed_dim + char_embed_dim + tag_embed_dim + bert_embed_dim
        self.encoder = LSTM(input_size=inp_size,
                            hidden_size=hidden_size,
                            num_layers=num_rnn_layer,
                            dropout=dropout)

        self.biaff_scorer = BiaffineScorer(2*hidden_size, ffnn_size=ffnn_size, num_cls=num_lbl, ffnn_drop=ffnn_drop)
        self.reset_params()

    def reset_params(self):
        nn.init.xavier_uniform_(self.bert_fc.weight)

    def forward(self, wd_ids, char_ids, tag_ids, bert_inps):
        # bert_inps: bert_ids, segments, bert_mask, bert_lens
        seq_mask = wd_ids.gt(0)
        bert_embed = self.bert_norm(self.bert_fc(self.bert_embedding(*bert_inps)))
        wd_embed = self.word_embedding(wd_ids)
        char_embed = self.char_embedding(char_ids)
        tag_embed = self.tag_embedding(tag_ids)
        enc_inp = torch.cat((bert_embed, wd_embed, char_embed, tag_embed), dim=-1).contiguous()

        if self.training:
            enc_inp = timestep_dropout(enc_inp, p=self.dropout)

        enc_out = self.encoder(enc_inp, non_pad_mask=seq_mask)[0]

        if self.training:
            enc_out = timestep_dropout(enc_out, p=self.dropout)

        # (bs, len, len, nb_cls)
        span_score = self.biaff_scorer(enc_out)

        return span_score


