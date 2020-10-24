import torch.nn as nn
from modules.rnn import LSTM
from modules.layers import Embeddings, CharCNNEmbedding
from modules.BertModel_fixed import BertEmbedding
from modules.dropout import *
from modules.crf import CRF
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


class SeqNERModel(nn.Module):
    def __init__(self, num_wds, num_chars, num_tags,
                 wd_embed_dim, char_embed_dim, tag_embed_dim, bert_embed_dim,
                 hidden_size, num_rnn_layer, num_lbl, bert_path, num_bert_layer=4,
                 dropout=0.0, embed_weight=None):
        super(SeqNERModel, self).__init__()
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

        self.hidden2tag = nn.Linear(2*hidden_size, num_lbl)
        self.tag_crf = CRF(num_tags=num_lbl, batch_first=True)
        self.reset_params()

    def reset_params(self):
        nn.init.xavier_uniform_(self.bert_fc.weight)
        nn.init.xavier_uniform_(self.hidden2tag.weight)

    def forward(self, wd_ids, char_ids, tag_ids, bert_inps):
        # bert_inps: bert_ids, segments, bert_mask, bert_lens
        seq_mask = wd_ids.gt(0)
        bert_embed = self.bert_norm(self.bert_fc(self.bert_embedding(*bert_inps)))
        wd_embed = self.word_embedding(wd_ids)
        char_embed = self.char_embedding(char_ids)
        tag_embed = self.tag_embedding(tag_ids)
        enc_inp = torch.cat((bert_embed, wd_embed, char_embed, tag_embed), dim=-1).contiguous()

        if self.training:
            enc_inp = timestep_dropout(enc_inp, self.dropout)

        enc_out = self.encoder(enc_inp, non_pad_mask=seq_mask)[0]

        if self.training:
            enc_out = timestep_dropout(enc_out, p=self.dropout)

        tag_score = self.hidden2tag(enc_out)

        return tag_score

    def tag_loss(self, tag_score, gold_tags, mask=None, alg='crf'):
        '''
        :param tag_score: (b, t, nb_cls)
        :param gold_tags: (b, t)
        :param mask: (b, t)  1对应有效部分，0对应pad
        :param alg: 'greedy' and 'crf'
        :return:
        '''
        assert alg in ['greedy', 'crf']
        if alg == 'crf':
            lld = self.tag_crf(tag_score, tags=gold_tags, mask=mask)
            return lld.neg()
        else:
            sum_loss = F.cross_entropy(tag_score.transpose(1, 2), gold_tags, ignore_index=0, reduction='sum')
            return sum_loss / mask.sum()

    def tag_decode(self, tag_score, mask=None, alg='crf'):
        '''
        :param tag_score: (b, t, nb_cls)  emission probs
        :param mask: (b, t)  1对应有效部分，0对应pad
        :param alg:
        :return:
        '''
        assert alg in ['greedy', 'crf']
        if alg == 'crf':
            best_tag_seq = self.tag_crf.decode(tag_score, mask=mask)
            # return best segment tags
            return pad_sequence(best_tag_seq, batch_first=True, padding_value=0)
        else:
            return tag_score.data.argmax(dim=-1) * mask.long()

