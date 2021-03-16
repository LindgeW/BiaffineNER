import torch.nn as nn
from modules.rnn import LSTM
from modules.dropout import *
import torch.nn.functional as F
from modules.BertModel import BertEmbedding


class BertSpanTagger(nn.Module):
    def __init__(self, bert_embed_dim, hidden_size, num_rnn_layer,
                 num_tag, num_bert_layer=4,
                 dropout=0.0, bert_model_path=None):
        super(BertSpanTagger, self).__init__()
        self.bert_embed_dim = bert_embed_dim
        self.dropout = dropout
        self.num_tag = num_tag
        self.bert = BertEmbedding(bert_model_path, num_bert_layer, use_proj=False)
        self.bert_fc = nn.Linear(self.bert.hidden_size, self.bert_embed_dim)
        nn.init.xavier_uniform_(self.bert_fc.weight)

        self.bert_norm = nn.LayerNorm(self.bert_embed_dim, eps=1e-6)

        self.seq_encoder = LSTM(input_size=self.bert_embed_dim,
                                hidden_size=hidden_size,
                                num_layers=num_rnn_layer,
                                dropout=dropout)

        self.hidden2start = nn.Linear(2 * hidden_size, num_tag)

        self.hidden2end = nn.Sequential(
            nn.Linear(2 * hidden_size + num_tag, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size, eps=1e-6),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_tag))

    def bert_named_params(self):
        return self.bert.named_parameters()

    def non_bert_params(self):
        bert_param_names = set()
        for name, param in self.bert.named_parameters():
            if param.requires_grad:
                bert_param_names.add(id(param))

        other_params = []
        for name, param in self.named_parameters():
            if param.requires_grad and id(param) not in bert_param_names:
                other_params.append(param)
        return other_params

    def forward(self, bert_inps, start_pos=None, end_pos=None, mask=None):
        '''
        :param bert_inps: bert_ids, segments, bert_masks, bert_lens
        :param start_pos: (bs, seq_len)
        :param end_pos: (bs, seq_len)
        :param mask: (bs, seq_len)  0 for padding
        :return:
        '''
        bert_embed = self.bert(*bert_inps)
        bert_repr = self.bert_norm(self.bert_fc(bert_embed))
        if self.training:
            bert_repr = timestep_dropout(bert_repr, p=self.dropout)

        enc_out = self.seq_encoder(bert_repr, non_pad_mask=mask)[0]

        if self.training:
            enc_out = timestep_dropout(enc_out, p=self.dropout)

        start_logit = self.hidden2start(enc_out)
        if start_pos is not None and self.training:
            bs, seq_len = start_pos.size()[:2]
            tag_logit = torch.zeros((bs, seq_len, self.num_tag), dtype=torch.float32, device=start_pos.device)
            tag_logit.scatter_(2, start_pos.unsqueeze(2), 1)
        else:
            tag_logit = F.softmax(start_logit, dim=-1)
        end_logit = self.hidden2end(torch.cat((enc_out, tag_logit), dim=-1).contiguous())
        outputs = (start_logit, end_logit)

        if start_pos is not None and end_pos is not None:
            start_loss = F.cross_entropy(start_logit[mask], start_pos[mask])
            end_loss = F.cross_entropy(end_logit[mask], end_pos[mask])
            loss = start_loss + end_loss
            outputs = (loss, ) + outputs

        return outputs



