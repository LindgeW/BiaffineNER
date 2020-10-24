import torch.nn as nn
from modules.layers import NonlinearMLP, Biaffine
from modules.dropout import *


class BiaffineScorer(nn.Module):
    def __init__(self, input_size,
                 ffnn_size,
                 num_cls,
                 ffnn_drop=0.33):
        super(BiaffineScorer, self).__init__()

        self.ffnn_size = ffnn_size
        self.ffnn_drop = ffnn_drop

        self._act = nn.ELU()
        self.mlp_start = NonlinearMLP(in_feature=input_size, out_feature=ffnn_size, activation=self._act)
        self.mlp_end = NonlinearMLP(in_feature=input_size, out_feature=ffnn_size, activation=self._act)
        self.span_biaff = Biaffine(ffnn_size, num_cls, bias=(True, True))

    def forward(self, enc_hn):
        start_feat = self.mlp_start(enc_hn)
        end_feat = self.mlp_end(enc_hn)

        if self.training:
            start_feat = timestep_dropout(start_feat, self.ffnn_drop)
            end_feat = timestep_dropout(end_feat, self.ffnn_drop)

        # (bz, len, len, num_lbl)
        span_score = self.span_biaff(start_feat, end_feat)
        return span_score
