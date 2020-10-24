from transformers.modeling_bert import *
from transformers import BertModel
from modules.scale_mix import ScalarMix


class BertEmbedding(nn.Module):
    def __init__(self, model_path, nb_layers=4, merge='mean'):
        super(BertEmbedding, self).__init__()
        assert merge in ['none', 'linear', 'mean']
        self.merge = merge
        self.bert = BertModel.from_pretrained(model_path, output_hidden_states=True)
        self.bert_layers = self.bert.config.num_hidden_layers
        self.nb_layers = nb_layers if nb_layers < self.bert_layers else self.bert_layers
        self.hidden_size = self.bert.config.hidden_size
        self.scale = ScalarMix(self.nb_layers)
        for n, p in self.bert.named_parameters():
            p.requires_grad = False

    def save_bert(self, save_dir):
        assert os.path.isdir(save_dir)
        self.bert.save_pretrained(save_dir)
        print('BERT Saved !!!')

    def forward(self, bert_ids, segments, bert_mask, bert_lens):
        '''
        :param bert_ids: (bz, bpe_seq_len) subword indexs
        :param segments: (bz, bpe_seq_len)  只有一个句子，全0
        :param bert_mask: (bz, bep_seq_len)  经过bpe切词
        :param bert_lens: (bz, seq_len)  每个token经过bpe切词后的长度
        :return:
        '''
        bz, seq_len = bert_lens.shape
        mask = bert_lens.gt(0)
        # bert_mask = bert_mask.type(torch.ByteTensor)
        bert_mask = bert_mask.type_as(mask)

        self.bert.eval()
        with torch.no_grad():
            last_enc_out, _, all_enc_outs = self.bert(bert_ids, token_type_ids=segments, attention_mask=bert_mask)
            top_enc_outs = all_enc_outs[-self.nb_layers:]

        if self.merge == 'linear':
            bert_out = self.scale(top_enc_outs)  # (bz, seq_len, 768)
        elif self.merge == 'mean':
            bert_out = sum(top_enc_outs) / len(top_enc_outs)
            # bert_out = torch.stack(tuple(top_enc_outs), dim=0).mean(0)
        else:
            # bert_out = last_enc_out
            bert_out = top_enc_outs[-2]

        # 根据bert piece长度切分
        bert_chunks = bert_out[bert_mask].split(bert_lens[mask].tolist())
        bert_out = torch.stack(tuple([bc.mean(0) for bc in bert_chunks[1:-1]]))  # excluding CLS and SEP
        bert_embed = bert_out.new_zeros(bz, seq_len-2, self.hidden_size)
        # 将bert_embed中mask对应1的位置替换成bert_out，0的位置不变
        mask[torch.arange(bz), mask.sum(dim=-1) - 1] = 0
        mask = mask[:, 1:-1]
        bert_embed = bert_embed.masked_scatter_(mask.unsqueeze(dim=-1), bert_out)
        return bert_embed  # (bz, seq_len, hidden_dim)
