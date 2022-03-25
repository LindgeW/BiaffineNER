import os
from utils.instance import Instance
from utils.vocab import Vocab, MultiVocab, BERTVocab
import torch
import collections
from utils.tag_util import extract_ner_bio_span


def read_insts(file_reader):
    insts = []
    for line in file_reader:
        try:
            tokens = line.strip().split()
            if line.strip() == '':
                if len(insts) > 0:
                    yield insts
                insts = []
            elif len(tokens) == 2:
                insts.append(Instance(*tokens))
        except Exception as e:
            print('exception occur: ', e)

    if len(insts) > 0:
        yield insts


def load_data(path):
    assert os.path.exists(path)
    dataset = []
    too_long = 0
    with open(path, 'r', encoding='utf-8', errors='ignore') as fr:
        for insts in read_insts(fr):
            if len(insts) < 512:
                dataset.append(insts)
            else:
                too_long += 1
    print(f'{too_long} sentences exceeds 512 tokens')
    return dataset


def create_vocab(data_sets, bert_model_path=None, embed_file=None):
    bert_vocab = BERTVocab(bert_model_path)
    # char_vocab = Vocab(min_count, bos=None, eos=None)
    ner_tag_vocab = Vocab(bos=None, eos=None)
    for insts in data_sets:
        for inst in insts:
            # char_vocab.add(inst.char)
            if inst.ner_tag != 'O':
                ner_tag = inst.ner_tag.split('-')[1]
                ner_tag_vocab.add(ner_tag)

    # if embed_file is not None:
    #     embed_count = char_vocab.load_embeddings(embed_file)
    #     print("%d word pre-trained embeddings loaded..." % embed_count)

    return MultiVocab(dict(
        # char=char_vocab,
        ner=ner_tag_vocab,
        bert=bert_vocab
    ))


def extract_ner_spans(start_logit, end_logit, mask):
    '''
    :param start_logit: (b, t, c)
    :param end_logit: (b, t, c)
    :param mask: (b, t)  0 for padding
    :return:
    '''
    start_pred = start_logit.argmax(dim=-1).detach().cpu().numpy()  # (b, t)
    end_pred = end_logit.argmax(dim=-1).detach().cpu().numpy()
    pred_spans = []
    lens = mask.sum(dim=1).tolist()
    for start_ids, end_ids, l in zip(start_pred, end_pred, lens):
        res = []
        for i, s in enumerate(start_ids[:l]):
            if s == 0:
                continue
            for j, e in enumerate(end_ids[i:l]):
                if s == e:
                    res.append((i, i+j, s))
                    break
        pred_spans.append(res)

    return pred_spans


def batch_variable(batch_data, mVocab):
    batch_size = len(batch_data)
    max_seq_len = 1 + max(len(insts) for insts in batch_data)

    bert_vocab = mVocab['bert']
    ner_tag_vocab = mVocab['ner']

    start_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    end_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)

    chars = []
    batch_spans = []
    for i, insts in enumerate(batch_data):
        seq_len = len(insts) + 1
        chars.append([inst.char for inst in insts])
        mask[i, :seq_len].fill_(1)

        tag_spans = extract_ner_bio_span(['O'] + [inst.ner_tag for inst in insts])
        gold_span = []
        for s, e, tag in tag_spans:
            tag_idx = ner_tag_vocab.inst2idx(tag)
            start_ids[i][s] = tag_idx
            end_ids[i][e] = tag_idx
            gold_span.append((s, e, tag_idx))
        batch_spans.append(gold_span)

    bert_inp = bert_vocab.batch_bertwd2id(chars)
    return Batch(bert_inp=bert_inp,
                 start_ids=start_ids,
                 end_ids=end_ids,
                 batch_spans=batch_spans,
                 mask=mask)


class Batch:
    def __init__(self, **args):
        for prop, v in args.items():
            setattr(self, prop, v)

    def to_device(self, device):
        for prop, val in self.__dict__.items():
            if torch.is_tensor(val):
                setattr(self, prop, val.to(device))
            elif isinstance(val, collections.abc.Sequence) or isinstance(val, collections.abc.Iterable):
                val_ = [v.to(device) if torch.is_tensor(v) else v for v in val]
                setattr(self, prop, val_)
        return self

