import os
from utils.instance import Instance
import numpy as np
from utils.vocab import Vocab, MultiVocab, BERTVocab
import torch
import collections


def read_insts(file_reader):
    insts = []
    for line in file_reader:
        try:
            tokens = line.strip().split(' ')
            if line.strip() == '' or len(tokens) < 4:
                if len(insts) > 0:
                    yield insts
                insts = []
            elif len(tokens) == 4:
                token = tokens[0]
                if 'DOCSTART' not in token:
                    # 基数词统一成0
                    # token = '0' if tokens[1] == 'CD' else token.lower()
                    insts.append(Instance(token, tokens[1], tokens[3]))
        except Exception as e:
            print('exception occur: ', e)

    if len(insts) > 0:
        yield insts


def load_data(path):
    assert os.path.exists(path)
    dataset = []
    with open(path, 'r', encoding='utf-8') as fr:
        for insts in read_insts(fr):
            dataset.append(insts)
    return dataset


def get_embed_vocab(embed_file):
    assert os.path.exists(embed_file)
    embed_vocab = Vocab(bos=None, eos=None)
    vec_dim = 0
    with open(embed_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            tokens = line.strip().split(' ')
            if len(tokens) < 10:
                continue
            embed_vocab.add(tokens[0])
            if vec_dim == 0:
                vec_dim = len(tokens[1:])

    embed_weights = np.random.uniform(-0.5 / vec_dim, 0.5 / vec_dim, (len(embed_vocab), vec_dim))
    with open(embed_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            tokens = line.strip().split(' ')
            if len(tokens) < 10:
                continue
            idx = embed_vocab.inst2idx(tokens[0])
            embed_weights[idx] = np.asarray(tokens[1:], dtype=np.float32)
    embed_weights[embed_vocab.pad_idx] = 0.
    embed_weights /= np.std(embed_weights)
    embed_vocab.embeddings = embed_weights
    return embed_vocab


'''
def create_vocab(data_path, embed_file=None, bert_vocab_path=None, min_count=2):
    wd_vocab = Vocab(min_count, bos=None, eos=None)
    char_vocab = Vocab(bos=None, eos=None)
    tag_vocab = Vocab(bos=None, eos=None)
    ner_vocab = Vocab(bos=None, eos=None)
    with open(data_path, 'r', encoding='utf-8') as fr:
        for insts in read_insts(fr):
            for inst in insts:
                wd_vocab.add(inst.word)
                char_vocab.add(list(inst.word))
                tag_vocab.add(inst.pos_tag)

                if inst.ner_tag != 'O':
                    # including PER ORG LOC MISC and UNK
                    ner_tag = inst.ner_tag.split('-')[1]
                    ner_vocab.add(ner_tag)

    embed_vocab = get_embed_vocab(embed_file) if embed_file is not None else None
    bert_vocab = BERTVocab(bert_vocab_path) if bert_vocab_path is not None else None

    return MultiVocab(dict(
        word=wd_vocab,
        char=char_vocab,
        tag=tag_vocab,
        ner=ner_vocab,
        ext_wd=embed_vocab,
        bert=bert_vocab
    ))
'''


def create_vocab(datasets, embed_file=None, bert_vocab_path=None, min_count=2):
    wd_vocab = Vocab(min_count, bos=None, eos=None)
    char_vocab = Vocab(bos=None, eos=None)
    tag_vocab = Vocab(bos=None, eos=None)
    ner_vocab = Vocab(bos=None, eos=None)
    for insts in datasets:
        for inst in insts:
            wd_vocab.add(inst.word)
            char_vocab.add(list(inst.word))
            tag_vocab.add(inst.pos_tag)

            if inst.ner_tag != 'O':
                # including PER ORG LOC MISC and UNK
                ner_tag = inst.ner_tag.split('-')[1]
                ner_vocab.add(ner_tag)

    embed_count = wd_vocab.load_embeddings(embed_file)
    print("%d word pre-trained embeddings loaded..." % embed_count)

    bert_vocab = BERTVocab(bert_vocab_path) if bert_vocab_path is not None else None

    return MultiVocab(dict(
        word=wd_vocab,
        char=char_vocab,
        tag=tag_vocab,
        ner=ner_vocab,
        bert=bert_vocab
    ))


def create_gold_map(sent_insts, ner_vocab):
    s = 0
    ner_tabs = dict()
    n = len(sent_insts)
    for i, inst in enumerate(sent_insts):
        if i+1 < n and inst.ner_tag == 'O' and sent_insts[i+1].ner_tag != 'O':
            s = i + 1
        if inst.ner_tag != 'O':
            if i+1 == n:
                ner_tag = inst.ner_tag.split('-')[1]
                ner_tabs[(s, i)] = ner_vocab.inst2idx(ner_tag)
            elif sent_insts[i+1].ner_tag == 'O':
                ner_tag = inst.ner_tag.split('-')[1]
                ner_tabs[(s, i)] = ner_vocab.inst2idx(ner_tag)
    return ner_tabs


def batch_variable(batch_data, mVocab):
    batch_size = len(batch_data)
    max_seq_len = max(len(insts) for insts in batch_data)
    max_wd_len = max(len(inst.word) for insts in batch_data for inst in insts)

    wd_vocab = mVocab['word']
    ch_vocab = mVocab['char']
    tag_vocab = mVocab['tag']
    ner_vocab = mVocab['ner']
    bert_vocab = mVocab['bert']

    wd_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    ch_ids = torch.zeros((batch_size, max_seq_len, max_wd_len), dtype=torch.long)
    tag_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    # span ner target
    ner_ids = torch.zeros((batch_size, max_seq_len, max_seq_len), dtype=torch.long)

    wd_lst = []
    for i, insts in enumerate(batch_data):
        seq_len = len(insts)
        wd_ids[i, :seq_len] = torch.tensor([wd_vocab.inst2idx(inst.word) for inst in insts])
        tag_ids[i, :seq_len] = torch.tensor([tag_vocab.inst2idx(inst.pos_tag) for inst in insts])
        for j, inst in enumerate(insts):
            ch_ids[i, j, :len(inst.word)] = torch.tensor(ch_vocab.inst2idx(list(inst.word)))

        wd_lst.append([inst.word for inst in insts])

        '''
        s <= e, and non-entity span index is 1:
            1 2 1 5 1
            0 1 4 3 1
            0 0 1 2 7
            0 0 0 6 1
            0 0 0 0 1
        '''
        ner_map = create_gold_map(insts, ner_vocab)
        for s in range(seq_len):
            for e in range(s, seq_len):
                # Note: unk_idx denotes the non-entity span
                ner_ids[i, s, e] = ner_map.get((s, e), ner_vocab.unk_idx)

    bert_inps = map(torch.LongTensor, bert_vocab.batch_bertwd2id(wd_lst))

    return Batch(wd_ids=wd_ids,
                 ch_ids=ch_ids,
                 tag_ids=tag_ids,
                 ner_ids=ner_ids,
                 bert_inps=bert_inps)


class Batch:
    def __init__(self, **args):
        for prop, v in args.items():
            setattr(self, prop, v)

    def to_device(self, device):
        for prop, val in self.__dict__.items():
            if torch.is_tensor(val):
                setattr(self, prop, val.to(device))
            elif isinstance(val, collections.abc.Sequence) or \
                    isinstance(val, collections.abc.Iterable):
                val = (v.to(device) for v in val if torch.is_tensor(v))
                setattr(self, prop, val)
        return self

