import os
from utils.instance import Instance
from utils.vocab import Vocab, BERTVocab, MultiVocab
import torch
import re
import numpy as np
import collections


def preprocess(txt):
    return re.sub(r'\d+', '0', txt.strip())


def read_insts(file_reader):
    insts = []
    for line in file_reader:
        try:
            tokens = line.strip().split(' ')
            if line.strip() == '' or len(tokens) < 4:
                if len(insts) > 0:

                    bio_tags = [inst.ner_tag for inst in insts]
                    bioes_tags = bio2bioes(bio_tags)
                    for i, new_tag in enumerate(bioes_tags):
                        insts[i].ner_tag = new_tag

                    yield insts
                insts = []
            elif len(tokens) == 4:
                token = tokens[0]
                if 'DOCSTART' not in token:
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
# create vocab from dataset file
def create_vocab(data_path, embed_file=None, bert_vocab_path=None, min_count=2):
    wd_vocab = Vocab(min_count, bos=None, eos=None)
    char_vocab = Vocab(bos=None, eos=None)
    tag_vocab = Vocab(bos=None, eos=None)
    ner_vocab = Vocab(unk=None, bos=None, eos=None)
    with open(data_path, 'r', encoding='utf-8') as fr:
        for insts in read_insts(fr):
            for inst in insts:
                wd_vocab.add(inst.word)
                char_vocab.add(list(inst.word))
                tag_vocab.add(inst.pos_tag)
                ner_vocab.add(inst.ner_tag)

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


# create vocab from all datasets
def create_vocab(datasets, embed_file=None, bert_vocab_path=None, min_count=2):
    wd_vocab = Vocab(min_count, bos=None, eos=None)
    char_vocab = Vocab(bos=None, eos=None)
    tag_vocab = Vocab(bos=None, eos=None)
    ner_vocab = Vocab(unk=None, bos=None, eos=None)
    for insts in datasets:
        for inst in insts:
            wd_vocab.add(inst.word)
            char_vocab.add(list(inst.word))
            tag_vocab.add(inst.pos_tag)
            ner_vocab.add(inst.ner_tag)

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


'''
BIO -> BIOES (Begin Inside Outside End Single)
如：OBOBIIIO -> OSOBIIEO
'''
def bio2bioes(bio_tags):
    tag_len = len(bio_tags)
    for i, t in enumerate(bio_tags):
        if 'B-' in t and (i+1 == tag_len or 'I-' not in bio_tags[i+1]):
            _type = bio_tags[i].split('-')[1]
            bio_tags[i] = 'S-' + _type
        elif 'I-' in t and (i+1 == tag_len or 'I-' not in bio_tags[i+1]):
            _type = bio_tags[i].split('-')[1]
            bio_tags[i] = 'E-' + _type

    return bio_tags


# B-LOC O B-MISC I-LOC B-PER I-PER O O B-ORG I-ORG I-ORG I-LOC I-LOC O B-MISC O B-LOC I-LOC
# => [(0, 0, 'LOC'), (2, 2, 'MISC'), (4, 5, 'PER'), (8, 10, 'ORG'), (14, 14, 'MISC'), (16, 17, 'LOC')]
def extract_ner_span(tag_seq: list):
    span_res = []
    n = len(tag_seq)
    s = 0
    type_b = None
    for i, tag in enumerate(tag_seq):
        if tag == 'O':
            s = -1
        elif tag.split('-')[0] == 'B':
            s = i
            type_b = tag.split('-')[1]
            if i + 1 == n or tag_seq[i+1].split('-')[0] != 'I':
                span_res.append((s, i, type_b))
        elif tag.split('-')[0] == 'I':
            if s != -1:
                type_i = tag.split('-')[1]
                if type_i != type_b:
                    span_res.append((s, i - 1, type_b))
                    s = -1
                elif i + 1 == n or tag_seq[i+1].split('-')[0] != 'I':
                    span_res.append((s, i, type_i))
                    s = -1
    return span_res


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
    ner_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)

    wd_lst = []
    for i, insts in enumerate(batch_data):
        seq_len = len(insts)
        wd_ids[i, :seq_len] = torch.tensor([wd_vocab.inst2idx(inst.word) for inst in insts])
        tag_ids[i, :seq_len] = torch.tensor([tag_vocab.inst2idx(inst.pos_tag) for inst in insts])
        ner_ids[i, :seq_len] = torch.tensor([ner_vocab.inst2idx(inst.ner_tag) for inst in insts])
        wd_lst.append([inst.word for inst in insts])

        for j, inst in enumerate(insts):
            ch_ids[i, j, :len(inst.word)] = torch.tensor(ch_vocab.inst2idx(list(inst.word)))

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

