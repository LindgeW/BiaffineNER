import numpy as np


class DataSet(object):
    '''
    data_path-> instance对象集 -> 生成batch -> to_index (vocab) -> padding -> to_tensor
             -> 创建vocab

    bert_path -> bert_model / bert_tokenizer (vocab)

    embed_path -> pre_embeds / pre_vocab
    '''
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class MyDataSet(DataSet):
    def __init__(self, insts, transform=None):
        self.insts = insts
        self.transform = transform

    def __len__(self):
        return len(self.insts)

    def __getitem__(self, idx):
        sample = self.insts[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __iter__(self):
        for inst in self.insts:
            yield inst

    def index(self, item):
        return self.insts.index(item)

    def data_split(self, split_rate=0.33, shuffle=False):
        assert self.insts and len(self.insts) > 0
        if shuffle:
            np.random.shuffle(self.insts)
        val_size = int(len(self.insts) * split_rate)
        train_set = MyDataSet(self.insts[:-val_size])
        val_set = MyDataSet(self.insts[-val_size:])
        return train_set, val_set


def data_split(data_set, split_rate: list, shuffle=False):
    assert len(data_set) != 0, 'Empty dataset !'
    assert len(split_rate) != 0, 'Empty split rate list !'

    n = len(data_set)
    if shuffle:
        range_idxs = np.random.permutation(n)
    else:
        range_idxs = np.asarray(range(n))

    k = 0
    parts = []
    base = sum(split_rate)
    for i, part in enumerate(split_rate):
        part_size = int((part / base) * n)
        parts.append([data_set[j] for j in range_idxs[k: k+part_size]])
        k += part_size
    return tuple(parts)


class DataLoader(object):
    def __init__(self, dataset: list, batch_size=1, shuffle=False, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn

    def __iter__(self):
        n = len(self.dataset)
        if self.shuffle:
            idxs = np.random.permutation(n)
        else:
            idxs = range(n)

        batch = []
        for idx in idxs:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                if self.collate_fn:
                    yield self.collate_fn(batch)
                else:
                    yield batch
                batch = []

        if len(batch) > 0:
            if self.collate_fn:
                yield self.collate_fn(batch)
            else:
                yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


