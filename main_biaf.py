import time
import torch
import torch.nn.functional as F
import random
import numpy as np
from modules.optimizer import Optimizer
from model.biaffine_ner import NERParser
from config.conf import args_config, data_config
from utils.dataset import DataLoader
from utils.datautil import load_data, create_vocab, batch_variable
import torch.nn.utils as nn_utils
from logger.logger import logger


class Trainer(object):
    def __init__(self, args, data_config):
        self.args = args
        self.data_config = data_config

        genre = args.genre
        self.train_set, self.val_set, self.test_set = self.build_dataset(data_config, genre)
        # self.vocabs = self.build_vocabs(data_config[genre]['train'],
        #                                 data_config['pretrain']['word_embedding'],
        #                                 data_config['pretrain']['bert_vocab'])

        self.vocabs = self.build_vocabs(self.train_set+self.val_set+self.test_set,
                                        data_config['pretrain']['word_embedding'],
                                        data_config['pretrain']['bert_vocab'])

        self.model = NERParser(num_wds=len(self.vocabs['word']),
                              num_chars=len(self.vocabs['char']),
                              num_tags=len(self.vocabs['tag']),
                              wd_embed_dim=args.wd_embed_dim,
                              char_embed_dim=args.char_embed_dim,
                              tag_embed_dim=args.tag_embed_dim,
                              bert_embed_dim=args.bert_embed_dim,
                              hidden_size=args.hidden_size,
                              num_rnn_layer=args.rnn_depth,
                              ffnn_size=args.ffnn_size,
                              num_lbl=len(self.vocabs['ner']),
                              bert_path=data_path['pretrain']['bert_model'],
                              num_bert_layer=args.bert_layers,
                              ffnn_drop=args.ffnn_drop,
                              dropout=args.dropout,
                              embed_weight=self.vocabs['word'].embeddings).to(args.device)
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Training %d trainable parameters..." % total_params)

    def build_dataset(self, data_config, genre='conll_2003'):
        train_set = load_data(data_config[genre]['train'])
        val_set = load_data(data_config[genre]['dev'])
        test_set = load_data(data_config[genre]['test'])
        print('train data size:', len(train_set))
        print('validate data size:', len(val_set))
        print('test data size:', len(test_set))
        return train_set, val_set, test_set

    # def build_vocabs(self, train_data_path, embed_file=None, bert_vocab_path=None):
    #     vocabs = create_vocab(train_data_path, embed_file, bert_vocab_path)
    #     # save_to(self.args.vocab_chkp, vocabs)
    #     return vocabs

    def build_vocabs(self, datasets, embed_file=None, bert_vocab_path=None):
        vocabs = create_vocab(datasets, embed_file, bert_vocab_path)
        # save_to(self.args.vocab_chkp, vocabs)
        return vocabs

    def calc_loss(self, span_score, ner_ids):
        '''
        :param span_score: (b, t, t, c)
        :param ner_ids: (b, t, t)
        :return:
        '''
        num_ner = ner_ids.gt(0).sum()
        num_cls = span_score.size(-1)
        loss = F.cross_entropy(span_score.reshape(-1, num_cls), ner_ids.reshape(-1), ignore_index=0, reduction='sum')
        return loss / num_ner

    def ner_gold(self, ner_ids, sent_lens, ner_vocab=None):
        '''
        :param ner_ids: (b, t, t)
        :param sent_lens:  (b, )
        :param ner_vocab:
        :return:
        '''
        gold_res = []
        for ner_id, l in zip(ner_ids, sent_lens):
            res = []
            for s in range(l):
                for e in range(s, l):
                    type_id = ner_id[s, e].item()
                    if type_id not in [ner_vocab.pad_idx, ner_vocab.unk_idx]:
                        res.append((s, e, type_id))
            gold_res.append(res)

        return gold_res

    def ner_pred(self, pred_score, sent_lens, ner_vocab=None):
        '''
        :param pred_score: (b, t, t, c)
        :param sent_lens: (b, )
        # :param mask: (b, t)  1对应有效部分，0对应pad填充
        :return:
        '''
        # (b, t, t)
        type_idxs = pred_score.detach().argmax(dim=-1)
        # (b, t, t)
        span_max_score = pred_score.detach().gather(dim=-1, index=type_idxs.unsqueeze(-1)).squeeze(-1)
        final = []
        for span_score, tids, l in zip(span_max_score, type_idxs, sent_lens):
            cands = []
            for s in range(l):
                for e in range(s, l):
                    type_id = tids[s, e].item()
                    if type_id not in [ner_vocab.pad_idx, ner_vocab.unk_idx]:
                        cands.append((s, e, type_id, span_score[s, e].item()))

            pre_res = []
            for s, e, cls, _ in sorted(cands, key=lambda x: x[3], reverse=True):
                for s_, e_, _ in pre_res:
                    if s_ < s <= e_ < e or s < s_ <= e < e_:  # flat ner
                        break
                    if s <= s_ <= e_ <= e or s_ <= s <= e <= e_:  # nested ner
                        break
                else:
                    pre_res.append((s, e, cls))
            final.append(pre_res)

        return final

    def calc_acc(self, preds, golds, return_prf=False):
        '''
        :param preds: [(s, e, cls_id) ...]
        :param golds: [(s, e, cls_id) ...]
        :param return_prf: if True, return prf value, otherwise return number value
        :return:
        '''
        assert len(preds) == len(golds)
        nb_pred, nb_gold, nb_right = 0, 0, 0
        for pred_spans, gold_spans in zip(preds, golds):
            pred_span_set = set(pred_spans)
            gold_span_set = set(gold_spans)
            nb_pred += len(pred_span_set)
            nb_gold += len(gold_span_set)
            nb_right += len(pred_span_set & gold_span_set)

        if return_prf:
            return self.calc_prf(nb_right, nb_pred, nb_gold)
        else:
            return nb_right, nb_pred, nb_gold

    def calc_prf(self, nb_right, nb_pred, nb_gold):
        p = nb_right / (nb_pred + 1e-30)
        r = nb_right / (nb_gold + 1e-30)
        f = (2 * nb_right) / (nb_gold + nb_pred + 1e-30)
        return p, r, f

    def train_eval(self):
        train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size, shuffle=True)
        self.args.max_step = self.args.epoch * (len(train_loader) // self.args.update_step)
        print('max step:', self.args.max_step)
        optimizer = Optimizer(filter(lambda p: p.requires_grad, self.model.parameters()), args)
        best_dev_metric, best_test_metric = dict(), dict()
        patient = 0
        for ep in range(1, 1+self.args.epoch):
            train_loss = 0.
            self.model.train()
            t1 = time.time()
            train_right, train_pred, train_gold = 0, 0, 0
            for i, batcher in enumerate(train_loader):
                batch = batch_variable(batcher, self.vocabs)
                batch.to_device(self.args.device)

                pred_score = self.model(batch.wd_ids, batch.ch_ids, batch.tag_ids, batch.bert_inps)
                loss = self.calc_loss(pred_score, batch.ner_ids)
                loss_val = loss.data.item()
                train_loss += loss_val

                sent_lens = batch.wd_ids.gt(0).sum(dim=1)
                gold_res = self.ner_gold(batch.ner_ids, sent_lens, self.vocabs['ner'])
                pred_res = self.ner_pred(pred_score, sent_lens, self.vocabs['ner'])
                nb_right, nb_pred, nb_gold = self.calc_acc(pred_res, gold_res, return_prf=False)
                train_right += nb_right
                train_pred += nb_pred
                train_gold += nb_gold
                train_p, train_r, train_f = self.calc_prf(train_right, train_pred, train_gold)

                if self.args.update_step > 1:
                    loss = loss / self.args.update_step

                loss.backward()

                if (i + 1) % self.args.update_step == 0 or (i == self.args.max_step - 1):
                    nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()),
                                             max_norm=self.args.grad_clip)
                    optimizer.step()
                    self.model.zero_grad()

                logger.info('[Epoch %d] Iter%d time cost: %.2fs, lr: %.6f, train loss: %.3f, P: %.3f, R: %.3f, F: %.3f' % (
                    ep, i + 1, (time.time() - t1), optimizer.get_lr(), loss_val, train_p, train_r, train_f))

            dev_metric = self.evaluate('dev')
            if dev_metric['f'] > best_dev_metric.get('f', 0):
                best_dev_metric = dev_metric
                test_metric = self.evaluate('test')
                if test_metric['f'] > best_test_metric.get('f', 0):
                    # check_point = {'model': self.model.state_dict(), 'settings': args}
                    # torch.save(check_point, self.args.model_chkp)
                    best_test_metric = test_metric
                patient = 0
            else:
                patient += 1

            logger.info('[Epoch %d] train loss: %.4f, lr: %f, patient: %d, dev_metric: %s, test_metric: %s' % (
                ep, train_loss, optimizer.get_lr(), patient, best_dev_metric, best_test_metric))

            # if patient >= (self.args.patient // 2 + 1):  # 训练一定epoch, dev性能不上升, decay lr
            #     optimizer.lr_decay(0.95)

            if patient >= self.args.patient:  # early stopping
                break

        logger.info('Final Metric: %s' % best_test_metric)

    def evaluate(self, mode='test'):
        if mode == 'dev':
            test_loader = DataLoader(self.val_set, batch_size=self.args.test_batch_size)
        elif mode == 'test':
            test_loader = DataLoader(self.test_set, batch_size=self.args.test_batch_size)
        else:
            raise ValueError('Invalid Mode!!!')

        self.model.eval()
        nb_right_all, nb_pred_all, nb_gold_all = 0, 0, 0
        with torch.no_grad():
            for i, batcher in enumerate(test_loader):
                batch = batch_variable(batcher, self.vocabs)
                batch.to_device(self.args.device)

                pred_score = self.model(batch.wd_ids, batch.ch_ids, batch.tag_ids, batch.bert_inps)
                sent_lens = batch.wd_ids.gt(0).sum(dim=1)
                gold_res = self.ner_gold(batch.ner_ids, sent_lens, self.vocabs['ner'])
                pred_res = self.ner_pred(pred_score, sent_lens, self.vocabs['ner'])
                nb_right, nb_pred, nb_gold = self.calc_acc(pred_res, gold_res, return_prf=False)
                nb_right_all += nb_right
                nb_pred_all += nb_pred
                nb_gold_all += nb_gold
        p, r, f = self.calc_prf(nb_right_all, nb_pred_all, nb_gold_all)
        return dict(p=p, r=r, f=f)


if __name__ == '__main__':
    random.seed(1347)
    np.random.seed(2343)
    torch.manual_seed(1453)
    torch.cuda.manual_seed(1347)
    torch.cuda.manual_seed_all(1453)

    print('cuda available:', torch.cuda.is_available())
    print('cuDNN available:', torch.backends.cudnn.enabled)
    print('gpu numbers:', torch.cuda.device_count())

    args = args_config()
    if torch.cuda.is_available() and args.cuda >= 0:
        args.device = torch.device('cuda', args.cuda)
        torch.cuda.empty_cache()
    else:
        args.device = torch.device('cpu')

    data_path = data_config('./config/data_path.json')
    trainer = Trainer(args, data_path)
    trainer.train_eval()


