import time
import torch
import random
import numpy as np
from modules.optimizer import Optimizer
from model.seq_ner import SeqNERModel
from config.conf import args_config, data_config
from utils.dataset import DataLoader
from utils.datautil_seq import load_data, create_vocab, batch_variable, extract_ner_span
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

        self.model = SeqNERModel(num_wds=len(self.vocabs['word']),
                              num_chars=len(self.vocabs['char']),
                              num_tags=len(self.vocabs['tag']),
                              wd_embed_dim=args.wd_embed_dim,
                              char_embed_dim=args.char_embed_dim,
                              tag_embed_dim=args.tag_embed_dim,
                              bert_embed_dim=args.bert_embed_dim,
                              hidden_size=args.hidden_size,
                              num_rnn_layer=args.rnn_depth,
                              num_lbl=len(self.vocabs['ner']),
                              dropout=args.dropout,
                              bert_path=data_path['pretrain']['bert_model'],
                              num_bert_layer=args.bert_layers,
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

    '''
    def build_vocabs(self, train_data_path, embed_file=None, bert_vocab_path=None):
        vocabs = create_vocab(train_data_path, embed_file, bert_vocab_path)
        # save_to(self.args.vocab_chkp, vocabs)
        return vocabs
    '''

    def build_vocabs(self, datasets, embed_file=None, bert_vocab_path=None):
        vocabs = create_vocab(datasets, embed_file, bert_vocab_path)
        # save_to(self.args.vocab_chkp, vocabs)
        return vocabs

    def calc_train_acc(self, pred_score, gold_tags, mask=None):
        '''
        :param pred_score: (b, t, nb_tag)
        :param gold_tags: (b, t)
        :param mask: (b, t) 1对于有效部分，0对应pad
        :return:
        '''
        pred_tags = pred_score.data.argmax(dim=-1)
        nb_right = ((pred_tags == gold_tags) * mask).sum().item()
        nb_total = mask.sum().item()
        return nb_right, nb_total

    # BIO eval
    def eval_bio_acc(self, pred_tag_ids, gold_tag_ids, mask, ner_vocab, return_prf=False):
        '''
        :param pred_tag_ids: (b, t)
        :param gold_tag_ids: (b, t)
        :param mask: (b, t)  0 for padding
        :return:
        '''
        seq_lens = mask.sum(dim=1).tolist()
        nb_right, nb_pred, nb_gold = 0, 0, 0
        for i, l in enumerate(seq_lens):
            pred_tags = ner_vocab.idx2inst(pred_tag_ids[i][:l].tolist())
            gold_tags = ner_vocab.idx2inst(gold_tag_ids[i][:l].tolist())
            pred_ner_spans = set(extract_ner_span(pred_tags))
            gold_ner_spans = set(extract_ner_span(gold_tags))
            nb_pred += len(pred_ner_spans)
            nb_gold += len(gold_ner_spans)
            nb_right += len(pred_ner_spans & gold_ner_spans)

        if return_prf:
            return self.calc_prf(nb_right, nb_pred, nb_gold)
        else:
            return nb_right, nb_pred, nb_gold

    # BIOES eval
    def eval_bioes_acc(self, pred, target, mask, ner_vocab, return_prf=False):
        pred = pred.masked_select(mask).tolist()
        target = target.masked_select(mask).tolist()
        assert len(pred) == len(target)

        nb_right, nb_gold, nb_pred = 0, 0, 0
        # 统计pred tags中总的实体数
        entity_type = None
        valid = False
        for p in pred:
            _type = ner_vocab.idx2inst(p)
            if 'S-' in _type:
                nb_pred += 1
                valid = False
            elif 'B-' in _type:
                entity_type = _type.split('-')[1]
                valid = True
            elif 'I-' in _type:
                if entity_type != _type.split('-')[1]:
                    valid = False
            elif 'E-' in _type:
                if entity_type == _type.split('-')[1] and valid:
                    nb_pred += 1
                valid = False

        # 统计gold tags中总实体数以及预测正确的实体数
        begin = False
        for i, (t, p) in enumerate(zip(target, pred)):
            _type = ner_vocab.idx2inst(t)
            if 'S-' in _type:
                nb_gold += 1
                if t == p:
                    nb_right += 1
            elif 'B-' in _type:
                if t == p:
                    begin = True
            elif 'I-' in _type:
                if t != p:
                    begin = False
            elif 'E-' in _type:
                nb_gold += 1
                if t == p and begin:
                    nb_right += 1
                    begin = False

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
            train_right, train_total = 0, 0
            for i, batcher in enumerate(train_loader):
                batch = batch_variable(batcher, self.vocabs)
                batch.to_device(self.args.device)

                pred_score = self.model(batch.wd_ids, batch.ch_ids, batch.tag_ids, batch.bert_inps)

                mask = batch.wd_ids.gt(0)
                loss = self.model.tag_loss(pred_score, batch.ner_ids, mask)
                loss_val = loss.data.item()
                train_loss += loss_val

                nb_right, nb_total = self.calc_train_acc(pred_score, batch.ner_ids, mask)
                train_right += nb_right
                train_total += nb_total

                if self.args.update_step > 1:
                    loss = loss / self.args.update_step

                loss.backward()

                if (i + 1) % self.args.update_step == 0 or (i == self.args.max_step - 1):
                    nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()),
                                             max_norm=self.args.grad_clip)
                    optimizer.step()
                    self.model.zero_grad()

                logger.info('[Epoch %d] Iter%d time cost: %.2fs, lr: %.6f, train loss: %.3f, ACC: %.3f' % (
                    ep, i + 1, (time.time() - t1), optimizer.get_lr(), loss_val, train_right/train_total))

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
                mask = batch.wd_ids.gt(0)
                pred_tag_ids = self.model.tag_decode(pred_score, mask)
                # nb_right, nb_pred, nb_gold = self.eval_bio_acc(pred_tag_ids, batch.ner_ids, mask, self.vocabs['ner'])
                nb_right, nb_pred, nb_gold = self.eval_bioes_acc(pred_tag_ids, batch.ner_ids, mask, self.vocabs['ner'])
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


