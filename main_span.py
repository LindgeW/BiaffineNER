import time
import torch
import random
import numpy as np
from modules.optimizer import Optimizer, AdamW, WarmupLinearSchedule
# from model.span_tagger import BertSpanTagger
from model.ner_model import NERTagger
from config.conf import args_config, data_config
from utils.dataset import DataLoader
from utils.datautil_span import load_data, create_vocab, batch_variable, extract_ner_spans
import torch.nn.utils as nn_utils
from logger.logger import logger

'''
BERT + Span for NER
'''


class Trainer(object):
    def __init__(self, args, data_config):
        self.args = args
        self.data_config = data_config

        genre = args.genre
        self.train_set, self.val_set, self.test_set = self.build_dataset(data_config, genre)

        self.vocabs = self.build_vocabs(self.train_set + self.val_set + self.test_set,
                                        data_config['pretrained']['bert_model'])

        self.model = NERTagger(
            bert_embed_dim=args.bert_embed_dim,
            hidden_size=args.hidden_size,
            num_rnn_layer=args.rnn_depth,
            # num_tag=len(self.vocabs['cws']),
            num_tag=len(self.vocabs['ner']),
            num_bert_layer=args.bert_layer,
            dropout=args.dropout,
            bert_model_path=data_config['pretrained']['bert_model']
        ).to(args.device)

        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Training %d trainable parameters..." % total_params)

    def build_dataset(self, data_config, genre):
        train_set = load_data(data_config[genre]['train'])
        val_set = load_data(data_config[genre]['dev'])
        test_set = load_data(data_config[genre]['test'])
        print('train data size:', len(train_set))
        print('validate data size:', len(val_set))
        print('test data size:', len(test_set))
        return train_set, val_set, test_set

    def build_vocabs(self, datasets, bert_model_path, embed_file=None):
        vocabs = create_vocab(datasets, bert_model_path, embed_file)
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

    # ner eval
    def eval_ner(self, pred_spans, gold_spans):
        nb_right, nb_pred, nb_gold = 0, 0, 0
        for pred_span, gold_span in zip(pred_spans, gold_spans):
            pred_span = set(pred_span)
            gold_span = set(gold_span)
            nb_pred += len(pred_span)
            nb_gold += len(gold_span)
            nb_right += len(pred_span & gold_span)
        return nb_right, nb_pred, nb_gold

    def calc_prf(self, nb_right, nb_pred, nb_gold):
        p = nb_right / (nb_pred + 1e-30)
        r = nb_right / (nb_gold + 1e-30)
        f = (2 * nb_right) / (nb_gold + nb_pred + 1e-30)
        return p, r, f

    def train_eval(self):
        train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size, shuffle=True)
        nb_batch = len(train_loader)
        self.args.max_step = self.args.epoch * (nb_batch // self.args.update_step)

        # ==> bert fine-tuning settings
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
        optimizer_bert_parameters = [
            {'params': [p for n, p in self.model.bert_named_params()
                        if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.bert_named_params()
                        if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0}
        ]
        bert_optimizer = AdamW(optimizer_bert_parameters, lr=self.args.bert_lr, eps=1e-8)
        bert_scheduler = WarmupLinearSchedule(bert_optimizer, warmup_steps=self.args.max_step//20, t_total=self.args.max_step)

        # bert_params = list(map(id, self.model.bert.parameters()))
        # base_params = filter(lambda p: id(p) not in bert_params and p.requires_grad, self.model.parameters())
        base_params = self.model.non_bert_params()
        optimizer = Optimizer(base_params, self.args)
        # ==> bert fine-tuning settings

        # bert_params = list(map(id, self.model.bert.parameters()))
        # base_params = filter(lambda p: id(p) not in bert_params and p.requires_grad, self.model.parameters())
        # model_params = [{'params': base_params},
        #                 {'params': filter(lambda p: p.requires_grad, self.model.bert.parameters()),
        #                  'lr': self.args.bert_lr}]
        # optimizer = Optimizer(model_params, self.args)

        best_dev_metric, best_test_metric = dict(), dict()
        patient = 0
        for ep in range(1, 1+self.args.epoch):
            self.model.train()
            t1 = time.time()
            train_loss = 0.
            train_right, train_total = 0, 0
            for i, batcher in enumerate(train_loader):
                batch = batch_variable(batcher, self.vocabs)
                batch.to_device(self.args.device)
                loss, start_score, end_score = self.model(batch.bert_inp, batch.start_ids, batch.end_ids, batch.mask)
                loss_val = loss.data.item()
                train_loss += loss_val

                if self.args.update_step > 1:
                    loss = loss / self.args.update_step

                loss.backward()

                start_right, start_total = self.calc_train_acc(start_score, batch.start_ids, batch.mask)
                end_right, end_total = self.calc_train_acc(end_score, batch.end_ids, batch.mask)
                train_right += (start_right + end_right)
                train_total += (start_total + end_total)
                train_acc = train_right / train_total

                if (i+1) % 10 == 0:
                    self.evaluate(self.val_set)

                if (i + 1) % self.args.update_step == 0 or (i + 1 == nb_batch):
                    nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.non_bert_params()),
                                             max_norm=self.args.grad_clip)
                    nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.bert.parameters()),
                                             max_norm=self.args.bert_grad_clip)
                    optimizer.step()
                    bert_optimizer.step()
                    bert_scheduler.step()
                    self.model.zero_grad()

                logger.info('[Epoch %d] Iter%d time cost: %.2fs, lr: %.6f, train loss: %.3f, Train ACC: %.3f' % (
                    ep, i, (time.time() - t1), optimizer.get_lr(), loss_val, train_acc))

            dev_metric = self.evaluate(self.val_set)
            if dev_metric['f'] > best_dev_metric.get('f', 0):
                best_dev_metric = dev_metric
                test_metric = self.evaluate(self.test_set)
                if test_metric['f'] > best_test_metric.get('f', 0):
                    # check_point = {'model': self.model.state_dict(), 'settings': args}
                    # torch.save(check_point, self.args.model_chkp)
                    best_test_metric = test_metric
                patient = 0
            else:
                patient += 1

            logger.info('[Epoch %d] train loss: %.4f, lr: %f, patient: %d, dev_metric: %s, test_metric: %s' % (
                ep, train_loss, optimizer.get_lr(), patient, best_dev_metric, best_test_metric))

            if patient >= self.args.patient:  # early stopping
                break

        logger.info('Final Test Metric: %s' % (best_test_metric))

    def evaluate(self, test_data):
        test_loader = DataLoader(test_data, batch_size=self.args.test_batch_size)
        self.model.eval()
        nb_right_all, nb_pred_all, nb_gold_all = 0, 0, 0
        with torch.no_grad():
            for i, batcher in enumerate(test_loader):
                batch = batch_variable(batcher, self.vocabs)
                batch.to_device(self.args.device)
                start_logit, end_logit = self.model(batch.bert_inp, mask=batch.mask)
                pred_spans = extract_ner_spans(start_logit, end_logit, batch.mask)
                nb_right, nb_pred, nb_gold = self.eval_ner(pred_spans, batch.batch_spans)
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


