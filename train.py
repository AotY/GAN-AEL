# -*- coding: utf-8 -*-
# Copyright © 2019 LeonTao
#
# Distributed under terms of the MIT license.

import os
import sys
import random
import math
import time
import argparse

from tqdm import tqdm
import torch
from torch.optim import optim
import torch.nn.funcional as F

from gan_ael import GANAEL
from misc.dataset import build_iterator, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
from modules.early_stopping import EarlyStopping

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='')
parser.add_argument('--embedding_size', type=int)
parser.add_argument('--hidden_size', type=int)
parser.add_argument('--bidirectional', action='store_true')
parser.add_argument('--num_layers', type=int)
parser.add_argument('--dropout', type=float)
parser.add_argument('--lr_g', type=float, default=0.001)
parser.add_argument('--lr_d', type=float, default=0.001)
parser.add_argument('--max_grad_norm', type=float, default=0.0)
parser.add_argument('--min_len', type=int, default=5)
parser.add_argument('--q_max_len', type=int, default=60)
parser.add_argument('--r_max_len', type=int, default=60)
parser.add_argument('--batch_size', type=int, help='')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--es_patience', type=int, help='early stopping patience.')
parser.add_argument('--lr_patience', type=int,
                    help='Number of epochs with no improvement after which learning rate will be reduced')
parser.add_argument('--device', type=str, help='cpu or cuda')
parser.add_argument('--save_model', type=str, help='save path')
parser.add_argument('--save_mode', type=str,
                    choices=['all', 'best'], default='best')
parser.add_argument('--checkpoint', type=str, help='checkpoint path')
parser.add_argument('--log', type=str, help='save log.')
parser.add_argument('--seed', type=str, help='random seed', default=23)
args = parser.parse_args()

print(' '.join(sys.argv))

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
device = torch.device(args.device)
args.device = device

# model
model = GANAEL(args).to(device)

print(model)

optimizer_G = optim.Adam(
    filter(lambda x: x.requires_grad, model.generator.parameters()),
    args.lr_g,
    betas=(0.9, 0.98),
    eps=1e-09
)

optimizer_D = optim.Adam(
    filter(lambda x: x.requires_grad, model.discriminator.parameters()),
    args.lr_d,
    betas=(0.9, 0.98),
    eps=1e-09
)

train_iterator, valid_iterator, q_field, r_field = build_iterator(args)

PAD_ID = q_field.vocab.stoi(PAD_TOKEN)
SOS_ID = q_field.vocab.stoi(SOS_TOKEN)
EOS_ID = q_field.vocab.stoi(EOS_TOKEN)

class GeneratorTraining:
    def __init__(self,):
        # early stopping
        self.early_stopping = EarlyStopping(
            type='min',
            min_delta=0.0001,
            patience=args.es_patience
        )

        pass

    def train_epochs(self):
        ''' Start training '''
        log_train_file = None
        log_valid_file = None

        if args.log:
            log_train_file = os.path.join(args.log, 'G.train.log')
            log_valid_file = os.path.join(args.log, 'G.valid.log')

            print('[Info] Training performance will be written to file: {} and {}'.format(
                log_train_file, log_valid_file))

            with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
                log_tf.write('epoch,loss,ppl,accuracy\n')
                log_vf.write('epoch,loss,ppl,accuracy\n')

        valid_accus = []
        for epoch in range(args.start_epoch, args.epochs + 1):
            print('[ G Epoch', epoch, ']')
            start = time.time()
            train_loss, train_accu = self.train(epoch)
            print(' (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '
                  'elapse: {elapse:3.3f} min'.format(
                      ppl=math.exp(min(train_loss, 100)),
                      accu=100*train_accu,
                      elapse=(time.time()-start)/60))

            start = time.time()
            valid_loss, valid_accu = eval(epoch)
            print(' (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '
                  'elapse: {elapse:3.3f} min'.format(
                      ppl=math.exp(min(valid_loss, 100)),
                      accu=100*valid_accu,
                      elapse=(time.time()-start)/60))

            valid_accus += [valid_accu]
            checkpoint = {
                'model': model.generator.state_dict(),
                'args': args,
                'epoch': epoch,
                'optimizer_G': optimizer_G.state_dict(),
                'valid_loss': valid_loss,
                'valid_accu': valid_accu
            }

            if args.save_model:
                if args.save_mode == 'all':
                    model_name = os.path.join(
                        args.save_model, 'G.accu_{accu:3.3f}.pth'.format(accu=100*valid_accu))
                    torch.save(checkpoint, model_name)
                elif args.save_mode == 'best':
                    model_name = os.path.join(args.save_model, 'G.best.pth')
                    if valid_accu >= max(valid_accus):
                        torch.save(checkpoint, model_name)
                        print('   - [Info] The checkpoint file has been updated.')

            if log_train_file and log_valid_file:
                with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                    log_tf.write('{epoch}, {loss: 8.5f}, {ppl: 8.5f}, {accu:3.3f}\n'.format(
                        epoch=epoch,
                        loss=train_loss,
                        ppl=math.exp(min(train_loss, 100)),
                        accu=100*train_accu))
                    log_vf.write('{epoch}, {loss: 8.5f}, {ppl: 8.5f}, {accu:3.3f}\n'.format(
                        epoch=epoch,
                        loss=valid_loss,
                        ppl=math.exp(min(valid_loss, 100)),
                        accu=100*valid_accu))

            # is early_stopping
            is_stop = self.early_stopping.step(valid_loss)
            if is_stop:
                print('Early Stopping.\n')
                return

    def train(self, epoch):
        ''' Epoch operation in training phase'''
        model.generator.train()
        #  model.discriminator.eval()
        total_loss = 0
        n_word_total = 0
        n_word_correct = 0
        for batch in tqdm(
                train_iterator, mininterval=2,
                desc=' (Training: %d) ' % epoch, leave=False):

            q_inputs, q_inputs_len, r_inputs, r_inputs_len = batch

            r_targets = r_inputs[1:, :]
            r_inputs = r_inputs[:-1, :]

            loss = 0

            optimizer_G.zero_grad()

            outputs = model.generator_forward( q_inputs, q_inputs_len, r_inputs)

            # backward
            loss, n_correct = self.cal_performance(outputs, r_targets)

            loss.backward()

            # update parameters
            optimizer_G.step()

            # note keeping
            total_loss += loss.item()

            non_pad_mask = r_targets.ne(PAD_ID)

            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

        loss_per_word = total_loss/n_word_total
        accuracy = n_word_correct/n_word_total
        return loss_per_word, accuracy

    def eval(self, epoch):
        ''' Epoch operation in evaluation phase '''
        model.generator.eval()

        total_loss = 0
        n_word_total = 0
        n_word_correct = 0

        with torch.no_grad():
            for batch in tqdm(
                    valid_iterator, mininterval=2,
                    desc=' (Validation: %d) ' % epoch, leave=False):

                q_inputs, q_inputs_len, r_inputs, r_inputs_len = batch

                r_targets = r_inputs[1:, :]
                r_inputs = r_inputs[:-1, :]

                outputs = model.generator_forward(q_inputs, q_inputs_len, r_inputs)

                # backward
                loss, n_correct = self.cal_performance(outputs, r_targets)

                # note keeping
                total_loss += loss.item()

                non_pad_mask = r_targets.ne(PAD_ID)
                n_word = non_pad_mask.sum().item()
                n_word_total += n_word
                n_word_correct += n_correct

        loss_per_word = total_loss/n_word_total
        accuracy = n_word_correct/n_word_total
        return loss_per_word, accuracy

    def cal_performance(self, pred, gold, smoothing=False):
        ''' Apply label smoothing if needed '''
        # pred: [max_len * batch_size, vocab_size]
        # gold: [max_len, batch_size]
        loss = self.cal_loss(pred, gold, smoothing)

        pred = pred.max(1)[1]
        gold = gold.contiguous().view(-1)
        non_pad_mask = gold.ne(PAD_ID)
        n_correct = pred.eq(gold)
        n_correct = n_correct.masked_select(non_pad_mask).sum().item()

        return loss, n_correct

    def cal_loss(self, pred, gold, smoothing):
        ''' Calculate cross entropy loss, apply label smoothing if needed. '''

        # [max_len * batch_size]
        gold = gold.contiguous().view(-1)

        if smoothing:
            eps = 0.1
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            non_pad_mask = gold.ne(PAD_ID)
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).sum()  # average later
        else:
            loss = F.cross_entropy(pred, gold, ignore_index=PAD_ID, reduction='sum')

        return loss


class DiscriminatorTraining:
    def __init__(self):
        # early stopping
        self.early_stopping = EarlyStopping(
            type='min',
            min_delta=0.0001,
            patience=args.es_patience
        )

    def train_epochs(self):
        ''' Start training '''
        log_train_file = None
        log_valid_file = None

        if args.log:
            log_train_file = os.path.join(args.log, 'D.train.log')
            log_valid_file = os.path.join(args.log, 'D.valid.log')
            print('[Info] Training performance will be written to file: {} and {}'.format(
                log_train_file, log_valid_file))
            with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
                log_tf.write('epoch,loss\n')
                log_vf.write('epoch,loss\n')

        valid_losses = []
        for epoch in range(args.start_epoch, args.epochs + 1):
            print('[ D Epoch', epoch, ']')
            start = time.time()
            train_loss = self.train(epoch)
            print(' (Training) loss: {loss:3.3f}, elapse: {elapse:3.3f} min'.format(
                loss=train_loss,
                elapse=(time.time()-start)/60))

            start = time.time()
            valid_loss = eval(epoch)
            print(' (Validation) loss: {loss:3.3f} %, '
                  'elapse: {elapse:3.3f} min'.format(
                      loss=valid_loss,
                      elapse=(time.time()-start)/60))

            valid_losses += [valid_loss]

            checkpoint = {
                'model': model.discriminator.state_dict(),
                'args': args,
                'epoch': epoch,
                'optimizer_D': optimizer_D.state_dict(),
                'valid_loss': valid_loss,
            }

            if args.save_model:
                if args.save_mode == 'all':
                    model_name = os.path.join(args.save_model,
                        'D.loss_{loss:3.3f}.pth'.format(accu=valid_loss))
                    torch.save(checkpoint, model_name)
                elif args.save_mode == 'best':
                    model_name = os.path.join(args.save_model, 'D.best.pth')
                    if valid_loss <= min(valid_losses):
                        torch.save(checkpoint, model_name)
                        print('   - [Info] The checkpoint file has been updated.')

            if log_train_file and log_valid_file:
                with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                    log_tf.write('{epoch}, {loss: 8.5f}\n'.format(epoch=epoch, loss=train_loss))
                    log_vf.write('{epoch}, {loss: 8.5f}\n'.format(epoch=epoch, loss=valid_loss))

            # is early_stopping
            is_stop = self.early_stopping.step(valid_loss)
            if is_stop:
                print('Early Stopping.\n')
                return

    def train(self, epoch):
        ''' Epoch operation in training phase'''
        model.discriminator.train()
        #  model.generator.eval()

        total_loss = 0
        times = 0
        for batch in tqdm(
                train_iterator, mininterval=2,
                desc=' (Training: %d) ' % epoch, leave=False):

            q_inputs, q_inputs_len, r_inputs, r_inputs_len = batch

            # generator fake embedded
            sos_input = torch.ones(1, self.config.batch_size, dtype=torch.long, device=args.device) * SOS_ID
            fake_embedded = model.generator.approximate(q_inputs, q_inputs_len, sos_input)

            # [batch_size], fake and real
            fake_outputs, real_outputs = model.discriminator_forward(q_inputs, r_inputs, fake_embedded)

            # backward
            loss = - torch.mean(torch.log(real_outputs) + torch.log(1. - fake_outputs))

            # forward
            optimizer_D.zero_grad()

            loss.backward()

            # update parameters
            optimizer_D.step()

            # note keeping
            total_loss += loss.item()
            times += 1

        loss_avg = total_loss / times
        return loss_avg

    def eval(self, epoch):
        ''' Epoch operation in evaluation phase '''
        model.discriminator.eval()

        total_loss = 0
        times = 0

        with torch.no_grad():
            for batch in tqdm(
                    valid_iterator, mininterval=2,
                    desc=' (Validation: %d) ' % epoch, leave=False):

                q_inputs, q_inputs_len, r_inputs, r_inputs_len = batch

                # generator fake embedded
                sos_input = torch.ones(1, self.config.batch_size, dtype=torch.long, device=args.device) * SOS_ID
                fake_embedded = model.generator.approximate(q_inputs, q_inputs_len, sos_input)

                # [batch_size], fake and real
                fake_outputs, real_outputs = model.discriminator_forward(q_inputs, r_inputs, fake_embedded)

                # backward
                loss = - torch.mean(torch.log(fake_outputs) + torch.log(1. - real_outputs))

                # note keeping
                total_loss += loss.item()
                times += 1

        loss_avg = total_loss / times
        return loss_avg

class AdversarialTraining:
    def __init__(self):
        pass

    def train_epochs(self):
        for epoch in range(args.start_epoch, args.epochs + 1):
            print('[ A Epoch', epoch, ']')
            start = time.time()
            train_loss_G, train_loss_D = self.train(epoch)
            print(' (Training) loss_G: {loss_G:3.3f}, loss_D: {loss_D: 3.3f},  elapse: {elapse:3.3f} min'.format(
                loss_G=train_loss_G,
                loss_D=train_loss_D,
                elapse=(time.time()-start)/60))

    def train(self, epoch):
        model.discriminator.train()
        model.generator.train()

        #  model.generator.encoder.requires_grad = False
        #  model.generator.decoder.ael.fc = False

        total_loss_G = 0
        total_loss_D = 0
        times = 0

        for batch in tqdm(
                train_iterator, mininterval=2,
                desc=' (Training: %d) ' % epoch, leave=False):

            q_inputs, q_inputs_len, r_inputs, r_inputs_len = batch

            sos_input = torch.ones(1, self.config.batch_size, dtype=torch.long, device=args.device) * SOS_ID
            fake_outputs, real_outputs = model(q_inputs, q_inputs_len, r_inputs, sos_input)

            # backward
            loss_D = -torch.mean(torch.log(real_outputs) + torch.log(1. - fake_outputs))
            loss_G = torch.mean(torch.log(1. - fake_outputs))

            # update parameters
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            total_loss_G += loss_G.item()
            total_loss_D += loss_D.item()

            times += 1

        loss_avg_G = total_loss_G / times
        loss_avg_D = total_loss_D / times
        return loss_avg_G, loss_avg_D

if __name__ == '__main__':
    # train G
    gt = GeneratorTraining()
    print('generator training ...')
    gt.train_epochs()
    print('End of generator training ...')

    # train D
    dt = DiscriminatorTraining()
    print('discriminator training ...')
    dt.train_epochs()
    print('End of discriminator training ...')

    # Adversarial
    at = AdversarialTraining()
    print('adversarial training ...')
    at.train_epochs()
