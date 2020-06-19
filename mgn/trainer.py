#!/usr/bin/env python
#-*- coding:utf-8 -*-

import json, os, sys, time
import torch
import utils.utils as utils
from utils.logger import Logger

import logging, time, platform
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")
import wandb

class Trainer():
    """Trainer"""

    def __init__(self, opt, train_loader, val_loader, model, executor):
        self.opt = opt
        self.reinforce = opt.reinforce
        self.reward_decay = opt.reward_decay
        self.entropy_factor = opt.entropy_factor
        self.num_iters = opt.num_iters
        self.run_dir = opt.run_dir
        self.display_every = opt.display_every
        self.checkpoint_every = opt.checkpoint_every
        self.visualize_training = opt.visualize_training
        self.visualize_training_wandb = opt.visualize_training_wandb
        if opt.dataset == 'clevr':
            self.vocab = utils.load_vocab(opt.clevr_vocab_path)
        elif opt.dataset == 'clevr-humans':
            self.vocab = utils.load_vocab(opt.human_vocab_path)
        else:
            raise ValueError('Invalid dataset')

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.executor = executor

        # Create Optimizer #
        # _params_bline = list(filter(lambda p: p.requires_grad, model.seq2seq_baseline.parameters()))
        _params = list(filter(lambda p: p.requires_grad, model.seq2seq.parameters()))
        _params_gnn = list(filter(lambda p: p.requires_grad, model.seq2seq.gnn.parameters()))
        _params_enc = list(filter(lambda p: p.requires_grad, model.seq2seq.encoder.parameters()))
        # self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.seq2seq.parameters()),
        #                                   lr=opt.learning_rate)
        self.optimizer = torch.optim.Adam(_params, lr=opt.learning_rate)
        self.stats = {
            'train_losses': [],
            'train_batch_accs': [],
            'train_accs_ts': [],
            'val_losses': [],
            'val_accs': [],
            'val_accs_ts': [],
            'best_val_acc': -1,
            'model_t': 0
        }
        if opt.visualize_training:
            # Tensorboard #
            # from reason.utils.logger import Logger
            self.logger = Logger('%s/logs' % opt.run_dir)

        if opt.visualize_training_wandb:
            # WandB: Log metrics with wandb #
            wandb_proj_name = opt.wandb_proj_name
            wandb_identifier = opt.run_identifier
            wandb_name = f"{wandb_identifier}"
            wandb.init(project=wandb_proj_name, name=wandb_name, notes="Running from mgn.reason.trainer.py")
            wandb.config.update(opt)
            wandb.watch(self.model.seq2seq)

    def train(self):
        training_mode = 'reinforce' if self.reinforce else 'seq2seq'
        print('| start training %s, running in directory %s' % (training_mode, self.run_dir))
        num_workers = self.opt.num_workers
        t = 0
        epoch = 0
        baseline = 0
        start = time.time()
        while t < self.num_iters:
            epoch += 1
            for x, y, ans, idx, g_data in self.train_loader:
                t += 1
                loss, reward = None, None
                self.model.set_input(x, y, g_data)
                self.optimizer.zero_grad()
                if self.reinforce:
                    pred = self.model.reinforce_forward()
                    reward = self.get_batch_reward(pred, ans, idx, 'train')
                    baseline = reward * (1 - self.reward_decay) + baseline * self.reward_decay
                    advantage = reward - baseline
                    self.model.set_reward(advantage)
                    self.model.reinforce_backward(self.entropy_factor)
                else:
                    loss = self.model.supervised_forward()
                    self.model.supervised_backward()
                self.optimizer.step()

                if t % self.display_every == 0:
                    if self.reinforce:
                        self.stats['train_batch_accs'].append(reward)
                        self.log_stats('training batch reward', reward, t)
                        print('| iteration %d / %d, epoch %d, reward %f' % (t, self.num_iters, epoch, reward))
                    else:
                        self.stats['train_losses'].append(loss)
                        self.log_stats('training batch loss', loss, t)
                        print('| iteration %d / %d, epoch %d, loss %f' % (t, self.num_iters, epoch, loss))
                    self.stats['train_accs_ts'].append(t)

                if t % self.checkpoint_every == 0 or t >= self.num_iters:
                    print('| checking validation accuracy')
                    val_acc = self.check_val_accuracy()
                    print('| validation accuracy %f' % val_acc)
                    if val_acc >= self.stats['best_val_acc']:
                        print('| best model')
                        self.stats['best_val_acc'] = val_acc
                        self.stats['model_t'] = t
                        checkpoint_fp = f"{self.run_dir}/checkpoint_best.pt"
                        self.model.save_checkpoint(checkpoint_fp)
                        self.model.save_checkpoint('%s/checkpoint_iter%08d.pt' % (self.run_dir, t))
                        if self.visualize_training_wandb:
                            wandb.save(checkpoint_fp)
                    if not self.reinforce:
                        val_loss = self.check_val_loss()
                        print('| validation loss %f' % val_loss)
                        self.stats['val_losses'].append(val_loss)
                        self.log_stats('val loss', val_loss, t)
                    self.stats['val_accs'].append(val_acc)
                    self.log_stats('val accuracy', val_acc, t)
                    self.stats['val_accs_ts'].append(t)
                    # Save Checkpoint #
                    self.model.save_checkpoint('%s/checkpoint.pt' % self.run_dir)

                    checkpoint_dict = {
                        'args': self.opt.__dict__,
                        'stats': self.stats
                    }
                    json_fp = '%s/stats.json' % self.run_dir
                    #json_fp = os.path.join(self.run_dir, f'stats_{self.opt.run_timestamp}.json')
                    logging.info(f'Saving train_opt_[ts].json at: {json_fp}')
                    with open(json_fp, 'w') as f:
                        json.dump(self.stats, f, indent=2)
                        #json.dump(checkpoint_dict, f, indent=1)
                    if self.visualize_training_wandb:
                        wandb.save(json_fp)
                    self.log_params(t)

                if t >= self.num_iters:
                    break
        end = time.time()
        logging.info(f"Finished {self.num_iters} or {epoch}epochs in {end-start}s, with {num_workers}workers")

    def check_val_loss(self):
        loss = 0
        t = 0
        for x, y, _, _, g_data in self.val_loader:
            self.model.set_input(x, y, g_data)
            loss += self.model.supervised_forward()
            t += 1
        return loss / t if t != 0 else 0

    def check_val_accuracy(self):
        reward = 0
        t = 0
        for x, y, ans, idx, g_data in self.val_loader:
            self.model.set_input(x, y, g_data)
            pred = self.model.parse()
            reward += self.get_batch_reward(pred, ans, idx, 'val')
            t += 1
        reward = reward / t if t != 0 else 0
        return reward

    def get_batch_reward(self, programs, answers, image_idxs, split):
        pg_np = programs.numpy()
        ans_np = answers.numpy()
        idx_np = image_idxs.numpy()
        reward = 0
        for i in range(pg_np.shape[0]):
            pred = self.executor.run(pg_np[i], idx_np[i], split)
            ans = self.vocab['answer_idx_to_token'][ans_np[i]]
            if pred == ans:
                reward += 1.0
        reward /= pg_np.shape[0]
        return reward

    def log_stats(self, tag, value, t):
        if self.visualize_training_wandb and wandb is not None:
            wandb.log({tag: value}, step=t)

        # Temsorboard #
        if self.visualize_training and self.logger is not None:
            self.logger.scalar_summary(tag, value, t)


    def log_params(self, t):
        if self.visualize_training_wandb and wandb is not None:
            train_qfn = self.opt.clevr_train_question_filename
            if train_qfn and not self.reinforce:
                wandb.log({"Info": wandb.Table(data=[train_qfn], columns=["train_question_filename"])})
            for k,v in self.stats.items():
                wandb.log({k: v}, step=t)

        # Tensorboard #
        if self.visualize_training and self.logger is not None:
            for tag, value in self.model.seq2seq.named_parameters():
                tag = tag.replace('.', '/')
                self.logger.histo_summary(tag, self._to_numpy(value), t)
                if value.grad is not None:
                    self.logger.histo_summary('%s/grad' % tag, self._to_numpy(value.grad), t)


    def _to_numpy(self, x):
        return x.data.cpu().numpy()