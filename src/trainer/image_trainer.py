from __future__ import print_function, division, absolute_import
import time
import numpy as np
from sklearn import preprocessing as prep
from numpy import linalg as LA
import imageio as im

from tqdm import tqdm
from multiprocessing import Pool
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
import os
from configs import TrainConfig

from paths import Paths
import math
from base import hyperparams as base_hp
from base.trainer import BaseTrainer
# from base.model import BaseGan
from models.mnist.gan import ImgGAN
from dataloaders.mnist import MnistDataLoader as Dataloader
from exp_context import ExperimentContext
import torchvision.utils as trv_utils
import imageio as imag

exp_name = ExperimentContext.exp_name
H = ExperimentContext.Hyperparams


class ImgTrainer(BaseTrainer):
    def __init__(self, model, data_loader, hyperparams, train_config, tensorboard_msg='', auto_encoder_dl=None):
        # type: (ImgGAN, Dataloader, base_hp.Hyperparams, TrainConfig, str) -> None

        self.H = hyperparams
        self.iter_no = 1
        self.n_iter_gen = 0
        self.n_iter_disc = 0
        self.train_config = train_config
        self.tensorboard_msg = tensorboard_msg
        self.train_generator = True
        self.data_loader = data_loader

        self.recon_dir = '../experiments/' + exp_name + '/images/recon/'
        self.gen_dir = '../experiments/' + exp_name + '/images/gen/'

        model.create_params_dir()

        self.writer = {
            'train': SummaryWriter(model.get_log_writer_path('train')),
            'test': SummaryWriter(model.get_log_writer_path('test')),
        }

        self.seed_z = data_loader.get_z_dist(self.H.seed_batch_size, dist_type='normal')

        seed_data, seed_labels = data_loader.random_batch('test', self.H.seed_batch_size)
        test_seed = {
            'x': seed_data,
            'l': seed_labels,
            'z': model.sample((self.H.seed_batch_size,))
        }

        seed_data, seed_labels = data_loader.random_batch('train', self.H.seed_batch_size)
        train_seed = {
            'x': seed_data,
            'l': seed_labels,
            'z': model.sample((self.H.seed_batch_size,))
        }

        self.seed_data = {
            'train': train_seed,
            'test': test_seed,
        }

        # toDO
        self.pool = Pool(processes=4)

        super(ImgTrainer, self).__init__(model, data_loader, self.H.n_iterations)

    # Check Functions for various operations
    def is_console_log_step(self):
        n_step = self.train_config.n_step_console_log
        return n_step > 0 and self.iter_no % n_step == 0

    def is_tboard_log_step(self):
        n_step = self.train_config.n_step_tboard_log
        return n_step > 0 and self.iter_no % n_step == 0

    def is_params_save_step(self):
        n_step = self.train_config.n_step_save_params
        return n_step > 0 and self.iter_no % n_step == 0

    def is_validation_step(self):
        n_step = self.train_config.n_step_validation
        return n_step > 0 and self.iter_no % n_step == 0

    def is_visualization_step(self):
        if self.H.show_visual_while_training:
            if self.iter_no % self.train_config.n_step_visualize == 0:
                return True
            elif self.iter_no < self.train_config.n_step_visualize:
                if self.iter_no % 200 == 0:
                    return True
        return False

    def log_console(self, metrics):
        print('Test Step', self.iter_no + 1)
        print('%s: step %i:     Disc Acc: %.3f' % (exp_name, self.iter_no, metrics['accuracy_dis_x']))
        print('%s: step %i:     Gen  Acc: %.3f' % (exp_name, self.iter_no, metrics['accuracy_gen_x']))
        print('%s: step %i: x_recon Loss: %.3f' % (exp_name, self.iter_no, metrics['loss_x_recon']))
        print('%s: step %i: z_recon Loss: %.3f' % (exp_name, self.iter_no, metrics['loss_z_recon']))
        print()

    #  complete taining and validation and seed
    def validation(self):
        self.model.eval()
        H = self.H
        model = self.model
        dl = self.data_loader

        iter_time_start = time.time()

        x_test, _ = dl.next_batch('test')
        z_test = model.sample((x_test.shape[0],))
        metrics = model.compute_metrics(x_test, z_test, separate_acc=True)
        g_acc, d_acc = metrics['accuracy_gen_x'], metrics['accuracy_dis_x']

        # Tensorboard Log
        if self.is_tboard_log_step():
            for tag, value in metrics.items():
                self.writer['test'].add_scalar(tag, value, self.iter_no)

        # Console Log
        if self.is_console_log_step():
            self.log_console(metrics)
            iter_time_end = time.time()

            print('Test Iter Time: %.4f' % (iter_time_end - iter_time_start))
            print('------------------------------------------------------------')
        self.model.train()

    def gen_train_limit_reached(self, gen_accuracy, disc_accuracy):
        return self.n_iter_gen == self.H.gen_iter_count  # or gen_accuracy >= 80  # or disc_accuracy <= 80

    def disc_train_limit_reached(self, disc_accuracy):
        return self.n_iter_disc == self.H.disc_iter_count  # or disc_accuracy >= 95

    # Conditional Switch - Training Networks
    def switch_train_mode(self, gen_accuracy, disc_accuracy):
        if self.train_generator:
            if self.gen_train_limit_reached(gen_accuracy, disc_accuracy):
                self.n_iter_gen = 0
                self.train_generator = False

        if not self.train_generator:
            if self.disc_train_limit_reached(disc_accuracy):
                self.n_iter_disc = 0
                self.train_generator = True

    # def get_z(self, dist_type='normal', batch_size=H.batch_size):
    #     return self.data_loader.get_z_dist(n_samples=batch_size, dist_type=dist_type).float()

    def train_step_ae(self, x_train, z_train):
        if self.H.train_autoencoder:
            # model.step_train_encoder(x_train, z_train)
            # model.step_train_decoder(z_train)
            self.model.step_train_autoencoder(x_train, z_train)

    def train_step_ad(self, x_train, z_train):
        model = self.model
        H = self.H

        if self.train_generator:
            self.n_iter_gen += 1
            if H.train_generator_adv:
                model.step_train_generator(x_train, z_train, True)
        else:
            self.n_iter_disc += 1
            model.step_train_discriminator(x_train, z_train, True)

    def save_img(self, test_seed=None, split='train'):
        # test_seed = self.fixed_seed if test_seed is None else test_seed
        x = test_seed['x']
        z = test_seed['z']

        x_recon = self.model.reconstruct_x(x)
        x_gen = self.model.decode(z)

        recon_img = trv_utils.make_grid(x_recon, nrow=8, padding=2)
        gen_img = trv_utils.make_grid(x_gen, nrow=8, padding=2)
        real = trv_utils.make_grid(x, nrow=8, padding=2)
        return recon_img, gen_img, real

    def visualize(self, split):
        self.model.eval()
        tic_viz = time.time()  # def visualize(self, split):
        tic_data_prep = time.time()  # self.model.eval()
        recon, gen, real = self.save_img(self.seed_data[split], split=split)
        # print(split, 'recon', recon.shape, recon.min(), recon.max())  # recon, gen, real = self.save_img(self.seed_data[split])
        # print(split, 'gen', gen.shape, gen.min(), gen.max())
        # print(split, 'real', real.shape, real.min(), real.max())  # writer = self.writer[split]
        tac_data_prep = time.time()  # image_tag = '%s-plot' % self.model.name
        time_data_prep = tac_data_prep - tic_data_prep  # iter_no = self.iter_no

        writer = self.writer[split]  # def callback(item):
        image_tag = '%s-plot' % self.model.name  # real, recon, gen, image_tag, iter_no = item
        iter_no = self.iter_no  # writer.add_image(image_tag + '-recon', recon, iter_no)
        #     writer.add_image(image_tag + '-gen', gen, iter_no)
        writer.add_image(image_tag + '-recon', recon, iter_no)  # writer.add_image(image_tag + '-real', real, iter_no)
        writer.add_image(image_tag + '-gen', gen, iter_no)
        # writer.add_image(image_tag + '-recon', recon, iter_no)
        writer.add_image(image_tag + '-real', real, iter_no)
        # self.pool.apply_async(log_images, (real, recon, gen, image_tag, iter_no), callback=callback)

        self.model.train()  # self.model.train()

    def full_train_step(self, validation=True, save_params=True, visualize=True, separate_acc=True):
        dl = self.data_loader
        model = self.model  # type: ImgGAN
        H = self.H

        iter_time_start = time.time()

        x_train, _ = dl.next_batch('train')
        #print("line 224 image_trainer",x_train)
        z_train = model.sample((H.batch_size,))
        # if self.iter_no % 3 == 0:
        self.train_step_ae(x_train, z_train)
        self.train_step_ad(x_train, z_train)

        # Train Losses Computation
        self.model.eval()
        x_train_batch = self.seed_data['train']['x']
        z_train_batch = self.seed_data['train']['z']

        metrics = model.compute_metrics(x_train_batch, z_train_batch, separate_acc)
        self.model.train()

        g_acc, d_acc = metrics['accuracy_gen_x'], metrics['accuracy_dis_x']
        # Switch Training Networks - Gen | Disc
        self.switch_train_mode(g_acc, d_acc)

        # Tensorboard Log
        if self.is_tboard_log_step():
            for tag, value in metrics.items():
                self.writer['train'].add_scalar(tag, value.item(), self.iter_no)
            self.writer['train'].add_scalar('switch_train_mode', int(self.train_generator), self.iter_no)

        # Validation Computations
        if validation and self.is_validation_step():
            self.validation()
        #
        # Weights Saving
        if save_params and self.is_params_save_step():
            tic_save = time.time()
            model.save_params(dir_name='iter', weight_label='iter', iter_no=self.iter_no)
            tac_save = time.time()
            save_time = tac_save - tic_save
            if self.is_console_log_step():
                print('Param Save Time: %.4f' % (save_time))
                print('------------------------------------------------------------')

        # # Visualization
        if visualize and self.is_visualization_step():
            # previous_backend = plt.get_backend()
            # plt.switch_backend('Agg')
            self.visualize('train')
            self.visualize('test')
            # plt.switch_backend(previous_backend)

    def train(self, n_iterations=None, enable_tqdm=True, *args, **kwargs):
        n_iterations = n_iterations or self.n_iterations
        start_iter = self.iter_no
        end_iter = start_iter + n_iterations + 1

        if enable_tqdm:
            with tqdm(total=n_iterations) as pbar:
                for self.iter_no in range(start_iter, end_iter):
                    self.full_train_step(*args, **kwargs)
                    pbar.update(1)


        else:
            for self.iter_no in range(start_iter, end_iter):
                self.full_train_step(*args, **kwargs)
