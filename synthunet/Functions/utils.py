import copy
import os
import sys
import threading
import time
import numpy as np
from abc import ABC
import pdb
import synthunet.Functions.plotter as plt
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter as sw
import torch
from torch import nn

from niclib.utils import RemainingTimeEstimator, remove_extension, get_timestamp, save_to_csv


class Trainer:
    """Class for torch Module training using training and validation generators with basic metric support.
    It also has support for plugins, a way to embed any functionality in the training procedure.

    :param int max_epochs: maximum number of epochs to train. Training can be interrupted with a plugin by setting the
        attribute `keep_training` to False.
    :param torch.nn.Module loss_func: loss function.
    :param optimizer: torch.optim.Optimizer derived optimizer.
    :param dict train_metrics: dictionary with the desired metrics as key value pairs specifiying the metric name and
        the function object (callable) respectively. The loss function is automatically included under the key 'loss'.
    :param dict val_metrics: same as `train_metrics`.
    :param list plugins: List of plugin objects (inheriting from ``TrainerPlugin``).
    :param str device: torch torch_device i.e. 'cpu', 'cuda', 'cuda:0'...
    :param bool multigpu: (default: False) if True, it instances the model as a DataParallel model before training.

    Plugins are made by inheriting from the base class :py:class:`~niclib.net.train.TrainerPlugin` and overriding
    the inherited functions to implement the desired functionality at specific points during the training.
    """

    def __init__(self, filepath,filepath2, max_epochs, loss_func, feature_loss,matching_loss, optimizer_D,optimizer_D_2,optimizer_T,optimizer_T_2,
                 optimizer_T_opts=None,optimizer_T_2_opts=None,optimizer_D_opts=None,optimizer_D_2_opts=None, labels = 1,lam = 10,
                 train_metrics=None, val_metrics=None,b_size =32, plugins=None, device='cuda', multigpu=False,
                 save='best', metric_name='loss', mode='min', min_delta=1e-4):
        # assert all([isinstance(plugin, TrainerPlugin) for plugin in plugins])
        assert save in {'best', 'all', 'last'}
        assert mode in {'min', 'max'}

        self.filepath = filepath
        self.filepath2 = filepath2
        self.save = save
        self.labels = labels
        self.mode = mode
        self.metric_name = metric_name
        self.lam = lam


        self.min_delta = min_delta
        if train_metrics is not None:
            assert isinstance(train_metrics, dict) and all([callable(m) for m in train_metrics.values()])
        if  val_metrics is not None:
            assert isinstance(val_metrics, dict) and all([callable(m) for m in val_metrics.values()])
        # Basic training variables

        self.device = device
        self.use_multigpu = multigpu
        self.best_metric_1 = dict( epoch=-1, value=sys.float_info.max )
        self.best_metric = dict( epoch=-1, value=sys.float_info.max )
        self.best_metric_2 = dict( epoch=-1, value=sys.float_info.max )

        self.max_epochs = max_epochs
        self.filepath = filepath
        self.loss_func = loss_func
        self.matching_loss = matching_loss
        self.feature_loss = feature_loss
        self.optimizer_D = optimizer_D
        self.optimizer_D_2 = optimizer_D_2
        self.optimizer_T = optimizer_T
        self.optimizer_T_2 = optimizer_T_2
        self.optimizer_D_opts = optimizer_D_opts if optimizer_D_opts is not None else {}
        self.optimizer_T_opts = optimizer_T_opts if optimizer_T_opts is not None else {}
        self.optimizer_D_2_opts = optimizer_D_2_opts if optimizer_D_opts is not None else {}
        self.optimizer_T_2_opts = optimizer_T_2_opts if optimizer_T_opts is not None else {}
        self.bs = b_size
        # Metric functions
        self.train_metric_funcs = {'loss': copy.copy(loss_func)}
        if train_metrics is not None:
            self.train_metric_funcs.update(train_metrics)

        self.val_metric_funcs = {'loss': copy.copy(loss_func)}
        if val_metrics is not None:
            self.val_metric_funcs.update(val_metrics)

        # Plugin functionality
        self.plugins = plugins # TODO avoid empty function calls (function call overhead)
        [plugin.on_init(self) for plugin in self.plugins]

        # Runtime variables
        self.keep_training = True
        self.model_D =None
        self.model_D_1 =None
        self.model_D_2 =None
        self.model_T = None
        self.model_T_1 = None
        self.model_T_2 = None
        self.model_optimizer, self.model_optimizer_D, self.model_optimizer_T = None, None, None
        self.model_optimizer_D_2, self.model_optimizer_T_2 = None, None
        self.model_optimizer_T_1 = None

        self.train_gen = None
        self.val_gen = None
        self.middle = None
        self.lesion = None
        self.x, self.y, self.y_pred = None, None, None
        self.y_translated, self.translation_score, self.fake_score, self.real_score = None, None, None, None
        self.input, self.y_fake, self.target = None, None, None
        self.loss = None
        self.loss_T1, self.loss_T2 = None, None
        self.fake_loss, self.loss_D, self.real_loss = None, None, None
        self.fake_t1_loss, self.loss_D_1, self.real_t1_loss = None, None, None
        self.fake_flair_loss, self.loss_D_2, self.real_flair_loss = None, None, None
        self.loss_T = None
        self.fake_t1_score = None
        self.fake_flair_score = None
        self.real_t1_score = None
        self.real_flair_score = None
        self.adv_T_1, self.adv_T_2 = None, None
        self.loss_GAN = None
        self.adv_loss = None
        self.m_loss = None
        self.cycle_loss, self.cycle_loss_1, self.cycle_loss_2 = None, None, None

        self.train_metrics = dict()
        self.train_metrics_D = dict()
        self.train_metrics_D_1 = dict()
        self.train_metrics_D_2 = dict()
        self.train_metrics_T = dict()
        self.train_metrics_GAN = dict()
        self.val_metrics = dict()
        self.val_metrics_T = dict()
        self.image, self.in_img, self.tgt_img = None,None,None

        self.epoch_start = 0
        self.c_epoch = 0

    def train_D(self,input,target):
        self.model_D.zero_grad()
        self.target = target.to(self.device)
        self.input = input.to(self.device)

        # train discriminator on real
        x_real= self.target
        x_real = x_real.to( self.device )

        D_output = self.model_D( x_real )
        # soft label
        if self.labels ==1:
            y_real = torch.ones((D_output.size()))
        else:
            y_real = (1-0.7)*torch.rand((D_output.size()))+0.7

        y_real= y_real.to( self.device )
        D_real_loss = self.loss_func( D_output, y_real )
        D_output = torch.sigmoid(D_output)
        self.real_score = D_output

        # train discriminator on fake

        x = self.input.to( self.device )
        x_fake = self.model_T( x )
        D_output = self.model_D( x_fake.detach() )

        #soft label
        if self.labels == 1:
            y_fake =torch.zeros((D_output.size())).to( self.device )
        else:
            y_fake = 0.3*torch.rand((D_output.size())).to(self.device)
        D_fake_loss = self.loss_func( D_output, y_fake )
        D_output = torch.sigmoid(D_output)
        self.fake_score = D_output

        # gradient backprop & optimize ONLY D's parameters
        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        self.loss_D=(0.5)*D_loss
        self.model_optimizer_D.step()
        return self.loss_D

    def train_Discriminators(self, input, target):
        """
        :param input: T1 patches
        :param target: Flair patches
        :return: D_1_Loss and D_2_Loss
        D_1 = Discriminator trained to distinguish between real and fakes FLAIR patches
        D_2 = Discriminator trained to distinguish between real and fakes T1 patches
        """
        self.model_D_1.zero_grad()
        self.model_D_2.zero_grad()
        t1 = input.to(self.device) #2
        flair = target.to(self.device) #1
        # self.middle = middle.to(self.device)
        # self.lesion = lesion.to(self.device)

        D_output = self.model_D_1( flair )
        # soft label
        if self.labels ==1:
            ones = torch.ones((D_output.size()))
            # ones = torch.zeros((D_output.size()))
        else:
            ones = (1-0.7)*torch.rand((D_output.size()))+0.7
            # ones = 0.3*torch.rand((D_output.size()))

        ones= ones.to( self.device )
        D_real_flair_loss = self.loss_func( D_output, ones )
        D_output = torch.sigmoid(D_output)
        self.real_flair_score = D_output

        flair_fake = self.model_T_1( t1 )
        D_output = self.model_D_1( flair_fake.detach() )
        #soft label
        if self.labels == 1:
            zeros =torch.zeros((D_output.size())).to( self.device )
            # zeros =torch.ones((D_output.size())).to( self.device )
        else:
            # zeros = torch.zeros( (D_output.size()) ).to( self.device )
            zeros = 0.3*torch.rand((D_output.size())).to(self.device)
            # zeros =(1-0.7)*torch.rand((D_output.size()))+0.7
            # zeros.to(self.device)

        D_fake_flair_loss = self.loss_func( D_output, zeros )
        D_output = torch.sigmoid(D_output)
        self.fake_flair_score = D_output

        D_1_loss = (0.5)*(D_real_flair_loss + D_fake_flair_loss)

        #real t1 score must be calculated on 1 - 1 so i use middle
        D_output = self.model_D_2( t1 )
        # D_output = self.model_D_2( middle )
        if self.labels ==1:
            ones2 = torch.ones((D_output.size()))
            # ones = torch.zeros((D_output.size()))
        else:
            ones2 = (1-0.7)*torch.rand((D_output.size()))+0.7
            # ones = 0.3*torch.rand((D_output.size()))
        ones2 = ones2.to( self.device )
        D_real_t1_loss = self.loss_func( D_output, ones2 ) #check dimensionality
        D_output = torch.sigmoid(D_output)
        self.real_t1_score = D_output
        #I want t1 fake to be monodimensional so T_2 is 1-1
        t1_fake = self.model_T_2( flair )
        D_output = self.model_D_2( t1_fake.detach() )
        #soft label
        if self.labels == 1:
            zeros =torch.zeros((D_output.size())).to( self.device )
            # zeros =torch.ones((D_output.size())).to( self.device )
        else:
            # zeros = torch.zeros( (D_output.size()) ).to( self.device )
            zeros = 0.3*torch.rand((D_output.size())).to(self.device)
            # zeros =(1-0.7)*torch.rand((D_output.size()))+0.7
            # zeros.to(self.device)
        D_fake_t1_loss = self.loss_func( D_output, zeros )
        D_output = torch.sigmoid(D_output)
        self.fake_t1_score = D_output

        D_2_loss = 0.5 * (D_real_t1_loss + D_fake_t1_loss)
        D_1_loss.backward()
        D_2_loss.backward()
        self.loss_D_1=D_1_loss
        self.loss_D_2=D_2_loss
        self.model_optimizer_D.step()
        self.model_optimizer_D_2.step()

        return self.loss_D_1, self.loss_D_2



    def train_T(self, input, target):
        self.model_T.zero_grad()
        x = input.to( self.device )
        y = target.to(self.device)
        G_output = self.model_T( x )
        D_output = self.model_D( G_output )
        # D_output = torch.sigmoid(D_output)
        if self.labels ==1:
            y = torch.ones( (D_output.size())).to( self.device )
        else:
            y = (1 - 0.7) * torch.rand( (D_output.size()) ) + 0.7
            y = y.to(self.device)
        G_loss1 = self.loss_func( D_output, y )
        G_loss2 = self.feature_loss(G_output,target)

        G_loss = (0.5)*G_loss1 + (30)*G_loss2
        self.translation_score = D_output
        # gradient backprop & optimize ONLY G's parameters
        G_loss.backward()
        self.model_optimizer_T.step()
        self.loss_T = G_loss
        return self.loss_T


    def train_Translators(self, input, target, middle):

        self.model_T_1.zero_grad()
        self.model_T_2.zero_grad()
        t1 = input.to(self.device)
        flair = target.to(self.device)
        self.middle = middle.to( self.device )

        # real_t1_feature,_ = self.model_D_2(t1.detach(), matching = True)
        # real_flair_feature,_ = self.model_D_1(flair.detach(), matching = True)


        fake_flair = self.model_T_1(t1) #2 or 1
        fake_t1 = self.model_T_2(flair)# 1


        # fake_flair_feature, fake_flair_score = self.model_D_1(fake_flair, matching=True)
        # fake_t1_feature, fake_t1_score = self.model_D_2(fake_t1, matching=True)


        fake_flair_score = self.model_D_1(fake_flair)
        fake_t1_score = self.model_D_2(fake_t1)
        # real_flair_feature = torch.mean(real_flair_feature,0)
        # real_t1_feature = torch.mean(real_t1_feature,0)

        # fake_flair_feature = torch.mean(fake_flair_feature,0)
        # fake_t1_feature = torch.mean(fake_t1_feature,0)
        if self.labels ==1:
            ones = torch.ones( (fake_t1_score.size())).to( self.device )
            # ones = torch.zeros( (fake_t1_score.size())).to( self.device )
        else:
            ones = (1 - 0.7) * torch.rand( (fake_t1_score.size()) ) + 0.7
            # ones = 0.3 * torch.rand( (fake_t1_score.size()) )
            ones = ones.to(self.device)
        adv_flair_loss = self.loss_func(fake_flair_score, ones)
        adv_t1_loss = self.loss_func(fake_t1_score, ones)

        #feature loss goes with respect to single channel t1 so middle cause i produce 1 challen t1
        forward_c_loss = self.feature_loss(self.model_T_2(fake_flair),middle)
        backward_c_loss = self.feature_loss(self.model_T_1(fake_t1), flair)

        # adversarial_loss = adv_flair_loss+ adv_t1_loss
        # cycle_consistency_loss = forward_c_loss + backward_c_loss

        # m1_loss = self.matching_loss(fake_flair_feature,real_flair_feature.detach())
        # m2_loss = self.matching_loss(fake_t1_feature, real_t1_feature.detach())

        # self.m_loss = m1_loss + m2_loss
        # self.adv_loss = adversarial_loss
        # self.cycle_loss = cycle_consistency_loss

        #Create T1Loss as Adv_flair_loss+ lambda/2 * backward + m1_loss and save model_T1 according to it
        self.loss_T1 = adv_flair_loss + ((self.lam)/2)* backward_c_loss# + m1_loss
        self.loss_T2 = adv_t1_loss + ((self.lam)/2)* forward_c_loss# + m2_loss
        #Create T2Loss as Adv_t1_loss + lambda/2 * forward + m2_loss and save model_T2 according to it
        #Adapt self.loss_GAN as T1_Loss + T2_Loss
        # self.loss_GAN = (1)*adversarial_loss + (self.lam) * cycle_consistency_loss + self.m_loss
        self.loss_GAN = self.loss_T1 + self.loss_T2
        # adversarial_loss.backward()
        # cycle_consistency_loss.backward()
        total_loss = self.loss_GAN
        total_loss.backward()
        self.model_optimizer_T_1.step()
        self.model_optimizer_T_2.step()

        return self.loss_T1, self.loss_T2
    def train_Translators2(self, input, target):

        self.model_T_1.zero_grad()
        self.model_T_2.zero_grad()

        t1 = input.to(self.device)
        flair = target.to(self.device)
        # self.middle = middle.to( self.device )


        fake_flair = self.model_T_1(t1) #2->1
        cycle_t1 = self.model_T_2(fake_flair) #1->1

        fake_t1 = self.model_T_2(flair) #1->1
        # print('fake_t1_shape: ',fake_t1.shape)
        # print('lesion_shape: ',lesion.shape)
        # pdb.set_trace()
        # fake_t1_les = np.stack([fake_t1, lesion] ,axis =0) #1->2
        # fake_t1_les = torch.cat([fake_t1, lesion] ,dim =1) #1->2
        cycle_flair = self.model_T_1(fake_t1)#2->1

        # same_t1 = self.model_T_2(t1)
        same_t1 = self.model_T_2(t1)
        # flair_les = np.stack([flair, lesion], axis =0)
        # flair_les = torch.cat([flair, lesion], dim =1)
        same_flair = self.model_T_1(flair) #2->1

        fake_flair_score = self.model_D_1(fake_flair)
        fake_t1_score = self.model_D_2(fake_t1)

        if self.labels ==1:
            ones = torch.ones( (fake_t1_score.size())).to( self.device )
        else:
            ones = (1 - 0.7) * torch.rand( (fake_t1_score.size()) ) + 0.7
            ones = ones.to(self.device)

        self.adv_T_1 = self.loss_func(fake_flair_score, ones)
        self.adv_T_2 = 2 * self.loss_func(fake_t1_score, ones)

        self.cycle_loss = (self.lam * self.feature_loss(cycle_t1, t1)) + (self.lam * self.feature_loss(cycle_flair, flair))
        id_T_1 = 4* self.lam * 0.5* self.matching_loss(same_flair, flair)
        id_T_2 = 3 * self.lam * 0.5* self.matching_loss(same_t1, t1)
        loss_T1 = self.adv_T_1 + self.cycle_loss + id_T_1
        loss_T2 = self.adv_T_2 + self.cycle_loss + id_T_2
        self.loss_T1 = loss_T1
        self.loss_T2 = loss_T2
        self.loss_GAN = loss_T1 + loss_T2
        self.loss_GAN.backward()

        self.model_optimizer_T_1.step()
        self.model_optimizer_T_2.step()

        return self.loss_T1, self.loss_T2

    def train_Translators3(self, input, target):

        self.model_T_1.zero_grad()
        self.model_T_2.zero_grad()

        t1 = input.to( self.device )
        flair = target.to( self.device )

        fake_flair = self.model_T_1( t1 )  # 2->1
        cycle_t1 = self.model_T_2( fake_flair )  # 1->1 F(G(x))
        fake_t1 = self.model_T_2( flair )  # 1->1
        cycle_flair = self.model_T_1( fake_t1 ) # 2->1 G(F(y))

        same_t1 = self.model_T_2( t1 )
        same_flair = self.model_T_1( flair )  # 2->1

        fake_flair_score = self.model_D_1( fake_flair )
        cycle_flair_score = self.model_D_1(cycle_flair)
        fake_t1_score = self.model_D_2( fake_t1 )
        cycle_t1_score = self.model_D_2(cycle_t1)

        if self.labels == 1:
            ones = torch.ones( (fake_t1_score.size()) ).to( self.device )
        else:
            ones = (1 - 0.7) * torch.rand( (fake_t1_score.size()) ) + 0.7
            ones = ones.to( self.device )

        # T_loss_1 = 3 * self.lam * self.feature_loss(fake_flair, flair)
        # T_loss_2 = 3 * self.lam * self.feature_loss(fake_t1, t1)

        self.adv_T_1 = self.loss_func( fake_flair_score, ones ) #BCE(D_1(G(x)),1)
        self.adv_T_2 = self.loss_func( fake_t1_score, ones )  #BCE (D_2(F(y)),1)

        self.cycle_loss_1 = self.lam * self.feature_loss( cycle_t1, t1 )
        self.cycle_loss_2 = self.lam * self.feature_loss( cycle_flair, flair )
        self.cycle_loss = self.cycle_loss_1 +  self.cycle_loss_2
        # id_T_1 = 4 * self.lam * self.matching_loss( same_flair, flair )
        # id_T_2 = 3 * self.lam * self.matching_loss( same_t1, t1 )
        loss_T1 = 0.5 * self.adv_T_1  #+  T_loss_1 #+ 10 * id_T_1
        loss_T2 = 0.5 * self.adv_T_2 #+ self.cycle_loss #+ T_loss_2 #+ 10 * id_T_2
        # mul_T1 =  self.adv_T_1 *  (self.cycle_loss)**(1/4) * T_loss_1
        # mul_T2 =  self.adv_T_2 *  (self.cycle_loss)**(1/4) * T_loss_2
        self.loss_T1 = loss_T1
        self.loss_T2 = loss_T2
        self.loss_GAN = self.loss_T1 + self.loss_T2 + self.cycle_loss
        self.loss_GAN.backward()


        self.model_optimizer_T_1.step()
        self.model_optimizer_T_2.step()

        return self.loss_T1, self.loss_T2
    def train_gan(self, model_D, model_T, train_gen, val_gen, checkpoint=None):
        """
        Trains a given model using the provided :class:`torch.data.DataLoader` generator.

        :param torch.nn.Module model: the model to train.
        :param torch.utils.data.DataLoader train_gen: An iterator returning (x, y) pairs for training.
        :param torch.utils.data.DataLoader val_gen: An iterator returning (x, y) pairs for validation.
        :param dict checkpoint: (optional) checkpoint that can include 'epoch', 'model', 'optimizer' or 'loss'
        :return: None, after training you should load the stored model from disk using torch.load()
        """

        # Store given arguments in runtime variables to be available to plugins
        self.model_T, self.model_D, self.train_gen, self.val_gen = model_T, model_D, train_gen, val_gen
        self.model_optimizer_D = self.optimizer_D(self.model_D.parameters(), **self.optimizer_D_opts)
        self.model_optimizer_T = self.optimizer_T(self.model_T.parameters(), **self.optimizer_T_opts )

        if checkpoint is not None:
            if 'epoch' in checkpoint:
                self.epoch_start = checkpoint['epoch']
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint:
                self.model_optimizer.load_state_dict(checkpoint['optimizer'])
            if 'loss' in checkpoint:
                self.loss_func = checkpoint['loss']

        self.model_D = model_D.to(self.device)
        self.model_T = model_T.to(self.device)
        if self.use_multigpu:
            print("Using multigpu for training")
            self.model_D = nn.DataParallel(self.model_D)
            self.model_T = nn.DataParallel( self.model_T)

        print("Training model for {} epochs".format(self.max_epochs))
        [plugin.on_train_start(self) for plugin in self.plugins]

        for epoch_num in range(self.epoch_start, self.max_epochs):
            if self.keep_training is False:
                break
            self.c_epoch = epoch_num
            # Train current epoch
            [plugin.on_train_epoch_start(self, epoch_num) for plugin in self.plugins]
            self.train_gan_epoch()
            [plugin.on_train_gan_epoch_end(self, epoch_num) for plugin in self.plugins]

            # Validate on validation set
            [plugin.on_val_epoch_start(self, epoch_num) for plugin in self.plugins]
            self.validate_epoch_gan(filepath=self.filepath)
            [plugin.on_val_gan_epoch_end(self, epoch_num) for plugin in self.plugins]

        print("Training finished\n")
        [plugin.on_train_end(self) for plugin in self.plugins]


    def train_cyclegan(self, model_D_1,model_D_2, model_T_1,model_T_2, train_gen, val_gen, checkpoint=None):
        """
        Trains the CycleGan :class:`torch.data.DataLoader` generator.

        :param torch.nn.Module model_D_1: Discriminator which recognises real Flair.
        :param torch.nn.Module model_D_2: Discriminator which recognises real T1.
        :param torch.nn.Module model_T_1: Translator which translates from t1 to flair.
        :param torch.nn.Module model_T_2: Translator which translates from flair to t1.
        :param torch.utils.data.DataLoader train_gen: An iterator returning (x, y) pairs for training.
        :param torch.utils.data.DataLoader val_gen: An iterator returning (x, y) pairs for validation.
        :param dict checkpoint: (optional) checkpoint that can include 'epoch', 'model', 'optimizer' or 'loss'
        :return: None, after training you should load the stored model from disk using torch.load()
        """

        # Store given arguments in runtime variables to be available to plugins
        self.model_T_1, self.model_D_1, self.model_T_2, self.model_D_2 = model_T_1, model_D_1, model_T_2, model_D_2
        self.train_gen, self.val_gen = train_gen, val_gen
        self.model_optimizer_D = self.optimizer_D(self.model_D_1.parameters(), **self.optimizer_D_opts)
        self.model_optimizer_D_2 = self.optimizer_D_2(self.model_D_2.parameters(), **self.optimizer_D_2_opts)
        self.model_optimizer_T_1 = self.optimizer_T(self.model_T_1.parameters(), **self.optimizer_T_opts )
        self.model_optimizer_T_2 = self.optimizer_T_2(self.model_T_2.parameters(), **self.optimizer_T_2_opts )


        if checkpoint is not None:
            if 'epoch' in checkpoint:
                self.epoch_start = checkpoint['epoch']
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint:
                self.model_optimizer.load_state_dict(checkpoint['optimizer'])
            if 'loss' in checkpoint:
                self.loss_func = checkpoint['loss']

        self.model_D_1 = model_D_1.to(self.device)
        self.model_D_2 = model_D_2.to(self.device)
        self.model_T_1 = model_T_1.to(self.device)
        self.model_T_2 = model_T_2.to(self.device)
        if self.use_multigpu:
            print("Using multigpu for training")
            self.model_D_1 = nn.DataParallel(self.model_D_1)
            self.model_D_2 = nn.DataParallel(self.model_D_2)
            self.model_T_1 = nn.DataParallel( self.model_T_1)
            self.model_T_2 = nn.DataParallel( self.model_T_2)

        print("Training model for {} epochs".format(self.max_epochs))
        [plugin.on_train_start(self) for plugin in self.plugins]

        for epoch_num in range(self.epoch_start, self.max_epochs):
            if self.keep_training is False:
                break
            self.c_epoch = epoch_num
            # Train current epoch
            [plugin.on_train_epoch_start(self, epoch_num) for plugin in self.plugins]
            self.train_cyclegan_epoch()
            [plugin.on_train_gan_epoch_end(self, epoch_num) for plugin in self.plugins]

            # Validate on validation set
            [plugin.on_val_epoch_start(self, epoch_num) for plugin in self.plugins]
            self.validate_epoch_cyclegan(filepath=self.filepath,filepath2=self.filepath2)
            [plugin.on_val_gan_epoch_end(self, epoch_num) for plugin in self.plugins]

        print("Training finished\n")
        [plugin.on_train_end(self) for plugin in self.plugins]

    def train_gan_epoch(self):
        self.model_D.train()  # Set the models in training mode (for correct dropout, batchnorm, etc.)
        self.model_T.train()
        print('Training the models')
        # Prepare the metrics data structure
        self.train_metrics_D = dict()
        self.train_metrics_T = dict()
        b_size =self.bs
        for k in self.train_metric_funcs.keys():
            self.train_metrics_D['train_{}'.format( k )] = 0.0
            self.train_metrics_T['train_{}'.format( k )] = 0.0

        for batch_idx, (input, target) in enumerate( self.train_gen ):
            [plugin.on_train_batch_start( self, batch_idx ) for plugin in self.plugins]

            self.input, self.target = self._to_device( input, target )
            self.loss_D =self.train_D(self.input, self.target)
            self.loss_T = self.train_T(self.input,self.target)

            for k, eval_func in self.train_metric_funcs.items():  # Update training metrics
                self.train_metrics_D['train_{}'.format(k)] += self.loss_D.item()
                self.train_metrics_T['train_{}'.format(k)] += self.loss_T.item()
            self.image = self.model_T(self.input).detach()
            self.in_img = self.input
            self.tgt_img = self.target
            [plugin.on_train_batch_end( self, batch_idx ) for plugin in self.plugins]

        # Compute average metrics for epoch
        for k, v in self.train_metrics_D.items():
            self.train_metrics_D[k] = float( v / len( self.train_gen ) )
        for k, v in self.train_metrics_T.items():
            self.train_metrics_T[k] = float( v / len( self.train_gen ) )

    def train_cyclegan_epoch(self):
        #TO ADAPT TO CYCLEGAN
        self.model_D_1.train()  # Set the models in training mode (for correct dropout, batchnorm, etc.)
        self.model_D_2.train()  # Set the models in training mode (for correct dropout, batchnorm, etc.)
        self.model_T_1.train()
        self.model_T_2.train()
        print('Training the models')
        # Prepare the metrics data structure
        self.train_metrics_D_1 = dict()
        self.train_metrics_D_2 = dict()
        self.train_metrics_T_1 = dict()
        self.train_metrics_T_2 = dict()
        self.train_metrics_GAN = dict()
        b_size =self.bs
        for k in self.train_metric_funcs.keys():
            self.train_metrics_D_1['train_{}'.format( k )] = 0.0
            self.train_metrics_D_2['train_{}'.format( k )] = 0.0
            self.train_metrics_T_1['train_{}'.format( k )] = 0.0
            self.train_metrics_T_2['train_{}'.format( k )] = 0.0

        for batch_idx, (input, target) in enumerate( self.train_gen ):
            [plugin.on_train_batch_start( self, batch_idx ) for plugin in self.plugins]

            self.input, self.target = self._to_device( input, target )
            # self.middle = middle.to(self.device)
            # self.lesion =lesion.to(self.device)
            # self.loss_D_1, self.loss_D_2 =self.train_Discriminators(self.input, self.target, self.middle)
            self.loss_D_1, self.loss_D_2 =self.train_Discriminators(self.input, self.target)
            # self.loss_GAN= self.train_Translators(self.input,self.target, self.middle)
            # self.loss_T1, self.loss_T2= self.train_Translators2(self.input,self.target, self.middle)
            self.loss_T1, self.loss_T2 = self.train_Translators3(self.input,self.target)


            for k, eval_func in self.train_metric_funcs.items():  # Update training metrics
                self.train_metrics_D_1['train_{}'.format(k)] += self.loss_D_1.item()
                self.train_metrics_D_2['train_{}'.format(k)] += self.loss_D_2.item()
                self.train_metrics_T_1['train_{}'.format(k)] += self.loss_T1.item()
                self.train_metrics_T_2['train_{}'.format(k)] += self.loss_T2.item()
                # self.train_metrics_GAN['train_{}'.format(k)] += self.loss_GAN.item()
            [plugin.on_train_batch_cyclegan_end( self, batch_idx ) for plugin in self.plugins]

        # Compute average metrics for epoch
        for k, v in self.train_metrics_D_1.items():
            self.train_metrics_D_1[k] = float( v / len( self.train_gen ) )
        for k,v in self.train_metrics_T_1.items():
            self.train_metrics_T_1[k] = float( v / len( self.train_gen ) )
        for k, v in self.train_metrics_T_2.items():
            self.train_metrics_T_2[k] = float( v / len( self.train_gen ) )
        for k, v in self.train_metrics_D_2.items():
            self.train_metrics_D_2[k] = float( v / len( self.train_gen ) )

    def validate_epoch_cyclegan(self, filepath, filepath2):
        # I Validate only on the Translator
        self.filepath = filepath
        self.filepath2 = filepath2
        #take metrics for both the translators
        self.val_metrics_T_1 = dict()
        self.val_metrics_T_2 = dict()
        # self.tot_loss = dict()
        for k in self.val_metric_funcs.keys():
            self.val_metrics_T_1['val_{}'.format( k )] = 0.0
            self.val_metrics_T_2['val_{}'.format( k )] = 0.0
            # self.tot_loss['val_{}'.format( k )] = 0.0

        #Models in evaluating mode
        self.model_T_1.eval()
        self.model_T_2.eval()

        with torch.no_grad():
            for batch_idx, (input, target) in enumerate( self.val_gen ):
                [plugin.on_val_batch_start( self, batch_idx ) for plugin in self.plugins]
                # input, target = input.reshape( (self.bs, 32 * 32 * 32) ), target.reshape( (self.bs, 32 * 32 * 32) )
                x = input.to( self.device )
                y = target.to(self.device)
                G_output = self.model_T_1( x )
                F_output = self.model_T_2( y )

                for k, eval_func in self.val_metric_funcs.items():
                    self.val_metrics_T_1['val_{}'.format( k )] += eval_func( G_output,y )
                    self.val_metrics_T_2['val_{}'.format( k )] += eval_func( F_output,x )
                # for k, eval_func in self.val_metric_funcs.items():
                #     self.tot_loss['val_{}'.format(k)] += self.val_metrics_T_1['val_{}'.format(k)] + self.val_metrics_T_2['val_{}'.format(k)]
                [plugin.on_val_batch_end( self, batch_idx ) for plugin in self.plugins]

        # Compute average metrics
        for k, v in self.val_metrics.items():
            self.val_metrics_T_1[k] = float( v / len( self.val_gen ) )
            self.val_metrics_T_2[k] = float( v / len( self.val_gen ) )
            # self.tot_loss[k] = float(v/len( self.val_gen ))
            # print( 'Validation Translator Metrics: ', self.val_metrics_T[k] )
        print('............Checking Things.....')

        if self.save == 'best':
            monitored_metric_1_value = self.val_metrics_T_1['val_{}'.format(self.metric_name)]
            monitored_metric_2_value = self.val_metrics_T_2['val_{}'.format(self.metric_name)]

            metric_diff_1 = self.best_metric_1['value'] - monitored_metric_1_value
            metric_diff_2 = self.best_metric_2['value'] - monitored_metric_2_value
            if self.mode == 'max':
                metric_diff_1 *= -1.0
                metric_diff_2 *= -1.0

            if metric_diff_1 > self.min_delta:
                print('metric 1 = {} > min delta ({})'.format(metric_diff_1, self.min_delta))
                self.best_metric_1.update(epoch=self.c_epoch, value=monitored_metric_1_value)
                print("Saving best model at {}".format(self.filepath))
                torch.save(self.model_T_1, self.filepath)
            if metric_diff_2 > self.min_delta:
                print( 'metric 2 = {} > min delta ({})'.format( metric_diff_2, self.min_delta ) )
                self.best_metric_2.update(epoch=self.c_epoch, value=monitored_metric_2_value)
                print("Saving best model at {}".format(self.filepath2))
                torch.save(self.model_T_2, self.filepath2)


        elif self.save == 'last':
            torch.save(self.model_T_1, self.filepath)
            torch.save(self.model_T_2, self.filepath2)

        elif self.save == 'all':
            torch.save(self.model_T_1, remove_extension(self.filepath) + '_{}.pt'.format(self.c_epoch))
            torch.save(self.model_T_2, remove_extension(self.filepath2) + '_{}.pt'.format(self.c_epoch))
        else:
            raise(ValueError, 'Save mode not valid')

    def validate_epoch_gan(self, filepath):
        # I Validate only on the Translator
        self.filepath = filepath
        self.val_metrics_T = dict()
        for k in self.val_metric_funcs.keys():
            self.val_metrics_T['val_{}'.format( k )] = 0.0

        self.model_T.eval()

        with torch.no_grad():
            for batch_idx, (input, target) in enumerate( self.val_gen ):
                [plugin.on_val_batch_start( self, batch_idx ) for plugin in self.plugins]
                # input, target = input.reshape( (self.bs, 32 * 32 * 32) ), target.reshape( (self.bs, 32 * 32 * 32) )
                x = input.to( self.device )
                y = target.to(self.device)
                G_output = self.model_T( x )


                for k, eval_func in self.val_metric_funcs.items():
                    self.val_metrics_T['val_{}'.format( k )] += eval_func( G_output,y )
                [plugin.on_val_batch_end( self, batch_idx ) for plugin in self.plugins]

        # Compute average metrics
        for k, v in self.val_metrics.items():
            self.val_metrics_T[k] = float( v / len( self.val_gen ) )
            print( 'Validation Translator Metrics: ', self.val_metrics_T[k] )
        print('............Checking Things.....')

        if self.save == 'best':
            monitored_metric_value = self.val_metrics_T['val_{}'.format(self.metric_name)]
            metric_diff = self.best_metric['value'] - monitored_metric_value
            if self.mode == 'max':
                metric_diff *= -1.0

            if metric_diff > self.min_delta:
                self.best_metric.update(epoch=self.c_epoch, value=monitored_metric_value)
                print("Saving best model at {}".format(self.filepath))
                torch.save(self.model_T, self.filepath)

        elif self.save == 'last':
            torch.save(self.model_T, self.filepath)

        elif self.save == 'all':
            torch.save(self.model_T, remove_extension(self.filepath) + '_{}.pt'.format(self.c_epoch))
        else:
            raise(ValueError, 'Save mode not valid')


    def _to_device(self, x, y=None):

        if isinstance(x, list) or isinstance(x, tuple):
            x = tuple(self._to_device(x_n) for x_n in x)
        else:
            x = x.to(self.device)

        if y is None:
            return x

        if isinstance(y, list) or isinstance(y, tuple):
            y = tuple(self._to_device(y_n) for y_n in y)
        else:
            y = y.to(self.device)

        return x, y

class TrainerPlugin(ABC):
    """
    Abstract Base Class for subclassing training plugins. Subclasses can override the following callbacks:

    * ``on_init(self, trainer)``
    * ``on_train_start(self, trainer)``
    * ``on_train_end(self, trainer)``
    * ``on_train_epoch_start(self, trainer, num_epoch)``
    * ``on_train_epoch_end(self, trainer, num_epoch)``
    * ``on_val_epoch_start(self, trainer, num_epoch)``
    * ``on_val_epoch_end(self, trainer, num_epoch)``
    * ``on_train_batch_start(self, trainer, batch_idx)``
    * ``on_train_batch_end(self, trainer, batch_idx)``
    * ``on_val_batch_start(self, trainer, batch_idx)``
    * ``on_val_batch_end(self, trainer, batch_idx)``

    where:

    :param trainer: :py:class:`Trainer <niclib.net.train.Trainer>` instance that grants access to all its runtime variables (i.e. trainer.keep_training):
    :param num_epoch: current epoch number (starting at 0)
    :param batch_idx: current batch iteration (starting at 0)

    The training runtime variables accesible through the trainer object are:

    :var bool keep_training: control flag to continue or interrupt the training. Checked at the start of each epoch.
    :var torch.nn.Module model: the partially trained model.
    :var torch.optim.Optimizer model_optimizer: the model optimizer.
    :var torch.nn.Module loss_func: The loss function object
    :var torch.utils.data.DataLoader train_gen:
    :var torch.utils.data.DataLoader val_gen:
    :var dict train_metrics:
    :var dict val_metrics:
    :var torch.Tensor x:
    :var torch.Tensor y:
    :var torch.Tensor y_pred:
    :var torch.Tensor loss:

    :Example:

    # >>> # Here we define a plugin that will segment and compute metrics on test_images at the end of each epoch
    # >>> class EpochEvaluatorPlugin(TrainerPlugin):
    # >>>     def __init__(self, test_images, test_targets):
    # >>>         super().__init__()
    # >>>         self.images, self.targets = test_images, test_targets
    # >>>         self.predictor = niclib.net.test.PatchTester(
    # >>>             patch_shape=(1, 32, 32, 32),
    # >>>             patch_out_shape=(3, 32, 32, 32),
    # >>>             extraction_step=(16, 16, 16),
    # >>>             normalize='image',
    # >>>             activation=torch.nn.Softmax(dim=1))
    # >>>
    # >>>     def on_val_epoch_end(self, trainer, num_epoch):
    # >>>         pred_images = [self.predictor.predict(trainer.model, img) for img in self.images]
    # >>>         pred_metrics = niclib.metrics.compute_metrics(
    # >>>             outputs=[np.argmax(img, axis=0) for img in pred_images],
    # >>>             targets=self.targets,
    # >>>             metrics={
    # >>>                 'dsc': niclib.metrics.dsc,
    # >>>                 'acc': niclib.metrics.accuracy})
    # >>>         niclib.save_to_csv('metrics_epoch{}.csv'.format(num_epoch), pred_metrics)
    """
    def on_init(self, trainer):
        pass

    def on_train_start(self, trainer):
        pass
    def on_train_gan_start(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass
    def on_train_gan_end(self, trainer):
        pass

    def on_train_epoch_start(self, trainer, num_epoch):
        pass
    def on_train_gan_epoch_start(self, trainer, num_epoch):
        pass

    def on_train_epoch_end(self, trainer, num_epoch):
        pass
    def on_train_gan_epoch_end(self, trainer, num_epoch):
        pass

    def on_val_epoch_start(self, trainer, num_epoch):
        pass
    def on_val_gan_epoch_start(self, trainer, num_epoch):
        pass

    def on_val_epoch_end(self, trainer, num_epoch):
        pass
    def on_val_gan_epoch_end(self, trainer, num_epoch):
        pass

    def on_train_batch_start(self, trainer, batch_idx):
        pass

    def on_train_batch_end(self, trainer, batch_idx):
        pass

    def on_val_batch_start(self, trainer, batch_idx):
        pass

    def on_val_batch_end(self, trainer, batch_idx):
        pass
    def on_train_batch_cyclegan_end(self, trainer, batch_idx):
        pass


class ModelCheckpoint_GAN(TrainerPlugin):
    """
    :param filepath: filepath for checkpoint storage
    :param str save: save mode, either ``best``, ``all`` or ``last``.
    :param metric_name: (only for ``save='best'``) name of the monitored metric to evaluate the model and store.
    :param str mode: either ``min`` or ``max`` of the monitored metric to store the model.
    """

    def __init__(self, filepath, save='best', metric_name='loss', mode='min', min_delta=1e-4):
        assert save in {'best', 'all', 'last'}
        assert mode in {'min', 'max'}

        self.filepath = filepath
        self.save = save
        self.mode = mode
        self.metric_name = metric_name

        self.best_metric = None
        self.min_delta = min_delta

    def on_init(self, trainer):
        self.best_metric = dict(epoch=-1, value=sys.float_info.max)

    def on_val_epoch_end(self, trainer, num_epoch):
        if self.save == 'best':
            monitored_metric_value = trainer.val_metrics_T['val_{}'.format(self.metric_name)]



            metric_diff = self.best_metric['value'] - monitored_metric_value
            if self.mode == 'max':
                metric_diff *= -1.0

            if metric_diff > self.min_delta:
                self.best_metric.update(epoch=num_epoch, value=monitored_metric_value)
                print("Saving best model at {}".format(self.filepath))
                torch.save(trainer.model_T, self.filepath)


        elif self.save == 'last':
            torch.save(trainer.model_T, self.filepath)

        elif self.save == 'all':
            torch.save(trainer.model_T, remove_extension(self.filepath) + '_{}.pt'.format(num_epoch))
        else:
            raise(ValueError, 'Save mode not valid')

class EarlyStopping(TrainerPlugin):
    """
    Plugin to interrupt the training with the early stopping technique.

    :param filepath: Filepath where the best model will be stored.
    :param str metric_name: monitored metric name. The name has to exist as a key in the val_metrics provided to the :class:`Trainer`.
    :param str mode: either ``min`` or ``max`` of the monitored metric to stop the training.
    :param int patience: Number of epochs to wait since the last best model before interrupting training.
    :param min_delta: minimum change between metric values to consider a new best model.
    """

    def __init__(self, metric_name, mode, patience, min_delta=1e-5):
        self.min_delta = min_delta

        self.metric_name = metric_name
        self.mode = mode
        self.patience = patience

        # Runtime variables
        self.best_metric = None

    def on_init(self, trainer):
        self.best_metric = dict(epoch=-1, value=sys.float_info.max)

    def on_val_epoch_end(self, trainer, num_epoch):
        monitored_metric_value = trainer.val_metrics['val_{}'.format(self.metric_name)]

        metric_diff = self.best_metric['value'] - monitored_metric_value
        if self.mode == 'max':
            metric_diff *= -1.0

        if metric_diff > self.min_delta:
            self.best_metric.update(epoch=num_epoch, value=monitored_metric_value)

        # Interrupt training if metric didnt improve in last 'patience' epochs
        if num_epoch - self.best_metric['epoch'] >= self.patience:
            trainer.keep_training=False
    def on_train_batch_cyclegan_end(self, trainer, batch_idx):
        pass
class VisdomLogger(TrainerPlugin):
    def __init__(self, env):
        import visdom
        self.v = visdom.Visdom()
        self.env = env
        self.fake_score, self.real_score, self.D_loss, self.T_loss = None, None, None, None
        self.fake_t1_score = None
        self.fake_flair_score = None
        self.real_t1_score = None
        self.real_flair_score = None
        self.D_1_loss = None
        self.D_2_loss = None
        self.adv_loss = None
        self.cy_1, self.cy_2 = None, None
        self.cycle_loss = None
    def on_init(self, trainer):
        self.fake_score =[]
        self.cy_1, self.cy_2 = [], []
        self.fake_t1_score =[]
        self.fake_flair_score =[]
        self.real_t1_score =[]
        self.real_flair_score =[]
        self.real_score =[]
        self.D_loss =[]
        self.D_1_loss =[]
        self.D_2_loss =[]
        self.T_loss =[]
        self.adv_loss = []
        self.cycle_loss =  []


    def on_train_batch_end(self, trainer, batch_idx):
        self.fake_score.append(trainer.fake_score.mean().item())
        self.real_score.append(trainer.real_score.mean().item())
        self.D_loss.append(trainer.loss_D.item())
        self.T_loss.append(trainer.loss_T.item())
        #Create the graphs
        win =self.v.line(
            Y = np.column_stack((np.array( self.fake_score ), np.array(self.real_score))) ,
            X = np.column_stack((np.arange( len( self.fake_score ) ), np.arange(len(self.real_score)) )),
            env= self.env,
            win = 'Score',
            opts=dict( title='Fake vs Real Score', xlabel='Batches',
                       ylabel='Value'    , legend = ['Fake Score', 'Real Score'])
        )

        win2 = self.v.line(
            Y=np.column_stack( (np.array( self.D_loss ), np.array( self.T_loss )) ),
            X=np.column_stack( (np.arange( len( self.D_loss ) ), np.arange( len( self.T_loss ) )) ),
            env=self.env,
            win='Losses',
            opts=dict( title='Translator vs Discriminator Loss', xlabel='Batches',
                       ylabel='Value', legend=['Discriminator Loss', 'Translator Loss'] )
        )
        #Update the curves
        self.v.line(
            Y = np.column_stack( (np.array( self.fake_score ), np.array( self.real_score )) ),
            X = np.column_stack( (np.arange( len( self.fake_score ) ), np.arange( len( self.real_score ) )) ),
            env=self.env,
            update='replace',
            win=win,
            opts=dict( title='Fake vs Real Score', xlabel='Batches',
                       ylabel='Value' , legend = ['Fake Score', 'Real Score'])
        )

        self.v.line(
            Y=np.column_stack( (np.array( self.D_loss ), np.array( self.T_loss )) ),
            X=np.column_stack( (np.arange( len( self.D_loss ) ), np.arange( len( self.T_loss ) )) ),
            env=self.env,
            update='replace',
            win=win2,
            opts=dict( title='Translator vs Discriminator Loss', xlabel='Batches',
                       ylabel='Value', legend=['Discriminator Loss', 'Translator Loss'] )
        )


    def on_train_batch_cyclegan_end(self, trainer, batch_idx):

        self.fake_t1_score.append(trainer.fake_t1_score.mean().item())
        self.fake_flair_score.append(trainer.fake_flair_score.mean().item())
        self.real_t1_score.append(trainer.real_t1_score.mean().item())
        self.real_flair_score.append(trainer.real_flair_score.mean().item())



        self.D_1_loss.append(trainer.loss_D_1.item())
        self.D_2_loss.append(trainer.loss_D_2.item())
        self.adv_loss.append(trainer.loss_T1.item())
        self.cycle_loss.append(trainer.loss_T2.item())
        self.cy_1.append(trainer.cycle_loss_1.item())
        self.cy_2.append(trainer.cycle_loss_2.item())

            # self.T_loss.append(trainer.loss_GAN.item())

            # self.adv_loss.append(trainer.adv_loss.item())
            #Create the graphs
        if batch_idx % 15 == 0:
            win =self.v.line(
                Y = np.column_stack((np.array( self.fake_t1_score ), np.array(self.real_t1_score),
                                     np.array( self.fake_flair_score ), np.array(self.real_flair_score))) ,
                X = np.column_stack((np.arange( len( self.fake_t1_score ) ), np.arange(len(self.real_t1_score)),
                                     np.arange( len( self.fake_flair_score ) ), np.arange(len(self.real_flair_score)))),
                env= self.env,
                win = 'Score',
                opts=dict( title='Fake vs Real Scores', xlabel='Batches',
                           ylabel='Value'    , legend = ['Fake A Score', 'Real A Score' , 'Fake B Score', 'Real B Score' ])
            )

            win2 = self.v.line(
                Y=np.column_stack( (np.array( self.D_1_loss ), np.array( self.D_2_loss )) ),
                X=np.column_stack( (np.arange( len( self.D_1_loss ) ), np.arange( len( self.D_2_loss ) )) ),
                env=self.env,
                win='Discriminators Losses',
                opts=dict( ytickmin = 0.0001, ytickmax = 5,title='D1 vs D2 Loss', xlabel='Batches',
                           ylabel='Value', legend=['D1 Loss', 'D2 Loss'] )
            )
            win3 = self.v.line(
                Y=np.column_stack( (np.array( self.adv_loss ), np.array( self.cycle_loss ), np.array(self.cy_1), np.array(self.cy_2)) ),
                X=np.column_stack( (np.arange( len( self.adv_loss ) ), np.arange( len( self.cycle_loss ) ), np.arange(len(self.cy_1)), np.arange(len(self.cy_2))) ),
                env=self.env,
                win='Translator Losses',
                opts=dict( title='Translators Losses Comparison', xlabel='Batches',
                           ylabel='Value', legend=['T_1 Loss', 'T_2 Loss', 'Cycle G(F(x))', 'Cycle F(G(y))'] )
            )
            #Update the curves
            self.v.line(
                Y=np.column_stack( (np.array( self.fake_t1_score ), np.array( self.real_t1_score ),
                                    np.array( self.fake_flair_score ), np.array( self.real_flair_score )) ),
                X=np.column_stack( (np.arange( len( self.fake_t1_score ) ), np.arange( len( self.real_t1_score ) ),
                                    np.arange( len( self.fake_flair_score ) ), np.arange( len( self.real_flair_score ) )) ),
                env=self.env,
                update='replace',
                win=win,
                opts=dict( title='Fake vs Real Scores', xlabel='Batches',
                           ylabel='Value' , legend = ['Fake A Score', 'Real A Score' , 'Fake B Score', 'Real B Score' ])
            )

            self.v.line(
                Y=np.column_stack( (np.array( self.D_1_loss ), np.array( self.D_2_loss )) ),
                X=np.column_stack( (np.arange( len( self.D_1_loss ) ), np.arange( len( self.D_2_loss ) )) ),
                env=self.env,
                update='replace',
                win=win2,
                opts=dict( ytickmin = 0.0001, ytickmax = 5,title='D1 vs D2 Loss', xlabel='Batches',
                           ylabel='Value', legend=['D1 Loss', 'D2 Loss'] )
            )

            # self.v.line(
            #     Y=np.column_stack( (np.array( self.adv_loss ), np.array( self.cycle_loss )) ),
            #     X=np.column_stack( (np.arange( len( self.adv_loss ) ), np.arange( len( self.cycle_loss ) )) ),
            #     env=self.env,
            #     update='replace',
            #     win=win3,
            #     opts=dict( title='Adversarial vs Cycle Consistency Loss', xlabel='Batches',
            #                ylabel='Value', legend=['Adversarial Loss', 'Cycle Consistency Loss'] )
            # )
            self.v.line(
                Y=np.column_stack( (np.array( self.adv_loss ), np.array( self.cycle_loss ), np.array(self.cy_1), np.array(self.cy_2)) ),
                X=np.column_stack( (np.arange( len( self.adv_loss ) ), np.arange( len( self.cycle_loss ) ), np.arange(len(self.cy_1)), np.arange(len(self.cy_2))) ),
                env=self.env,
                update='replace',
                win=win3,
                opts=dict(title='Translators Losses Comparison', xlabel='Batches',
                          ylabel='Value', legend=['T_1 Loss', 'T_2 Loss', 'Cycle F(G(x))', 'Cycle G(F(y))'] )
            )
class ProgressBarGAN(TrainerPlugin):
    """Plugin to print a progress bar and metrics during the training procedure.

    :param float print_interval: time in seconds between print updates.
    """
    def __init__(self, log_path, print_interval=0.4, printa = False):
        self.print_interval = print_interval
        self.writer = sw(log_path)
        # Runtime variables
        self.eta = None
        self.print_flag = True
        self.D_1_loss, self.D_2_loss, self.T_1_loss,self.T_2_loss = [],[],[],[]
        self.print_lock = threading.Lock()
        self.print_timer = None
        self.printa = printa

    def on_train_epoch_start(self, trainer, num_epoch):
        print("\nEpoch {}/{}".format(num_epoch+1, trainer.max_epochs))

        self.eta = RemainingTimeEstimator(len(trainer.train_gen))
        self.print_flag = True
        self.print_timer = None

    def on_train_gan_epoch_start(self, trainer, num_epoch):
        print( "\nEpoch {}/{}".format( num_epoch+1, trainer.max_epochs ) )

        self.eta = RemainingTimeEstimator( len( trainer.train_gen ) )
        self.print_flag = True
        self.print_timer = None

    def on_train_batch_end(self, trainer, batch_idx):
        # PRINTING LOGIC
        self.print_lock.acquire()
        if self.print_flag:
            # Compute average metrics
            avg_metrics_T = dict()
            avg_metrics_D = dict()
            for k, v in trainer.train_metrics_T.items():
                avg_metrics_T[k] = float(v / (batch_idx + 1))
            for k, v in trainer.train_metrics_D.items():
                avg_metrics_D[k] = float(v / (batch_idx + 1))
            if self.printa == True:
                self._printProgressBar(batch_idx, len(trainer.train_gen), self.eta.update(batch_idx + 1), avg_metrics_T, avg_metrics_D,self.printa)
            else:
                self._printProgressBar(batch_idx, len(trainer.train_gen), self.eta.update(batch_idx + 1))
            self.print_flag, self.print_timer = False, threading.Timer(self.print_interval, self._setPrintFlag)
            self.print_timer.start()

        self.print_lock.release()
    def on_train_batch_cyclegan_end(self, trainer, batch_idx):
        # PRINTING LOGIC
        self.print_lock.acquire()
        if self.print_flag:
            # Compute average metrics
            avg_metrics_T_1 = dict()
            avg_metrics_T_2 = dict()
            avg_metrics_D_1 = dict()
            avg_metrics_D_2 = dict()
            # self.fake_t1_score.append(trainer.fake_t1_score.mean().item())
            # self.fake_flair_score.append(trainer.fake_flair_score.mean().item())
            # self.real_t1_score.append(trainer.real_t1_score.mean().item())
            # self.real_flair_score.append(trainer.real_flair_score.mean().item())


            self.D_1_loss.append(trainer.loss_D_1.item())
            self.D_2_loss.append(trainer.loss_D_2.item())
            self.T_1_loss.append(trainer.loss_T1.item())
            self.T_2_loss.append(trainer.loss_T2.item())

            for k, v in trainer.train_metrics_T_1.items():
                avg_metrics_T_1[k] = float(v / (batch_idx + 1))
            for k, v in trainer.train_metrics_T_2.items():
                avg_metrics_T_2[k] = float( v / (batch_idx + 1) )
            for k, v in trainer.train_metrics_D_1.items():
                avg_metrics_D_1[k] = float(v / (batch_idx + 1))
            for k, v in trainer.train_metrics_D_2.items():
                avg_metrics_D_2[k] = float( v / (batch_idx + 1) )
            # self.writer.add_scalar('T_1 Loss', trainer.loss_T1.item(), len(trainer.train_gen) )
            # self.writer.add_scalar('T_2 Loss', trainer.loss_T2.item(), len(trainer.train_gen) )
            # self.writer.add_scalar('D_1 Loss', trainer.loss_D_1.item(), len(trainer.train_gen) )
            # self.writer.add_scalar('D_1 Loss', trainer.loss_D_2.item(), len(trainer.train_gen) )
            # self.writer.close()
            if self.printa == True:
                self._printProgressBar(batch_idx, len(trainer.train_gen), self.eta.update(batch_idx + 1),
                                       avg_metrics_T_1, avg_metrics_D_1,self.printa)
            else:
                self._printProgressBar(batch_idx, len(trainer.train_gen), self.eta.update(batch_idx + 1))
            self.print_flag, self.print_timer = False, threading.Timer(self.print_interval, self._setPrintFlag)
            self.print_timer.start()

        self.print_lock.release()

    def on_train_epoch_end(self, trainer, num_epoch):
        self.print_timer.cancel()
        self._printProgressBar(len(trainer.train_gen), len(trainer.train_gen), self.eta.elapsed_time(), trainer.train_metrics_T, trainer.train_metrics_D)

    def on_train_gan_epoch_end(self, trainer, num_epoch):
        self.print_timer.cancel()
        self._printProgressBar(len(trainer.train_gen), len(trainer.train_gen), self.eta.elapsed_time(), trainer.train_metrics_T, trainer.train_metrics_D)
    def on_val_epoch_end(self, trainer, num_epoch):
        for k, v in trainer.val_metrics.items():
            print(' - {}={:0<6.4f}'.format(k, v), end='')
        print('')

    def on_val_gan_epoch_end(self, trainer, num_epoch):
        for k, v in trainer.val_metrics_T.items():
            print( ' - {}={:0<6.4f}'.format( k, v ), end='' )
        print( '' )

    def _setPrintFlag(self, value=True):
        self.print_lock.acquire()
        self.print_flag = value
        self.print_lock.release()

    def _printProgressBar(self, batch_num, total_batches, eta,metrics_A='0.0', metrics_B = '0.0', printa = False):
        length, fill = 25, '='
        percent = "{0:.1f}".format(100 * (batch_num / float(total_batches)))
        filledLength = int(length * batch_num // total_batches)
        bar = fill * filledLength + '>' * min(length - filledLength, 1) + '.' * (length - filledLength - 1)
        if printa == True:
            metrics_stringa = ' - '.join(['{}={:0<6.4f}'.format(k, v) for k,v in metrics_A.items()])
            metrics_stringb = ' - '.join(['{}={:0<6.4f}'.format(k, v) for k,v in metrics_B.items()])
            print('\r [{}] {}/{} ({}%) ETA {} - Translator {} - Discriminator {}'.format(
                bar, batch_num, total_batches, percent, eta, metrics_stringa, metrics_stringb), end='')
        else:
            print('\r [{}] {}/{} ({}%) ETA {} '.format(bar, batch_num, total_batches, percent, eta), end='')

        sys.stdout.flush()
class Logger(TrainerPlugin):
    """Plugin that stores the time and average training and validation metrics for each epoch.

    :param str filepath: filepath to store the log as a .csv file.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.all_metrics = []

    def on_val_epoch_end(self, trainer, num_epoch):
        epoch_metrics = {'Epoch': num_epoch, 'Time finished': get_timestamp()}
        epoch_metrics.update(trainer.train_metrics.items())
        epoch_metrics.update(trainer.val_metrics.items())
        self.all_metrics.append(epoch_metrics)
        save_to_csv(self.filepath, self.all_metrics)


