# %% 
"""
wgan with different loss function, used the pure dcgan structure.
"""
import os 
import time
import torch
import datetime

import torch.nn as nn
import torchvision

from models.sagan import Generator, Discriminator
from utils.utils import *

# %%
class Trainer_sagan(object):
    def __init__(self, data_loader, config):
        super(Trainer_sagan, self).__init__()

        # data loader 
        self.data_loader = data_loader

        # exact model and loss 
        self.model = config.model
        self.adv_loss = config.adv_loss#gan

        # model hyper-parameters
        self.imsize = config.img_size #64
        self.g_num = config.g_num #5 train the generator every 5 steps
        self.z_dim = config.z_dim #100 noise dim
        self.channels = config.channels
        self.g_conv_dim = config.g_conv_dim #64
        self.d_conv_dim = config.d_conv_dim #64

        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers 
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr 
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model  #None

        self.dataset = config.dataset #哪个数据集
        self.use_tensorboard = config.use_tensorboard
        # path
        self.image_path = config.dataroot #'../data'
        self.log_path = config.log_path #'./logs'
        self.sample_path = config.sample_path #'./samples'
        self.log_step = config.log_step #10
        self.sample_step = config.sample_step #100
        self.version = config.version #test

        # path with version
        self.log_path = os.path.join(config.log_path, self.version) #'./logs/test'
        self.sample_path = os.path.join(config.sample_path, self.version)# './samples/test'

        if self.use_tensorboard:
            self.build_tensorboard() #一个函数

        self.build_model()#一个函数

    def train(self):
        '''Training'''
        # fixed input for debugging
        fixed_z = tensor2var(torch.randn(1000, self.z_dim, 1, 1)) # (100, 100, 1, 1)  tensor2var是utils的一个函数

        for epoch in range(self.epochs):
            # start time
            start_time = time.time()

            for i, (real_images, _) in enumerate(self.data_loader):

                # configure input 
                real_images = tensor2var(real_images)
                
                # adversarial ground truths
                valid = tensor2var(torch.full((real_images.size(0),), 0.9)) # (*, ) #torch.full：用0.9填充，形状为real_images.size(0)
                fake = tensor2var(torch.full((real_images.size(0),), 0.0)) #(*, )
                
                # ==================== Train D ==================
                self.D.train()
                self.G.train()

                #判别器
                self.D.zero_grad()

                # compute loss with real images 
                d_out_real = self.D(real_images)

                if self.adv_loss == 'gan':
                    d_loss_real = self.adversarial_loss_sigmoid(d_out_real, valid)
                elif self.adv_loss == 'hinge':
                    d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

                # noise z for generator
                z = tensor2var(torch.randn(real_images.size(0), self.z_dim, 1, 1)) # 64, 100, 1, 1

                fake_images = self.G(z) # (*, c, 64, 64)
                d_out_fake = self.D(fake_images) # (*,)

                if self.adv_loss == 'gan':
                    d_loss_fake = self.adversarial_loss_sigmoid(d_out_fake, fake)
                elif self.adv_loss == 'hinge':
                    d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

                # total d loss
                d_loss = d_loss_real + d_loss_fake

                d_loss.backward()
                # update D
                self.d_optimizer.step()

                # train the generator every 5 steps 生成器，每self.g_num轮跑一次
                if i % self.g_num == 0:

                    # =================== Train G and gumbel =====================
                    self.G.zero_grad()
                    # create random noise 
                    fake_images = self.G(z)

                    # compute loss with fake images 
                    g_out_fake = self.D(fake_images) # batch x n

                    if self.adv_loss == 'gan':
                        g_loss_fake = self.adversarial_loss_sigmoid(g_out_fake, valid)
                    elif self.adv_loss == 'hinge':
                        g_loss_fake = - torch.mean(g_out_fake)

                    g_loss_fake.backward()
                    # update G
                    self.g_optimizer.step()

            # log to the tensorboard
            self.logger.add_scalar('d_loss', d_loss.data, epoch)#判别器损失
            self.logger.add_scalar('g_loss_fake', g_loss_fake.data, epoch)#生成器损失
            # end one epoch

            # print out log info   10
            if (epoch) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out: {:.4f}, g_loss: {:.4f}, "
                    .format(elapsed, epoch, self.epochs, epoch,
                            self.epochs, d_loss.item(), g_loss_fake.item()))

            # sample images    100
            if (epoch) % self.sample_step == 0:
                self.G.eval()
                # save real image
                save_sample(self.sample_path + '/real_images/', real_images, epoch)
                
                with torch.no_grad():
                    fake_images = self.G(fixed_z)
                    # save fake image 
                    save_sample(self.sample_path + '/fake_images/', fake_images, epoch)
                    
                # sample sample one images
                save_sample_one_image(self.sample_path, real_images, fake_images, epoch)


    def build_model(self):

        self.G = Generator(image_size = self.imsize, z_dim = self.z_dim, conv_dim = self.g_conv_dim, channels = self.channels).cuda()
        self.D = Discriminator(image_size = self.imsize, conv_dim = self.d_conv_dim, channels = self.channels).cuda()
    
        # apply the weights_init to randomly initialize all weights
        # to mean=0, stdev=0.2
        self.G.apply(weights_init)
        self.D.apply(weights_init)
        
        # optimizer 
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        # for orignal gan loss function
        self.adversarial_loss_sigmoid = nn.BCEWithLogitsLoss()

        # print networks
        print(self.G)
        print(self.D)

    def build_tensorboard(self):
        from torch.utils.tensorboard import SummaryWriter
        self.logger = SummaryWriter(self.log_path)

    def save_image_tensorboard(self, images, text, step):
        if step % 100 == 0:
            img_grid = torchvision.utils.make_grid(images, nrow=8)

            self.logger.add_image(text + str(step), img_grid, step)
            self.logger.close()
