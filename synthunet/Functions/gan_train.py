import numpy as np
import torch
import os

from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
import nibabel as nib
from synthunet.Functions.plotter import *
import synthunet.Functions.models as mod
from torch.utils.data.dataset import Dataset
from torch.utils.data import dataloader
from skimage.transform import resize


BATCH_SIZE=4
max_epoch = 100
gpu = True
workers = 12

reg = 5e-10

gamma = 20
beta = 10

#setting latent variable sizes
latent_dim = 1000

#Pass the data to my trainer (DATALOADER)
#train_loader = my data (train_gen)



#MY MODELS
G = mod.ResUNet(inputsize=2, outputsize=1)
D = mod.Discriminator()

def traingan(D,G,train_gen, N_EPOCH =10, batch_size =32):

    G.cuda()
    D.cuda()

    g_optimizer = optim.Adam(G.parameters(), lr=0.0001)
    d_optimizer = optim.Adam(D.parameters(), lr=0.0001)

    real_y = Variable(torch.ones((batch_size, 1)).cuda())
    fake_y = Variable(torch.zeros((batch_size, 1)).cuda())
    criterion_bce = nn.BCELoss()
    criterion_l1 = nn.L1Loss()

    for epoch in range( N_EPOCH ):
        for step, real_images in enumerate( train_gen ):
            _batch_size = real_images.size( 0 )
            real_images = Variable( real_images, requires_grad=False ).cuda()
            z_rand = Variable( torch.randn( (_batch_size, latent_dim) ), requires_grad=False ).cuda()
            x_rand = G( z_rand )
            ###############################################
            # Train D
            ###############################################
            d_optimizer.zero_grad()

            d_real_loss = criterion_bce( D( real_images ), real_y[:_batch_size] )
            d_fake_loss = criterion_bce( D( x_rand ), fake_y[:_batch_size] )

            dis_loss = d_real_loss + d_fake_loss
            dis_loss.backward( retain_graph=True )
            d_optimizer.step()

            ###############################################
            # Train G
            ###############################################
            g_optimizer.zero_grad()
            output = D( real_images )
            d_real_loss = criterion_bce( output, real_y[:_batch_size] )
            d_recon_loss = criterion_bce( output, fake_y[:_batch_size] )
            output = D( x_rand )
            d_fake_loss = criterion_bce( output, fake_y[:_batch_size] )

            d_img_loss = d_real_loss + d_recon_loss + d_fake_loss
            gen_img_loss = -d_img_loss

            # rec_loss = ((x_rec - real_images) ** 2).mean()

            # err_dec = gamma * rec_loss + gen_img_loss
            err_dec = gen_img_loss
            err_dec.backward( retain_graph=True )
            g_optimizer.step()


            ###############################################
            # Visualization
            ###############################################
            #
            if step % 10 == 0:
                print( '[{}/{}]'.format( epoch, N_EPOCH ),
                       'D: {:<8.3}'.format( dis_loss.data[0].cpu().numpy() ),
                       # 'En: {:<8.3}'.format( err_enc.data[0].cpu().numpy() ),
                       'De: {:<8.3}'.format( err_dec.data[0].cpu().numpy() )
                       )
            #
            #     featmask = np.squeeze( (0.5 * real_images[0] + 0.5).data.cpu().numpy() )
            #     featmask = nib.Nifti1Image( featmask, affine=np.eye( 4 ) )
            #     plotting.plot_img( featmask, title="X_Real" )
            #     plotting.show()
            #
            #     featmask = np.squeeze( (0.5 * x_rec[0] + 0.5).data.cpu().numpy() )
            #     featmask = nib.Nifti1Image( featmask, affine=np.eye( 4 ) )
            #     plotting.plot_img( featmask, title="X_DEC" )
            #     plotting.show()
            #
            #     featmask = np.squeeze( (0.5 * x_rand[0] + 0.5).data.cpu().numpy() )
            #     featmask = nib.Nifti1Image( featmask, affine=np.eye( 4 ) )
            #     plotting.plot_img( featmask, title="X_rand" )
            #     plotting.show()

        torch.save( G.state_dict(), './chechpoint/G_VG_ep_' + str( epoch + 1 ) + '.pth' )
        torch.save( D.state_dict(), './chechpoint/D_VG_ep_' + str( epoch + 1 ) + '.pth' )