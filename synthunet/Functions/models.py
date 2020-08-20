import torch
from torch.nn import functional as F
import torch.nn as nn
import functools
from synthunet.Functions.spectral_norm import SpectralNorm
"""
HERE WE WILL PUT ALL OUR MODELS
"""

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        #input size = 32x32x32xin
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        #32x32x632xout
        nn.ReLU(inplace=True),
        #32x32x64
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
def conv(in_channels, out_channels):
    return nn.Sequential(
        #input size = 32x32x32xin
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        #32x32x632xout
        nn.LeakyReLU(0.2,inplace=True)
        #32x32x64
    )
def double_conv_1(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

class Unet(nn.Module):
    """
    Basic U-net model
    """

    def __init__(self, input_size, output_size):
        super(Unet, self).__init__()

        # conv1 down
        self.conv1 = nn.Conv3d(in_channels=input_size,out_channels=32,kernel_size=3, padding=1)
        # max-pool 1
        self.pool1 = nn.Conv3d(in_channels=32,out_channels=32,kernel_size=2,stride=2)
        # conv2 down
        self.conv2 = nn.Conv3d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        # max-pool 2
        self.pool2 = nn.Conv3d(in_channels=64,out_channels=64, kernel_size=2,stride=2)
        # conv3 down
        self.conv3 = nn.Conv3d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        # max-pool 3
        self.pool3 = nn.Conv3d(in_channels=128,out_channels=128,kernel_size=2,stride=2)
        # conv4 down (latent space)
        self.conv4 = nn.Conv3d(in_channels=128,out_channels=256,kernel_size=3,padding=1)
        self.pool4 = nn.Conv3d(256,256,2,2)

        self.conv9 = nn.Conv3d( in_channels=256, out_channels=512, kernel_size=3, padding=1 )
        # up-sample conv4
        # self.upsample = nn.Upsample( scale_factor=2, mode='trilinear', align_corners=True )
        self.up5 = nn.ConvTranspose3d(512,256,2,2)
        self.conv10 = nn.Conv3d(256,256,3,1)
        self.up1 = nn.ConvTranspose3d(in_channels=256, out_channels=128,kernel_size=2,stride=2)
        # conv 5 (add up1 + conv3)
        self.conv5 = nn.Conv3d(in_channels=128,out_channels=128,kernel_size=3,padding=1)
        # up-sample conv5
        self.up2 = nn.ConvTranspose3d(in_channels=128,out_channels=64,kernel_size=2,stride=2)
        # conv6 (add up2 + conv2)
        self.conv6 = nn.Conv3d(in_channels=64,out_channels=64,kernel_size=3,padding=1)
        # up 3
        self.up3 = nn.ConvTranspose3d(in_channels=64,out_channels=32,kernel_size=2,stride=2)
        # conv7 (add up3 + conv1)
        self.conv7 = nn.Conv3d(in_channels=32,out_channels=32,kernel_size=3,padding=1)
        # conv8 (classification)
        self.conv8 = nn.Conv3d(in_channels=32,out_channels=output_size,kernel_size=1)
        # self.softmax = nn.Sigmoid()

    def forward(self, x):
        # encoder
        x1 = F.relu(self.conv1(x))
        x1p = self.pool1(x1)
        x2 = F.relu(self.conv2(x1p))
        x2p = self.pool2(x2)
        x3 = F.relu(self.conv3(x2p))
        x3p = self.pool3(x3)

        # latent space
        x4 = F.relu(self.conv4(x3p))
        # x4p = self.pool4(x4)
        #
        # x9 = F.relu(self.conv9(x4p))
        # up10 = self.up5(x9)
        # x10 = F.relu(self.conv10(up10+x4))


        # decoder
        up1 = self.up1(x4)
        x5 = F.relu(self.conv5(up1 + x3))  # look how layers are added :o
        up2 = self.up2(x5)
        x6 = F.relu(self.conv6(up2 + x2))
        up3 = self.up3(x6)
        x7 = F.relu(self.conv7(up3 + x1))

        # output layer (1 classes)
        out = self.conv8(x7)
        # out = self.softmax(out) #I should have it because of the range adjust in thei nput
        return out



class Unet_Upsampled(nn.Module):
    """
    Basic U-net model with upsampling instead of transposeconvolutions
    """

    def __init__(self, input_size, output_size):
        super(Unet_Upsampled, self).__init__()

        self.dconv_down1 = double_conv( input_size, 64 )
        self.dconv_down2 = double_conv( 64, 128 )
        self.dconv_down3 = double_conv( 128, 256 )
        self.dconv_down4 = double_conv( 256, 512 )

        self.maxpool = nn.MaxPool3d(2)
        # self.down_1 = nn.Conv3d(64,64,kernel_size=2,stride=2)
        # self.down_2 = nn.Conv3d(128,128,kernel_size=2,stride=2)
        # self.down_3 = nn.Conv3d(256,256,kernel_size=2,stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.dconv_up3 = double_conv( 256 + 512, 256 )
        self.dconv_up2 = double_conv( 128 + 256, 128 )
        self.dconv_up1 = double_conv( 128 + 64, 64 )


        self.conv_last = nn.Conv3d( 64, output_size, 1 )


    def forward(self, x):
        conv1 = self.dconv_down1( x )
        x = self.maxpool( conv1 )
        # x = self.down_1(conv1)
        conv2 = self.dconv_down2( x )
        x = self.maxpool( conv2 )
        # x = self.down_2( conv2 )
        conv3 = self.dconv_down3( x )
        x = self.maxpool( conv3 )
        # x = self.down_3( conv3 )
        x = self.dconv_down4( x )
        #4x4x4x512

        x = self.upsample( x )
        #8x8x8x512
        x = torch.cat( [x, conv3], dim=1 )
        #8x8x8x512 + 8x8x8x256

        x = self.dconv_up3( x )
        # 8x8x8x256
        x = self.upsample( x )
        #16x16x16x256
        x = torch.cat( [x, conv2], dim=1 )
        # 16x16x16x256 + #8x8x8x128
        x = self.dconv_up2( x )
        x = self.upsample( x )
        x = torch.cat( [x, conv1], dim=1 )

        x = self.dconv_up1( x )

        out = self.conv_last( x )
        return out
class Unet_1(nn.Module):
    """
    Basic U-net model with upsampling instead of transposeconvolutions
    """

    def __init__(self, input_size, output_size):
        super(Unet_1, self).__init__()

        self.dconv_down1 = conv( input_size, 64 )
        self.dconv_down2 = conv( 64, 128 )
        self.dconv_down3 = conv( 128, 256 )
        self.dconv_down4 = conv( 256, 512 )

        # self.maxpool = nn.MaxPool3d(2)

        self.down_1 = nn.Conv3d(64,64,kernel_size=2,stride=2)
        self.down_2 = nn.Conv3d(128,128,kernel_size=2,stride=2)
        self.down_3 = nn.Conv3d(256,256,kernel_size=2,stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.dconv_up3 = conv( 256 + 512, 256 )
        self.dconv_up2 = conv( 128 + 256, 128 )
        self.dconv_up1 = conv( 128 + 64, 64 )


        self.conv_last = nn.Conv3d( 64, output_size, 1 )


    def forward(self, x):
        conv1 = self.dconv_down1( x )
        # x = self.maxpool( conv1 )
        x = self.down_1(conv1)
        conv2 = self.dconv_down2( x )
        # x = self.maxpool( conv2 )
        x = self.down_2( conv2 )
        conv3 = self.dconv_down3( x )
        # x = self.maxpool( conv3 )
        x = self.down_3( conv3 )
        x = self.dconv_down4( x )
        #4x4x4x512

        x = self.upsample( x )
        #8x8x8x512
        x = torch.cat( [x, conv3], dim=1 )
        #8x8x8x512 + 8x8x8x256

        x = self.dconv_up3( x )
        # 8x8x8x256
        x = self.upsample( x )
        #16x16x16x256
        x = torch.cat( [x, conv2], dim=1 )
        # 16x16x16x256 + #8x8x8x128
        x = self.dconv_up2( x )
        x = self.upsample( x )
        x = torch.cat( [x, conv1], dim=1 )

        x = self.dconv_up1( x )

        out = self.conv_last( x )
        return out
class TryDis(nn.Module):
    def __init__(self,input_size):
        super(TryDis, self).__init__()
        self.dconv_down1 =(double_conv( input_size, 64 ))
        self.dconv_down2 = (double_conv( 64, 128 ))
        self.dconv_down3 = (double_conv( 128, 256 ))
        self.dconv_down4 = (double_conv( 256, 512 ))
        self.down_1 = nn.Conv3d(64,64,kernel_size=2,stride=2)
        self.down_2 = (nn.Conv3d(128,128,kernel_size=2,stride=2))
        self.down_3 = (nn.Conv3d(256,256,kernel_size=2,stride=2))
        # self.maxpool = nn.MaxPool3d(2)
        self.seq = nn.Sequential(
            nn.Linear( 512, 1024 ),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear( 1024, 64 ),
            nn.LeakyReLU(0.2, inplace=True ),
            nn.Linear( 64, 1 )
        )


    def forward(self, x):
        conv1 = self.dconv_down1( x )
        # x = self.maxpool( conv1 )
        x = self.down_1(conv1)
        conv2 = self.dconv_down2( x )
        # x = self.maxpool( conv2 )
        x = self.down_2( conv2 )
        conv3 = self.dconv_down3( x )
        # x = self.maxpool( conv3 )
        x = self.down_3( conv3 )
        x = self.dconv_down4( x )
        feature = x.view( -1, 512 )
        out = self.seq(feature)
        return out



class ResUNetBody(nn.Module):

    def __init__(self, k=16, outputsize=2, inputsize=4, option = False):
        super(ResUNetBody, self).__init__()
        self.outputsize = outputsize
        self.option = option
        self.inputsize = inputsize
        self. k = k

        self.conv = ConvBlock(inputsize, k, 3)
        self.denseBlock1 = DenseBlock(n=2, inputsize=k)
        self.TD1 = TD(inputsize=k, outputsize=k*2)
        self.denseBlock2 = DenseBlock(n=2, inputsize=k*2)
        self.TD2 = TD(inputsize=k*2, outputsize=k*4)
        self.denseBlock3 = DenseBlock(n=2, inputsize=k*4)
        self.TD3 = TD(inputsize=k*4, outputsize=k * 4)
        self.denseBlock4 = DenseBlock(n=2, inputsize=k*4)
        self.TD4 = TD(inputsize=k*4, outputsize=k*4)
        self.denseBlockmid = DenseBlock(n=2, inputsize=k*4)
        if option:
            self.TD5 = TD(inputsize = k*4, outputsize= k*6)
            self.denseBlock5 = DenseBlock(n =2, inputsize=k*6)
            self.TD6 = TD(inputsize = k*6, outputsize= k*8)
            self.denseBlock6 = DenseBlock(n =2, inputsize=k*8)
            self.TD7 = TD(inputsize = k*8, outputsize= k*8)
            self.denseBlock7 = DenseBlock(n =2, inputsize=k*8)
            self.UP7 = nn.ConvTranspose3d(k*8, k*8, 2, stride=2)
            self.denseBlock7_right = DenseBlock( n=2, inputsize=k * 8 )
            self.UP6 = nn.ConvTranspose3d( k * 8, k * 6, 2, stride=2 )
            self.denseBlock6_right = DenseBlock( n=2, inputsize=k * 6 )
            self.UP5 = nn.ConvTranspose3d( k * 6, k * 4, 2, stride=2 )
            self.denseBlock5_right = DenseBlock( n=2, inputsize=k * 4 )
        self.UP1 = nn.ConvTranspose3d(k*4, k*4, 2, stride=2)
        self.denseBlock4_right = DenseBlock(n=2, inputsize=k*8)
        self.UP2 = nn.ConvTranspose3d(k*8, k*4, 2, stride=2)
        self.denseBlock3_right = DenseBlock(n=2, inputsize=k*8)
        self.UP3 = nn.ConvTranspose3d(k*8, k*2, 2, stride=2)
        self.denseBlock2_right = DenseBlock(n=2, inputsize=k*4)
        self.UP4 = nn.ConvTranspose3d(k*4, k*1, 2, stride=2)
        self.denseBlock1_right = DenseBlock(n=2, inputsize=k*2)

    def forward(self, x):
        res = self.conv(x)
        res = self.denseBlock1(res)
        skip1 = res.clone()
        res = self.TD1(res)
        res = self.denseBlock2(res)
        skip2 = res.clone()
        res = self.TD2(res)
        res = self.denseBlock3(res)
        skip3 = res.clone()
        res = self.TD3(res)
        res = self.denseBlock4(res)
        skip4 = res.clone()
        res = self.TD4(res)
        res = self.denseBlockmid(res)

        res = self.UP1(res)
        skip4 = skip4
        res = torch.cat([res, skip4], dim=1)
        res = self.denseBlock4_right(res)
        res = self.UP2(res)
        skip3 = skip3
        res = torch.cat([res, skip3], dim=1)
        res = self.denseBlock3_right(res)
        res = self.UP3(res)
        skip2 = skip2
        res = torch.cat([res, skip2], dim=1)
        res = self.denseBlock2_right(res)
        res = self.UP4(res)
        skip1 = skip1
        res = torch.cat([res, skip1], dim=1)
        res = self.denseBlock1_right(res)
        return res




class ResUNet(nn.Module):

    def __init__(self, k=16, outputsize=2, inputsize=4, option = False):
        super(ResUNet, self).__init__()
        self.outputsize = outputsize
        self.option = option
        self.inputsize = inputsize
        self. k = k
        self.body = ResUNetBody(k=k, outputsize=outputsize, inputsize=inputsize, option =self.option)
        self.FC = ConvBlock(k*2, k*2, 1, padding=False)
        self.classifier = nn.Conv3d(k*2, self.outputsize, 1, padding=0)
    def forward(self, x):
        res = self.body(x)
        res = self.FC(res)
        res = self.classifier(res)
        return res


class DenseBlock(nn.Module):

    def __init__(self, k=10, n=4, inputsize=32):
        super(DenseBlock, self).__init__()
        self.k = k
        self.n = n
        self.inputsize = inputsize
        self.convolutions = nn.ModuleList([nn.Conv3d(inputsize, inputsize, 3, padding=1) for _ in range(0, self.n)])
        self.groupNorm = nn.ModuleList([nn.GroupNorm(inputsize, inputsize) for _ in range(0, self.n)])

    def forward(self, x):
        res = x
        for i in range(0, self.n):
            res = self.convolutions[i](res)
            res = self.groupNorm[i](res)
            res = F.leaky_relu(res)
        res.add(x)
        return res

    def getOutputImageSize(self, inputsize):
        outputsize = [i - (self.n * 2) for i in inputsize]
        return outputsize


class ConvBlock(nn.Module):

    def __init__(self, channels_in, channels_out, kernel_size, dropout=False, batchnorm=True, instancenorm=True,
                 padding=True):
        super(ConvBlock, self).__init__()
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.instancenorm = instancenorm
        if batchnorm:
            self.batchnorm_layer = nn.BatchNorm3d(channels_out)
        if padding:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv3d(channels_in, channels_out, kernel_size, padding=padding)
        if dropout:
            self.dropout_layer = nn.Dropout3d(p=0.2)
        if instancenorm:
            self.instance_layer = nn.InstanceNorm3d(channels_in)

    def forward(self, x):
        if self.dropout:
            x = self.dropout_layer(x)
        x = self.conv(x)
        if self.batchnorm:
            x = self.batchnorm_layer(x)
        if self.instancenorm:
            x = self.instance_layer(x)
        x = F.leaky_relu(x)
        return x


class TD(nn.Module):

    def __init__(self, inputsize=32, outputsize=32):
        super(TD, self).__init__()
        self.inputsize = inputsize
        self.outputsize = outputsize
        self.convolution = nn.Conv3d(self.inputsize, self.outputsize, 3, stride=2, padding=1)

    def forward(self, x):
        res = self.convolution(x)
        return res

    def getOutputImageSize(self, inputsize):
        outputsize = [i // 2 for i in inputsize]
        return outputsize

    def getOutputChannelSize(self):
        return self.k


class Discriminator(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super(Discriminator, self).__init__()
        # conv1 down
        self.conv1 = nn.Conv3d( in_channels=input_size,
                                out_channels=32,
                                kernel_size=3,
                                padding=1 )
        # max-pool 1
        self.pool1 = nn.Conv3d( in_channels=32,
                                out_channels=32,
                                kernel_size=2,
                                stride=2 )
        # conv2 down
        self.conv2 = nn.Conv3d( in_channels=32,
                                out_channels=64,
                                kernel_size=3,
                                padding=1 )
        # max-pool 2
        self.pool2 = nn.Conv3d( in_channels=64,
                                out_channels=64,
                                kernel_size=2,
                                stride=2 )
        # conv3 down
        self.conv3 = nn.Conv3d( in_channels=64,
                                out_channels=128,
                                kernel_size=3,
                                padding=1 )
        # max-pool 3
        self.pool3 = nn.Conv3d( in_channels=128,
                                out_channels=128,
                                kernel_size=2,
                                stride=2 )
        # conv4 down (latent space)
        self.conv4 = nn.Conv3d( in_channels=128,
                                out_channels=256,
                                kernel_size=3,
                                padding=1 )
        # conv8 (classification)
        self.conv8 = nn.Conv3d( in_channels=256,
                                out_channels=output_size,
                                kernel_size=4 )

        self.softmax = nn.Sigmoid()

    #Add more layers
        # self.conv9 = nn.Conv3d( in_channels=output_size,
        #                         out_channels=output_size,
        #                         kernel_size=4 )

    def forward(self, x):
        # encoder
        x1 = F.relu( self.conv1( x ) )
        x1p = self.pool1( x1 )
        x2 = F.relu( self.conv2( x1p ) )
        x2p = self.pool2( x2 )
        x3 = F.relu( self.conv3( x2p ) )
        x3p = self.pool3( x3 )

        # latent space
        x4 = F.relu( self.conv4( x3p ) )
        # output layer (1 classes)
        out = self.conv8( x4 )
        out = self.softmax(out)
        return out

class DiscriminatorBody( nn.Module ):

    def __init__(self, k=32, outputsize=1, inputsize=1):
        super( DiscriminatorBody, self ).__init__()
        self.outputsize = outputsize
        self.inputsize = inputsize
        self.k = k

        self.conv = ConvBlock( inputsize, k, 3 )
        self.denseBlock1 = DenseBlock( n=2, inputsize=k )
        self.TD1 = TD( inputsize=k, outputsize=k * 2 )
        self.denseBlock2 = DenseBlock( n=2, inputsize=k * 2 )
        self.TD2 = TD( inputsize=k * 2, outputsize=k * 4 )
        self.denseBlock3 = DenseBlock( n=2, inputsize=k * 4 )
        self.TD3 = TD( inputsize=k * 4, outputsize=k * 6 )
        self.denseBlock4 = DenseBlock( n=2, inputsize=k * 6 )
        self.TD4 = TD( inputsize=k * 6, outputsize=k * 8 )
        self.denseBlockmid = DenseBlock( n=2, inputsize=k * 8 )
        self.T_destination = TD(inputsize= k*8, outputsize=outputsize)
        self.finalblock = DenseBlock(n =2, inputsize = outputsize)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = self.conv( x )
        res = self.denseBlock1( res )
        skip1 = res.clone()
        res = self.TD1( res )
        res = self.denseBlock2( res )
        skip2 = res.clone()
        res = self.TD2( res )
        res = self.denseBlock3( res )
        skip3 = res.clone()
        res = self.TD3( res )
        res = self.denseBlock4( res )
        skip4 = res.clone()
        res = self.TD4( res )
        res = self.denseBlockmid( res )
        res = self.T_destination(res)
        res = self.finalblock(res)
        return res

class Discriminatore(nn.Module):
    def __init__(self, input_size = 1, output_size = 1):
        super(Discriminatore,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.val = 32
        self.main = nn.Sequential(
            nn.Conv3d(input_size, out_channels=self.val, kernel_size=2, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(in_channels=self.val, out_channels=self.val*2, kernel_size=2,stride=2, padding=1),
            # nn.BatchNorm3d(self.val*2),
            nn.LeakyReLU(0.2),
            nn.Conv3d(self.val*2, self.val*4, kernel_size=2 ,stride=2, padding=1),
            # nn.BatchNorm3d(self.val*4),
            nn.LeakyReLU(0.2),
            nn.Conv3d(self.val*4, self.val*6, kernel_size=2,stride=2, padding=1),
            # nn.BatchNorm3d( self.val * 6 ),
            nn.LeakyReLU( 0.2 ),
            nn.Conv3d(self.val*6, output_size, kernel_size=2),
            # nn.BatchNorm3d( self.val * 8 ),
            # nn.LeakyReLU( 0.2 ),
            # nn.Conv3d(self.val*8, output_size, kernel_size=3),
            nn.Sigmoid()
        )
    def forward(self, input) :
        return self.main(input)

#CODE FROM   https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L318
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 3
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias= use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
        self.softmax = nn.Sigmoid()
    #Check documentation if the sigmoid has to be inside or outside the sequential
    def forward(self, input):
        """Standard forward."""
        our = self.model(input)
        out = self.softmax(our)
        return out


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm3d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        # if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
        #     use_bias = norm_layer.func == nn.InstanceNorm2d
        # else:
        #     use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv3d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class AnotherDisc( nn.Module ):

    def __init__(self, inp, out):
        super( AnotherDisc, self ).__init__()
        x = 20
        self.net = nn.Sequential(
            nn.Conv3d( inp, x, 3 ),
            nn.ReLU( inplace=True ),
            nn.MaxPool3d(2),
            nn.Conv3d( x, x*2, 3 ),
            nn.ReLU( inplace=True ),
            nn.MaxPool3d( 2 ),
            nn.Conv3d( x*2, x*4, 3 ),
            nn.ReLU( inplace=True ),
            nn.Conv3d( x*4, x*4, 3 ),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(640,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,out),
            # nn.ReLU( inplace=True ),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.net( x )
        return out

class AnotherDisc2D( nn.Module ):

    def __init__(self, inp, out):
        super( AnotherDisc2D, self ).__init__()
        x = 20
        self.net = nn.Sequential(
            nn.Conv2d( inp, x, 3 ),
            nn.ReLU( inplace=True ),
            nn.MaxPool2d( 2 ),
            nn.Conv2d( x, x * 2, 3 ),
            nn.ReLU( inplace=True ),
            nn.MaxPool2d( 2 ),
            nn.Conv2d( x * 2, x * 4, 3 ),
            nn.ReLU( inplace=True ),
            nn.Conv2d( x * 4, x * 4, 3 ),
            nn.ReLU( inplace=True ),
            nn.Flatten(),
            nn.Linear( 640, 256 ),
            nn.ReLU( inplace=True ),
            nn.Linear( 256, out ),
            # nn.ReLU( inplace=True ),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.net( x )
        return out


class ResUNetBody2D(nn.Module):

    def __init__(self, k=16, outputsize=2, inputsize=4):
        super(ResUNetBody2D, self).__init__()
        self.outputsize = outputsize
        self.inputsize = inputsize
        self. k = k

        self.conv = ConvBlock2D(inputsize, k, 3)
        self.denseBlock1 = DenseBlock2D(n=2, inputsize=k)
        self.TD1 = TD(inputsize=k, outputsize=k*2)
        self.denseBlock2 = DenseBlock2D(n=2, inputsize=k*2)
        self.TD2 = TD(inputsize=k*2, outputsize=k*4)
        self.denseBlock3 = DenseBlock2D(n=2, inputsize=k*4)
        self.TD3 = TD(inputsize=k*4, outputsize=k * 4)
        self.denseBlock4 = DenseBlock2D(n=2, inputsize=k*4)
        self.TD4 = TD(inputsize=k*4, outputsize=k*4)
        self.denseBlockmid = DenseBlock2D(n=2, inputsize=k*4)
        self.UP1 = nn.ConvTranspose2d(k*4, k*4, 2, stride=2)
        self.denseBlock4_right = DenseBlock2D(n=2, inputsize=k*8)
        self.UP2 = nn.ConvTranspose2d(k*8, k*4, 2, stride=2)
        self.denseBlock3_right = DenseBlock2D(n=2, inputsize=k*8)
        self.UP3 = nn.ConvTranspose2d(k*8, k*2, 2, stride=2)
        self.denseBlock2_right = DenseBlock2D(n=2, inputsize=k*4)
        self.UP4 = nn.ConvTranspose2d(k*4, k*1, 2, stride=2)
        self.denseBlock1_right = DenseBlock2D(n=2, inputsize=k*2)

    def forward(self, x):
        res = self.conv(x)
        res = self.denseBlock1(res)
        skip1 = res.clone()
        res = self.TD1(res)
        res = self.denseBlock2(res)
        skip2 = res.clone()
        res = self.TD2(res)
        res = self.denseBlock3(res)
        skip3 = res.clone()
        res = self.TD3(res)
        res = self.denseBlock4(res)
        skip4 = res.clone()
        res = self.TD4(res)
        res = self.denseBlockmid(res)
        res = self.UP1(res)
        skip4 = skip4
        res = torch.cat([res, skip4], dim=1)
        res = self.denseBlock4_right(res)
        res = self.UP2(res)
        skip3 = skip3
        res = torch.cat([res, skip3], dim=1)
        res = self.denseBlock3_right(res)
        res = self.UP3(res)
        skip2 = skip2
        res = torch.cat([res, skip2], dim=1)
        res = self.denseBlock2_right(res)
        res = self.UP4(res)
        skip1 = skip1
        res = torch.cat([res, skip1], dim=1)
        res = self.denseBlock1_right(res)
        return res




class ResUNet2D(nn.Module):

    def __init__(self, k=16, outputsize=2, inputsize=4):
        super(ResUNet2D, self).__init__()
        self.outputsize = outputsize
        self.inputsize = inputsize
        self. k = k
        self.body = ResUNetBody2D(k=k, outputsize=outputsize, inputsize=inputsize)
        self.FC = ConvBlock2D(k*2, k*2, 1, padding=False)
        self.classifier = nn.Conv2d(k*2, self.outputsize, 1, padding=0)
        # self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        res = self.body(x)
        # print('After the Body: ',res.size())
        res = self.FC(res)
        # print('After FC: ', res.size())
        res = self.classifier(res)
        # print( 'After classifier: ', res.size() )
        # res = self.softmax(res)
        # print( 'After Softmax: ', res.size() )
        res = self.sigmoid(res)
        return res


class DenseBlock2D(nn.Module):

    def __init__(self, k=10, n=4, inputsize=32):
        super(DenseBlock2D, self).__init__()
        self.k = k
        self.n = n
        self.inputsize = inputsize
        self.convolutions = nn.ModuleList([nn.Conv2d(inputsize, inputsize, 3, padding=1) for _ in range(0, self.n)])
        self.groupNorm = nn.ModuleList([nn.GroupNorm(inputsize, inputsize) for _ in range(0, self.n)])

    def forward(self, x):
        res = x
        for i in range(0, self.n):
            res = self.convolutions[i](res)
            res = self.groupNorm[i](res)
            res = F.leaky_relu(res)
        res.add(x)
        return res

    def getOutputImageSize(self, inputsize):
        outputsize = [i - (self.n * 2) for i in inputsize]
        return outputsize


class ConvBlock2D(nn.Module):

    def __init__(self, channels_in, channels_out, kernel_size, dropout=False, batchnorm=True, instancenorm=True,
                 padding=True):
        super(ConvBlock2D, self).__init__()
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.instancenorm = instancenorm
        if batchnorm:
            self.batchnorm_layer = nn.BatchNorm2d(channels_out)
        if padding:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size, padding=padding)
        if dropout:
            self.dropout_layer = nn.Dropout2d(p=0.2)
        if instancenorm:
            self.instance_layer = nn.InstanceNorm2d(channels_in)

    def forward(self, x):
        if self.dropout:
            x = self.dropout_layer(x)
        x = self.conv(x)
        if self.batchnorm:
            x = self.batchnorm_layer(x)
        if self.instancenorm:
            x = self.instance_layer(x)
        x = F.leaky_relu(x)
        return x


class TD2D(nn.Module):

    def __init__(self, inputsize=32, outputsize=32):
        super(TD2D, self).__init__()
        self.inputsize = inputsize
        self.outputsize = outputsize
        self.convolution = nn.Conv2d(self.inputsize, self.outputsize, 3, stride=2, padding=1)

    def forward(self, x):
        res = self.convolution(x)
        return res

    def getOutputImageSize(self, inputsize):
        outputsize = [i // 2 for i in inputsize]
        return outputsize

    def getOutputChannelSize(self):
        return self.k


class Generator( nn.Module ):
    def __init__(self, g_input_dim, g_output_dim):
        super( Generator, self ).__init__()
        self.fc1 = nn.Linear( g_input_dim, 256 )
        self.fc2 = nn.Linear( self.fc1.out_features, self.fc1.out_features * 2 )
        self.fc3 = nn.Linear( self.fc2.out_features, self.fc2.out_features * 2 )
        self.fc4 = nn.Linear( self.fc3.out_features, g_output_dim )

    # forward method
    def forward(self, x):
        x = F.leaky_relu( self.fc1( x ), 0.2 )
        x = F.leaky_relu( self.fc2( x ), 0.2 )
        x = F.leaky_relu( self.fc3( x ), 0.2 )
        return torch.sigmoid( self.fc4( x ) )


class Discriminator( nn.Module ):
    def __init__(self, d_input_dim):
        super( Discriminator, self ).__init__()
        self.fc1 = nn.Linear( d_input_dim, 1024 )
        self.fc2 = nn.Linear( self.fc1.out_features, self.fc1.out_features // 2 )
        self.fc3 = nn.Linear( self.fc2.out_features, self.fc2.out_features // 2 )
        self.fc4 = nn.Linear( self.fc3.out_features, 1 )

    # forward method
    def forward(self, x):
        x = F.leaky_relu( self.fc1( x ), 0.2 )
        x = F.dropout( x, 0.3 )
        x = F.leaky_relu( self.fc2( x ), 0.2 )
        x = F.dropout( x, 0.3 )
        x = F.leaky_relu( self.fc3( x ), 0.2 )
        x = F.dropout( x, 0.3 )
        return torch.sigmoid( self.fc4( x ) )


class Generatori( nn.Module ):
    def __init__(self, g_input_dim, g_output_dim):
        super( Generatori, self ).__init__()
        self.conv1 = nn.Conv3d( in_channels=g_input_dim,out_channels=32, kernel_size=3, padding=1 )
        # max-pool 1
        self.pool1 = nn.Conv3d( in_channels=32, out_channels=32,kernel_size=2,stride=2 )
        # conv2 down
        self.conv2 = nn.Conv3d( in_channels=32,out_channels=64,kernel_size=3, padding=1 )
        # max-pool 2
        self.pool2 = nn.Conv3d( in_channels=64, out_channels=256, kernel_size=2, stride=2 )
        self.fc1 = nn.Linear( 256, 256 )
        self.fc2 = nn.Linear( self.fc1.out_features, self.fc1.out_features * 2 )
        self.fc3 = nn.Linear( self.fc2.out_features, self.fc2.out_features * 2 )
        self.fc4 = nn.Linear( self.fc3.out_features, g_output_dim )

    # forward method
    def forward(self, x):
        x = F.leaky_relu( self.conv1(x), 0.2)
        x = F.leaky_relu(self.pool1(x))
        x = F.leaky_relu( self.conv2(x), 0.2)
        x = F.leaky_relu(self.pool2(x))
        x = F.leaky_relu( self.fc1( x ), 0.2 )
        x = F.leaky_relu( self.fc2( x ), 0.2 )
        x = F.leaky_relu( self.fc3( x ), 0.2 )
        return torch.sigmoid( self.fc4( x ) )


class Discriminatori( nn.Module ):
    def __init__(self, d_input_dim):
        super( Discriminatori, self ).__init__()
        self.conv1 = nn.Conv3d( in_channels=d_input_dim,out_channels=32, kernel_size=3, padding=1 )
        # max-pool 1
        self.pool1 = nn.Conv3d( in_channels=32, out_channels=32,kernel_size=2,stride=2 )
        # conv2 down
        self.conv2 = nn.Conv3d( in_channels=32,out_channels=64,kernel_size=3, padding=1 )
        # max-pool 2
        self.pool2 = nn.Conv3d( in_channels=64, out_channels=1024, kernel_size=2, stride=2 )
        self.fc1 = nn.Linear( 1024, 1024 )
        self.fc2 = nn.Linear( self.fc1.out_features, self.fc1.out_features // 2 )
        self.fc3 = nn.Linear( self.fc2.out_features, self.fc2.out_features // 2 )
        self.fc4 = nn.Linear( self.fc3.out_features, 1 )

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x),0.2)
        x = F.leaky_relu(self.pool1(x), 0.2)
        x = F.leaky_relu(self.conv2(x),0.2)
        x = F.leaky_relu(self.pool2(x), 0.2)
        x = F.leaky_relu( self.fc1( x ), 0.2 )
        x = F.dropout( x, 0.3 )
        x = F.leaky_relu( self.fc2( x ), 0.2 )
        x = F.dropout( x, 0.3 )
        x = F.leaky_relu( self.fc3( x ), 0.2 )
        x = F.dropout( x, 0.3 )
        return torch.sigmoid( self.fc4( x ) )


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self,x):
        return self.main(x)

class DoubleConvi( nn.Module ):
    def __init__(self, in_channels, out_channels):
        super( DoubleConvi, self ).__init__()
        self.main = nn.Sequential(
            nn.Conv3d( in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1,
                       padding_mode='reflect' ),
            nn.BatchNorm3d( num_features=out_channels ),
            nn.LeakyReLU( inplace=True ),

        )

    def forward(self, x):
        return self.main( x )

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super( ResBlock, self ).__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(num_features=out_channels)
        )
        self.doubleconv = DoubleConv(in_channels=in_channels, out_channels=out_channels)
        self.downsample = nn.MaxPool3d(2)
        self.relu = nn.LeakyReLU()
    def forward(self, x):
        identity = self.down(x)
        out = self.doubleconv(x)
        out = self.relu(out+identity)
        return self.downsample(out), out



class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dconv= DoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self,down_input, skip_input):
        x = self.up(down_input)
        x =torch.cat([x, skip_input],dim =1)
        return self.dconv(x)

class ResUnette(nn.Module):
    """
    Hybrid solution of resnet blocks and double conv blocks
    """

    def __init__(self, in_classes =1,out_classes=1):
        super( ResUnette, self ).__init__()

        self.dc1 = ResBlocki(in_channels=in_classes, out_channels=64)
        self.dc2 = ResBlocki(in_channels=64, out_channels=128)
        self.dc3 = ResBlocki(in_channels=128, out_channels=256)
        self.dc4 = ResBlocki(in_channels=256, out_channels=512)

        self.dconv = DoubleConve(in_channels=512, out_channels=1024)

        self.up4 = UpBlocka(in_channels=512 + 1024, out_channels=512)
        self.up3 = UpBlocka(in_channels=512 + 256, out_channels=256)
        self.up2 = UpBlocka(in_channels=256 + 128, out_channels=128)
        self.up1 = UpBlocka(in_channels=128 + 64, out_channels=64)

        self.last = nn.Conv3d(in_channels=64, out_channels=out_classes, kernel_size=1)

    def forward(self, x ):
        x, skip1 = self.dc1(x)
        x, skip2 = self.dc2(x)
        x, skip3 = self.dc3(x)
        x, skip4 = self.dc4(x)
        x = self.dconv(x)
        x = self.up4(x, skip4)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        x = self.last(x)
        return x




class ResDisc(nn.Module):
    def __init__(self, in_classes =1,out_classes=1):
        super( ResDisc, self ).__init__()
        self.dc1 = ResBlock( in_channels=in_classes, out_channels=64 )
        self.dc2 = ResBlock( in_channels=64, out_channels=128 )
        self.dc3 = ResBlock( in_channels=128, out_channels=256 )
        self.dc4 = ResBlock( in_channels=256, out_channels=512 )
        self.dconv = DoubleConv( in_channels=512, out_channels=1024 )
        self.last = nn.Conv3d( in_channels=1024, out_channels=out_classes, kernel_size=1 )
    def forward(self, x):
        x, skip1 = self.dc1( x )
        x, skip2 = self.dc2( x )
        x, skip3 = self.dc3( x )
        x, skip4 = self.dc4( x )
        x = self.dconv( x )
        x = self.last( x )
        return x
class DoubleConve(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConve, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
class ResBlocki(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlocki, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(out_channels))
        self.double_conv = DoubleConve(in_channels, out_channels)
        self.down_sample = nn.MaxPool3d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.double_conv(x)
        out = self.relu(out + identity)
        return self.down_sample(out), out

class UpBlocka(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlocka, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.double_conv = DoubleConve(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)
class ONet(nn.Module):
    def __init__(self, alpha=470, beta=40, out_classes=1):
        super(ONet, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.down_conv1 = ResBlocki(1, 64)
        self.down_conv2 = ResBlocki(64, 128)
        self.down_conv3 = ResBlocki(128, 256)
        self.down_conv4 = ResBlocki(256, 512)

        self.double_conv = DoubleConve(512, 1024)

        self.up_conv4 = UpBlocka(512 + 1024, 512)
        self.up_conv3 = UpBlocka(256 + 512, 256)
        self.up_conv2 = UpBlocka(128 + 256, 128)
        self.up_conv1 = UpBlocka(128 + 64, 64)

        self.conv_last = nn.Conv3d(64, 1, kernel_size=1)
        self.input_output_conv = nn.Conv3d(2, 1, kernel_size=1)


    def forward(self, inputs):
        input_tensor, bounding = inputs
        x, skip1_out = self.down_conv1(input_tensor + (bounding * self.alpha))
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.conv_last(x)
        input_output = torch.cat([x, bounding * self.beta], dim=1)
        x = self.input_output_conv(input_output)
        return x