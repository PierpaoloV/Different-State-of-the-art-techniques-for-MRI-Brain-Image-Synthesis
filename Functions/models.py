import torch
from torch.nn import functional as F
import torch.nn as nn
import torch.nn.init as init
import numpy as np

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

'''
MODULES TAKEN FROM 
https://github.com/ginobilinie/medSynthesisV1/blob/b06a33a6e3069c418140f704dd1605b167e6c78a/ResUnet3d_pytorch.py#L66
'''
class Police(nn.Module):
    def __init__(self, dim, n_classes):
        super(Police,self).__init__()
        self.dim = dim
        self.n_classess = n_classes
        #you can make abbreviations for conv and fc, this is not necessary
        #class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv3d(in_channels=self.n_classess,out_channels=dim,kernel_size=7)
        self.bn1 = nn.BatchNorm3d(num_features=dim)
        self.conv2 = nn.Conv3d(in_channels=dim,out_channels=dim*2,kernel_size=5)
        self.bn2 = nn.BatchNorm3d(num_features=dim*2)
        self.conv3 = nn.Conv3d(in_channels=dim*2,out_channels=dim*2,kernel_size=3)
        self.bn3 = nn.BatchNorm3d(num_features=dim*2)
        self.conv3 = nn.Conv3d(in_channels=dim*2,out_channels=dim*4,kernel_size=3)
        self.bn3 = nn.BatchNorm3d(num_features=dim*4)
        self.fc1 = nn.Linear(in_features=256,out_features=512)
        self.fc2 = nn.Linear(in_features=512,out_features=dim*32)
        self.fc3 = nn.Linear(in_features=dim*32,out_features=dim*16)
        self.fc4 = nn.Linear(in_features=dim*16,out_features=dim*8)
        #self.bn3= nn.BatchNorm1d(6)
        self.fc5 = nn.Linear(in_features=dim*8,out_features=dim*4)
        self.fc6 = nn.Linear(in_features=dim*4,out_features=1)
        # self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
#         print 'line 114: x shape: ',x.size()
        #x = F.max_pool3d(F.relu(self.bn1(self.conv1(x))),(2,2,2))#conv->relu->pool
        x = F.max_pool3d(F.relu(self.conv1(x)),(2,2,2))#conv->relu->pool
        x = F.max_pool3d(F.relu(self.conv2(x)),(2,2,2))#conv->relu->pool
        x = F.max_pool3d(F.relu(self.conv3(x)),(2,2,2))#conv->relu->pool
        #reshape them into Vector, review ruturned tensor shares the same data but have different shape, same as reshape in matlab
        x = x.view(-1,self.num_of_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        # x = F.sigmoid(x)
        #print 'min,max,mean of x in 0st layer',x.min(),x.max(),x.mean()
        return x

    def num_of_flat_features(self, x):
        size = x.size()[1:]  # we don't consider the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        # print('Features: ',num_features)
        return num_features


class Disc(nn.Module):
    def __init__(self, dim, n_classes):
        super(Disc,self).__init__()
        self.dim = dim
        self.n_classess = n_classes
        self.main = nn.Sequential(
            nn.Conv3d(in_channels=self.n_classess,out_channels=dim,kernel_size=3,stride=2, padding =1, bias=False, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(in_channels=dim, out_channels=dim*2, kernel_size=3, stride =2, padding=1, bias = False, padding_mode='reflect'),
            nn.BatchNorm3d(dim*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(dim*2, dim*4, kernel_size=3, stride = 2, padding=1, bias =False ,padding_mode='reflect'),
            nn.BatchNorm3d(dim*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(in_channels=dim*4, out_channels=dim*8, kernel_size=3, stride=2, padding=1, bias=False,padding_mode='reflect'),
            nn.BatchNorm3d(dim*8),

        )
        self.main2 = nn.Sequential(
            nn.Linear(1024,1)
        )

    def forward(self,x, matching = False):
        output = self.main(x)
        feature = output.view(-1,1024)
        output = self.main2(feature)
        if matching == True:
            return feature, output
        else:
            return output

class Police2(nn.Module):
    def __init__(self, dim):
        super(Police2,self).__init__()
        self.dim = dim
        #you can make abbreviations for conv and fc, this is not necessary
        #class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv3d(in_channels=1,out_channels=dim,kernel_size=7)
        self.bn1 = nn.BatchNorm3d(num_features=dim)
        self.conv2 = nn.Conv3d(in_channels=dim,out_channels=dim*2,kernel_size=4)
        self.bn2 = nn.BatchNorm3d(num_features=dim*2)
        self.conv3 = nn.Conv3d(in_channels=dim*2,out_channels=dim*2,kernel_size=4)
        self.bn3 = nn.BatchNorm3d(num_features=dim*2)
        self.conv3 = nn.Conv3d(in_channels=dim*2,out_channels=dim*4,kernel_size=4)
        self.bn3 = nn.BatchNorm3d(num_features=dim*4)
        self.fc1 = nn.Linear(in_features=256,out_features=512)
        self.fc2 = nn.Linear(in_features=512,out_features=dim*32)
        self.fc3 = nn.Linear(in_features=dim*32,out_features=dim*16)
        self.fc4 = nn.Linear(in_features=dim*16,out_features=dim*8)
        #self.bn3= nn.BatchNorm1d(6)
        self.fc5 = nn.Linear(in_features=dim*8,out_features=dim*4)
        self.fc6 = nn.Linear(in_features=dim*4,out_features=1)
        # self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
#         print 'line 114: x shape: ',x.size()
        #x = F.max_pool3d(F.relu(self.bn1(self.conv1(x))),(2,2,2))#conv->relu->pool
        x = F.max_pool3d(F.relu(self.conv1(x)),(2,2,2))#conv->relu->pool
        x = F.max_pool3d(F.relu(self.conv2(x)),(2,2,2))#conv->relu->pool
        x = F.max_pool3d(F.relu(self.conv3(x)),(2,2,2))#conv->relu->pool
        #reshape them into Vector, review ruturned tensor shares the same data but have different shape, same as reshape in matlab
        x = x.view(-1,self.num_of_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        # x = F.sigmoid(x)
        #print 'min,max,mean of x in 0st layer',x.min(),x.max(),x.mean()
        return x

    def num_of_flat_features(self, x):
        size = x.size()[1:]  # we don't consider the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        # print('Features: ',num_features)
        return num_features

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_size, out_size, kernel_size, stride=1, padding=1)
        self.bn = nn.BatchNorm3d(out_size)
        self.conv2 = nn.Conv3d(out_size, out_size, kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_size)
        self.activation = activation


        nn.init.xavier_uniform_(self.conv.weight, gain = np.sqrt(2.0))
        init.constant_(self.conv.bias,0)
        nn.init.xavier_uniform_(self.conv2.weight, gain = np.sqrt(2.0))
        init.constant_(self.conv2.bias,0)
    def forward(self, x):
        out = self.activation(self.bn(self.conv(x)))
        out = self.activation(self.bn2(self.conv2(out)))

        return out


'''    
 two-layer residual unit: two conv with BN/relu and identity mapping
'''
class residualUnit(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3,stride=1, padding=1, activation=F.relu):
        super(residualUnit, self).__init__()
        self.conv1 = nn.Conv3d(in_size, out_size, kernel_size, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight, gain = np.sqrt(2.0)) #or gain=1
        init.constant_(self.conv1.bias, 0)
        self.conv2 = nn.Conv3d(out_size, out_size, kernel_size, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv2.weight, gain = np.sqrt(2.0)) #or gain=1
        init.constant_(self.conv2.bias, 0)
        self.activation = activation
        self.bn1 = nn.BatchNorm3d(out_size)
        self.bn2 = nn.BatchNorm3d(out_size)
        self.in_size = in_size
        self.out_size = out_size
        if in_size != out_size:
            self.convX = nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0)
            self.bnX = nn.BatchNorm3d(out_size)

    def forward(self, x):
        out1 = self.activation(self.bn1(self.conv1(x)))
        out2 = self.activation(self.bn1(self.conv2(out1)))
        if self.in_size!=self.out_size:
            bridge = self.activation(self.bnX(self.convX(x)))
        output = torch.add(out2, bridge)

        return output


'''
    Ordinary UNet-Up Conv Block
'''
class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_size, out_size, 2, stride=2)
        self.bnup = nn.BatchNorm3d(out_size)
        self.conv = nn.Conv3d(in_size, out_size, kernel_size, stride=1, padding=1)
        self.bn = nn.BatchNorm3d(out_size)
        self.conv2 = nn.Conv3d(out_size, out_size, kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_size)
        self.activation = activation
        # nn.init.xavier_uniform_(self.up.weight, gain = np.sqrt(2.0))
        nn.init.xavier_uniform_(self.up.weight, gain = np.sqrt(2.0))
        init.constant_(self.up.bias,0)
        # nn.init.xavier_uniform_(self.conv.weight, gain = np.sqrt(2.0))
        nn.init.xavier_uniform_(self.conv.weight, gain = np.sqrt(2.0))
        init.constant_(self.conv.bias,0)
        # nn.init.xavier_uniform_(self.conv2.weight, gain = np.sqrt(2.0))
        nn.init.xavier_uniform_(self.conv2.weight, gain = np.sqrt(2.0))
        init.constant_(self.conv2.bias,0)

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height, layer_depth = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):
        up = self.up(x)
        up = self.activation(self.bnup(up))
        crop1 = self.center_crop(bridge, up.size()[2])
        out = torch.cat([up, crop1], 1)

        out = self.activation(self.bn(self.conv(out)))
        out = self.activation(self.bn2(self.conv2(out)))

        return out



'''
    Ordinary Residual UNet-Up Conv Block
'''
class UNetUpResBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetUpResBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_size, out_size, 2, stride=2)
        self.bnup = nn.BatchNorm3d(out_size)

        nn.init.xavier_uniform_(self.up.weight, gain = np.sqrt(2.0))
        init.constant_(self.up.bias,0)

        self.activation = activation

        self.resUnit = residualUnit(in_size, out_size, kernel_size = kernel_size)

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height, layer_depth = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):
        #print 'x.shape: ',x.shape
        up = self.activation(self.bnup(self.up(x)))
        #crop1 = self.center_crop(bridge, up.size()[2])
        #print 'up.shape: ',up.shape, ' crop1.shape: ',crop1.shape
        crop1 = bridge
        out = torch.cat([up, crop1], 1)

        out = self.resUnit(out)
        # out = self.activation(self.bn2(self.conv2(out)))

        return out


'''
    Ordinary UNet
'''
class UNet_2(nn.Module):
    def __init__(self, in_channel = 1, n_classes = 4):
        super(UNet_2, self).__init__()
#         self.imsize = imsize

        self.activation = F.relu

        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)
        self.pool3 = nn.MaxPool3d(2)
        # self.pool4 = nn.MaxPool3d(2)


        self.conv_block1_64 = UNetConvBlock(in_channel, 32)
        self.conv_block64_128 = UNetConvBlock(32, 64)
        self.conv_block128_256 = UNetConvBlock(64, 128)
        self.conv_block256_512 = UNetConvBlock(128, 256)
        # self.conv_block512_1024 = UNetConvBlock(512, 1024)
        # this kind of symmetric design is awesome, it automatically solves the number of channels during upsamping
        # self.up_block1024_512 = UNetUpBlock(1024, 512)
        self.up_block512_256 = UNetUpBlock(256, 128)
        self.up_block256_128 = UNetUpBlock(128, 64)
        self.up_block128_64 = UNetUpBlock(64, 32)

        self.last = nn.Conv3d(32, n_classes, 1, stride=1)


    def forward(self, x):
#         print 'line 70 ',x.size()
        block1 = self.conv_block1_64(x)
        pool1 = self.pool1(block1)

        block2 = self.conv_block64_128(pool1)
        pool2 = self.pool2(block2)

        block3 = self.conv_block128_256(pool2)
        pool3 = self.pool3(block3)

        block4 = self.conv_block256_512(pool3)
        # pool4 = self.pool4(block4)
        #
        # block5 = self.conv_block512_1024(pool4)
        #
        # up1 = self.up_block1024_512(block5, block4)

        up2 = self.up_block512_256(block4, block3)

        up3 = self.up_block256_128(up2, block2)

        up4 = self.up_block128_64(up3, block1)

        return self.last(up4)


'''
    Ordinary ResUNet
'''


class ResUNet_2(nn.Module):
    def __init__(self, in_channel=2, n_classes=4):
        super(ResUNet_2, self).__init__()
        #         self.imsize = imsize

        self.activation = F.relu

        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)
        self.pool3 = nn.MaxPool3d(2)
        # self.pool4 = nn.MaxPool3d(2)

        self.conv_block1_64 = UNetConvBlock(in_channel, 32)
        self.conv_block64_128 = residualUnit(32, 64)
        self.conv_block128_256 = residualUnit(64, 128)
        self.conv_block256_512 = residualUnit(128, 256)
        # self.conv_block512_1024 = residualUnit(512, 1024)
        # this kind of symmetric design is awesome, it automatically solves the number of channels during upsamping
        # self.up_block1024_512 = UNetUpResBlock(1024, 512)
        self.up_block512_256 = UNetUpResBlock(256, 128)
        self.up_block256_128 = UNetUpResBlock(128, 64)
        self.up_block128_64 = UNetUpResBlock(64, 32)

        self.last = nn.Conv3d(32, n_classes, 1, stride=1)
        #Print parameters


    def forward(self, x):
        #         print 'line 70 ',x.size()
        block1 = self.conv_block1_64(x)
        pool1 = self.pool1(block1)

        block2 = self.conv_block64_128(pool1)
        pool2 = self.pool2(block2)

        block3 = self.conv_block128_256(pool2)
        pool3 = self.pool3(block3)

        block4 = self.conv_block256_512(pool3)
        # pool4 = self.pool4(block4)
        #
        # block5 = self.conv_block512_1024(pool4)
        #
        # up1 = self.up_block1024_512(block5, block4)

        up2 = self.up_block512_256(block4, block3)

        up3 = self.up_block256_128(up2, block2)

        up4 = self.up_block128_64(up3, block1)

        return self.last(up4)




