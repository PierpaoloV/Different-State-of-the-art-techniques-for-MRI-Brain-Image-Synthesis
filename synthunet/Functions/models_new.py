import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
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
class UNet(nn.Module):
    def __init__(self, in_channel = 1, n_classes = 4):
        super(UNet, self).__init__()
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


class ResUNet(nn.Module):
    def __init__(self, in_channel=2, n_classes=4):
        super(ResUNet, self).__init__()
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


'''
    UNet (lateral connection) with long-skip residual connection (from 1st to last layer)
'''
class UNet_LRes(nn.Module):
    def __init__(self, in_channel = 1, n_classes = 4):
        super(UNet_LRes, self).__init__()
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


    def forward(self, x, res_x):
#         print 'line 70 ',x.size()
        block1 = self.conv_block1_64(x)
        pool1 = self.pool1(block1)

        block2 = self.conv_block64_128(pool1)
        pool2 = self.pool2(block2)

        block3 = self.conv_block128_256(pool2)
        pool3 = self.pool3(block3)

        block4 = self.conv_block256_512(pool3)
        # pool4 = self.pool4(block4)

        # block5 = self.conv_block512_1024(pool4)
        #
        # up1 = self.up_block1024_512(block5, block4)

        up2 = self.up_block512_256(block4, block3)

        up3 = self.up_block256_128(up2, block2)

        up4 = self.up_block128_64(up3, block1)

        last = self.last(up4)
        #print 'res_x.shape is ',res_x.shape,' and last.shape is ',last.shape
        if len(res_x.shape)==3:
            res_x = res_x.unsqueeze(1)
        out = torch.add(last,res_x)

        #print 'out.shape is ',out.shape
        return out


'''
    ResUNet (lateral connection) with long-skip residual connection (from 1st to last layer)
'''


class ResUNet_LRes(nn.Module):
    def __init__(self, in_channel=1, n_classes=4, dp_prob=0):
        super(ResUNet_LRes, self).__init__()
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
        self.Dropout = nn.Dropout3d(p=dp_prob)
        self.last = nn.Conv3d(32, n_classes, 1, stride=1)

    def forward(self, x, res_x):
        #         print 'line 70 ',x.size()
        block1 = self.conv_block1_64(x)
        # print 'block1.shape: ', block1.shape
        pool1 = self.pool1(block1)
        # print 'pool1.shape: ', block1.shape
        pool1_dp = self.Dropout(pool1)
        # print 'pool1_dp.shape: ', pool1_dp.shape
        block2 = self.conv_block64_128(pool1_dp)
        pool2 = self.pool2(block2)

        pool2_dp = self.Dropout(pool2)

        block3 = self.conv_block128_256(pool2_dp)
        pool3 = self.pool3(block3)

        pool3_dp = self.Dropout(pool3)

        block4 = self.conv_block256_512(pool3_dp)
        # pool4 = self.pool4(block4)
        #
        # pool4_dp = self.Dropout(pool4)
        #
        # # block5 = self.conv_block512_1024(pool4_dp)
        #
        # up1 = self.up_block1024_512(block5, block4)

        up2 = self.up_block512_256(block4, block3)

        up3 = self.up_block256_128(up2, block2)

        up4 = self.up_block128_64(up3, block1)

        last = self.last(up4)
        # print 'res_x.shape is ',res_x.shape,' and last.shape is ',last.shape
        if len(res_x.shape) == 3:
            res_x = res_x.unsqueeze(1)
        out = torch.add(last, res_x)

        # print 'out.shape is ',out.shape
        return out

