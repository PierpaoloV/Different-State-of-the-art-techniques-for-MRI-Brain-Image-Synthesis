import torch
import synthunet.Functions.models  as mod
import synthunet.Functions.models_new  as models
import niclib as nl
import numpy as np
import functools
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import synthunet.Functions.models_2d as mode

def build_Discriminator():
    """
    Simple function (expandible) which returns the type of Optimizer for the discriminator and its parameters
    ADAM: lr
    SGD: lr, momentum
    added possibility to use or not soft labels whule training D

    :returns D_optim, D_optimizer_parameters, labels
    """
    D_optim = None
    D_lr = None
    labels = None
    discriminator =None
    n_classes = None
    k = [1,2]
    D_optimizer_parameters = None
    while n_classes is None:
        n_classes = int(input('Number of input classes for the Discriminator: \n'))
        if n_classes not in k:
            n_classes = None
            print('Value not valid. Choose between 1 and 2.\n')
    # model_D = models.Police( dim=64,n_classes=n_classes)
    # model_D = models.Police2( dim=64 )
    # model_D = models.Disc(dim= 32, n_classes =1)
    #
    # num_params_D = nl.net.get_num_trainable_parameters( model_D )
    while discriminator is None:
        dtype = input('Select between these two discriminators: \n'
                      'Disc        [1]\n'
                      'ResDisc     [2] \n'
                      'Police     [3] \n'
                      'TryDis     [4] \n'
                      'NLayer2d     [5] \n'
                      'Pixel2d     [6] \n')
        if dtype == 'Disc' or dtype == '1':
            model_D = models.Disc(dim=32, n_classes=n_classes)
            num_params_D = nl.net.get_num_trainable_parameters( model_D )
            discriminator = 1
        elif dtype =='ResDisc' or dtype == '2':
            model_D = mod.ResDisc(in_classes=1, out_classes=n_classes)
            num_params_D = nl.net.get_num_trainable_parameters( model_D )
            discriminator = 2
        elif dtype == 'TryDis' or dtype == '4':
            model_D = mod.TryDis(1)
            num_params_D = nl.net.get_num_trainable_parameters( model_D )
            discriminator = 4
        elif dtype =='NLayer2d' or dtype =='5':
            model_D =mode.NLayerDiscriminator(input_nc=1, ndf=64,n_layers=3,norm_layer=functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True))
            num_params_D = nl.net.get_num_trainable_parameters(model_D)
            discriminator =4
        elif dtype =='Pixel2d' or dtype =='6':
            model_D =mode.PixelDiscriminator(input_nc=1,ndf=64,norm_layer=functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True))
            num_params_D = nl.net.get_num_trainable_parameters(model_D)
            discriminator =5
        elif dtype =='Police' or dtype =='3':
            model_D = models.Disc(dim= 64, n_classes =1)
            num_params_D = nl.net.get_num_trainable_parameters(model_D)
            discriminator =3
        else:
            discriminator = None
            print('Please repeat')
    while D_optim is None:
        optim = input( 'Specify the optimizer for the Discriminator.\n'
                       'Adam        [2]\n'
                       'SGD         [3]\n' )
        if optim == 'Adam' or (int(optim) == 2):
            D_optim = torch.optim.Adam
        elif optim == 'SGD' or (int(optim) == 3):
            D_optim = torch.optim.SGD
        else:
            D_optim = None
            print( 'Please insert a valid Optimizer' )

    while D_lr is None:
        if optim == 'Adam' or int(optim) == 2:
            D_lr = float( input( 'Specify the Learning Rate for the Discriminator Using Adam ( 1 = 1e-3)\n' ) )
            D_lr = D_lr / 1000.0
            D_optimizer_parameters = {'lr': D_lr}
            if D_lr <= 0:
                D_lr = None
                print( 'Value not correct.' )
        else:
            momentum = None
            D_lr = float( input( 'Specify the Learning Rate for the Discriminator Using SGD ( 1 = 1)\n' ) )
            D_lr = D_lr

            if D_lr <= 0:
                D_lr = None
                print( 'Value not correct.' )
            momentum = float(input('Specify the momentum you want. (typical 0.5 -0.9) \n '
                                   'Default value 0.5: '))
            D_optimizer_parameters = {'lr': D_lr, 'momentum': momentum}
            if not (0 < momentum < 1 ) :
                momentum = None
                print('Value not correct.')

    while labels is None:
        labels = bool( int( input( 'Do you want to use Soft Labels or Hard Labels? \n'
                                  'Hard Labels = 1 \n'
                                  'Soft Labels = 0 \n' ) ) )
        if labels not in [0 , 1]:
            labels = None
            print('Invalid Option')

    return model_D, D_optim, D_optimizer_parameters, num_params_D,labels

def build_Generator():
    """
    Simple function which builds different Translators and allows to specify the learning rate for
    the optimizer.

    :return:  modeltype, G_optimizer_parameters
    """
    modeltype =None
    T_lr = None
    n_classes = None
    o_classes = None
    k = [1,2]
    while n_classes is None:
        n_classes = int(input('Number of Input classes for the Translator: \n'))
        if n_classes not in k:
            n_classes = None
            print('Value not valid. Choose between 1 and 2.\n')
    while o_classes is None:
        o_classes = int( input( 'Number of Output classes for the Translator: \n' ) )
        if o_classes not in k:
            o_classes = None
            print( 'Value not valid. Choose between 1 and 2.\n' )
    while modeltype is None:
        modeltype = input( 'Specify the type of the model to use.   \n'
                           '\u001b[41m Resunet_1\u001b[0m       [1]\n'
                           '\u001b[41m Unet_Upsampling\u001b[0m       [2] \n'
                           '\u001b[42m Unet_1\u001b[0m      [3] \n'
                           '\u001b[42m Unet_2 \u001b[0m     [4] \n'
                           '\u001b[42m Resunette \u001b[0m     [5] \n'
                           '\u001b[42m Unet2D \u001b[0m     [6] \n'
                           '\u001b[42m ResUnet2D \u001b[0m     [7] \n'
                           '\u001b[42m Onet \u001b[0m     [8] \n')
        if modeltype == 'Resunet_1' or (int(modeltype) ==1):
            model_T = mod.ResUNet( inputsize=n_classes, outputsize=o_classes, k=16 )
            num_params_T = nl.net.get_num_trainable_parameters( model_T )
            modeltype = 'Resunet_1'
            # print( "This network has {} parameters".format( num_params ) )
        elif modeltype == 'Unet_1' or (int(modeltype) ==3):
            model_T = mod.Unet( input_size=n_classes, output_size=o_classes )
            # model_T = mod.Unet_1(input_size=n_classes, output_size=o_classes)
            num_params_T = nl.net.get_num_trainable_parameters( model_T )
            modeltype = 'Unet_1'
            # print( "This network has {} parameters".format( num_params ) )
        elif modeltype == 'Unet_2' or (int(modeltype) ==4):
            model_T = models.UNet( in_channel=n_classes, n_classes=o_classes )
            num_params_T = nl.net.get_num_trainable_parameters( model_T )
            modeltype = 'Unet_2'
            # print( "This network has {} parameters".format( num_params ) )
        elif modeltype == 'Unet_Upsampling'or (int(modeltype) ==2):
            # model_T = models.ResUNet( in_channel=n_classes, n_classes=o_classes )
            model_T = mod.Unet_Upsampled(input_size=n_classes, output_size=o_classes)
            num_params_T = nl.net.get_num_trainable_parameters( model_T )
            modeltype = 'Unet_upsampled'
        elif modeltype == 'ResUnet2D'or (int(modeltype) ==7):

            model_T =mode.ResnetGenerator(input_nc=n_classes,output_nc=1,ngf=64, norm_layer=functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True), use_dropout=True, num_blocks=9)
            num_params_T = nl.net.get_num_trainable_parameters( model_T )
            modeltype = 'ResUnet_2D'
            # print( "This network has {} parameters".format( num_params ) )
        elif modeltype == 'Resunette' or (int(modeltype)==5):
            pretra = input('Load a Pretrained version of it?')
            if pretra =='y' or '1':
                ver = int(input('T1->FLAIR     [1]\n'
                                'FLAIR->T1     [2]\n'))
                if ver ==1:
                    print('Loading First Translator')
                    model_T = torch.load(
                        '/home/pierpaolo/home/pierpaolo/01_07_2020_WMH20173D/checkpoints/1_07_20_pretrain_20ep_0f__net.pt')
                else:
                    print('Loadinf Second Translator')
                    model_T = torch.load(
                        '/home/pierpaolo/home/pierpaolo/02_07_2020_WMH20173D/checkpoints/2_07_20_pretrain_20ep_0f__net.pt' )
            else:
                print('Initializing from zero')
                model_T = mod.ResUnette(in_classes=n_classes, out_classes=o_classes)
            num_params_T = nl.net.get_num_trainable_parameters(model_T)
            modeltype = 'Resunette'
        elif modeltype =='Onet' or '8':
            model_T = mod.ONet(alpha=470, beta=40, out_classes=1)
            num_params_T = nl.net.get_num_trainable_parameters( model_T )
            modeltype ='Onet'
        elif modeltype =='Unet2D' or int(modeltype)==6:
            modeltype ='Unet_2D'
            model_T= mode.UnetGenerator( input_nc=n_classes, output_nc=o_classes, num_downs=7, ngf=64, norm_layer=functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True), use_dropout=True )
            num_params_T =nl.net.get_num_trainable_parameters(model_T)
        else:
            modeltype = None
            print( 'Please enter a valid model' )

    while T_lr is None:
        T_lr = float( input( 'Specify the Learning Rate for the Translator. ( 1 = 1e-3)\n' ) )
        T_lr = T_lr / 1000.0
        G_optimizer_parameters = {'lr': T_lr}
        if T_lr < 0:
            T_lr = None
            print( 'Value not correct.' )

    return modeltype,model_T,num_params_T, G_optimizer_parameters
def get_dataset():
    dset = None
    while dset == None:
        dset = int( input( 'Which dataset ? \n'
                           '0 = WMH\n'
                           '1 = Brats2018\n' ) )
        if dset not in [0, 1]:
            dset = None
            print( 'repeat' )
    return dset

def build_Backbone(cycle = False):
    """
    Simple function which builds the backbone of the program with essential parameters
    :returns: b_size, epoch_number, foldstoevaluate, samplesnumber, saveimage
    """

    samplesnumber = None
    b_size = None
    epoch_number = None
    saveimage = None
    foldstoevaluate = None
    intermediate_img = None
    lam = None
    flag = None
    while lam is None:
        lam = float(input('Insert a Lambda Factor for Cycle Consistency between 0 and 100\n'))
        if (lam < 0.0) or (lam > 100.0) :
            lam = None
            print('Repeat, value not valid. \n')
    while flag is None:

        flag = int(input('Please specify the number of channels for input image \n'
                         '1 , 2    '))
        if flag ==2:
            flag = 1
        elif flag == 1:
            flag = 0
        else:
            flag = None
            print('Please Insert a valid number.\n')
    while foldstoevaluate is None:
        foldstoevaluate = int( input( 'Which division of dataset you want to train on? \n'
                                      '0 = only fold 0 \n'
                                      '1 = only fold 1 \n'
                                      '2 = only fold 2 \n'
                                      '3 = only fold 3 \n'
                                      '-1 = All Folds \n'
                                      '-2 = only folds 1 , 2 , 3 \n'
                                      '-3 = only folds 2 , 3 \n' ) )
        if foldstoevaluate == 0:
            folds = [0]
        elif foldstoevaluate == 1:
            folds = [1]
        elif foldstoevaluate == 1:
            folds = [2]
        elif foldstoevaluate == 1:
            folds = [3]
        elif foldstoevaluate == -1:
            folds = [0, 1, 2, 3]
        elif foldstoevaluate == -2:
            folds = [1, 2, 3]
        elif foldstoevaluate == -3:
            folds = [2, 3]
        else:
            foldstoevaluate = None
            print( 'Please insert a valid condition.\n' )

    while epoch_number is None:
        epoch_number = int( input( 'Specify number of epochs. ' ) )
        if epoch_number <= 0:
            print( 'the number {} is not valid.\n'.format( epoch_number ) )
            epoch_number = None

    while b_size is None:
        possibilities = [1, 8, 16, 24 , 32, 64, 128]
        b_size = int( input( 'Specify batch size. \n'
                             'Allowed batch sizes : 1, 8, 16, 24, 32, 64, 128 \n' ) )
        if b_size not in possibilities:
            print( 'the number {} is not valid.\n'.format( b_size ) )
            b_size = None

    while samplesnumber is None:
        samplesnumber = int( input( 'Specify number of times you want the volume. ' ) )
        samplesnumber = samplesnumber * 384
        if samplesnumber <= 0:
            print( 'the number {} is not valid.\n'.format( samplesnumber ) )
            samplesnumber = None

    while saveimage is None:
        saveimage = bool( int( input( 'Do you want to save the images after training? \n'
                                  'Yes = 1 \n'
                                  'No = 0 \n' ) ) )
        if saveimage not in [0 , 1]:
            saveimage = None
            print('Invalid Option')
    if cycle == True:
        while intermediate_img is None:
            intermediate_img = bool( int( input( 'Do you want to save the images after training? \n'
                                      'Yes = 1 \n'
                                      'No = 0 \n' ) ) )
            if intermediate_img not in [0 , 1]:
                intermediate_imgimage = None
                print('Invalid Option')

        return b_size,flag, epoch_number, folds, samplesnumber, saveimage, intermediate_img, lam
    else:
        return b_size,flag, epoch_number, folds, samplesnumber, saveimage, lam




class UniformSampling(nl.generator.PatchSampling):
    """Uniform regular sampling of patch centers.

    :param tuple step: extraction step as tuple (step_x, step_y, step_z)
    :param int num_patches: (optional, default: None) By default all sampled centers are returned.
        If num_patches is given, the centers are regularly resampled to the given number.
    :param List[Any] masks: (optional) List of masks defining the area where sampling will be performed.
        To exclude a voxel, the value of ``mask[x,y,z]`` must evaluate to 0 or False.
        Must contain same number of masks as ``images`` when sampling with the same (x, y, z) dimensions.
    """

    def __init__(self, step, num_patches=None, masks=None):
        assert len(step) == 3, 'len({}) != 3'.format(step)
        if num_patches is not None:
            assert num_patches > 0, 'number of patches must be greater than 0'
        self.step = step
        self.npatches = num_patches
        self.masks = masks

    def sample_centers(self, images, patch_shape):
        assert len(patch_shape) == len(self.step) == 3, 'len({}) != 3 or len({}) != 3'.format(patch_shape, self.step)
        assert all(len(img.shape) == 4 for img in images)
        # if self.masks is not None:
            # assert all(img[0].shape == msk[0].shape for img, msk in zip(images, self.masks))

        self.masks = [None] * len(images) if self.masks is None else self.masks
        patches_per_image = int(np.ceil(self.npatches / len(images))) if self.npatches is not None else None
        return [nl.generator.patch.sample_centers_uniform(img[0], patch_shape, self.step, patches_per_image, img_mask)
                for img, img_mask in zip(images, self.masks)]