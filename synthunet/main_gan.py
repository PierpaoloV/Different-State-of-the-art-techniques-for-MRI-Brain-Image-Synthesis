import os

import torch
from torch.nn import functional as F
from synthunet.Functions.test import evaluate_model
import synthunet.Functions.utils as utils
import niclib as nl
import synthunet.Functions.models  as mod
import synthunet.Functions.models_new  as models
import torch.nn as nn
import time
import synthunet.Functions.preutils as pre
from synthunet.Functions.dataprep import preparedata, load_case_brats,get_datax, get_patch_gen, splitdata
#------------------------------------------------------------------------------
print('Initializing Options...')
modeltype,model_T, num_params_T, T_optimizer_parameters = pre.build_Generator()
model_D, D_optim, D_optimizer_parameters, labels, num_params_D = pre.build_Discriminator()
b_size, flag, epoch_number, folds, samplesnumber, saveimage, lam = pre.build_Backbone()


data = 'GAN_' + modeltype
data_path, checkpoints_path, results_path, metrics_path, log_path, mainfolder = preparedata(1, 'brats_gan_resunet')
summary = log_path + 'summary.txt'
f = open(summary, 'a')
f.write('Recap: \n'
      'Discriminator Number of Parameters:'+str(num_params_D/1000000)+' Milions \n'
      'Discriminator Optimizer:'+ str(D_optim) +'\n'
      'Discriminator Optimizer parameters:' +repr(D_optimizer_parameters)+'\n'
      'Hard - Soft Labels:'+ str(labels)+'\n'
      '\n'
      'Translator Selected:'+ modeltype+' \n'
      'Translator Number of Parameters: '+ str(num_params_T/1000000)+' Milions \n'
      'Translator Optimizer Parameters:' +repr(T_optimizer_parameters)+' \n'
      '\n'
      'Fold(s) to analyze: '+str(folds)+' \n'
      'Number of Epochs to train: '+str(epoch_number)+' \n'
      'Batch size: '+str(b_size)+'\n'
      'Number of Samples per Image: '+str(samplesnumber)+' \n'
      )
f.close()


print('-------------   RECAP   ------------- \n'
      'Discriminator Number of Parameters: \u001b[1m{}\u001b[0m Milions \n'
      'Discriminator Optimizer: \u001b[1m{}\u001b[0m \n'
      'Discriminator Optimizer parameters: \u001b[1m{}\u001b[0m \n'
      'Hard - Soft Labels: \u001b[1m{}\u001b[0m \n'
      '\n'
      'Translator Selected: \u001b[1m{}\u001b[0m \n'
      'Translator Number of Parameters: \u001b[1m{}\u001b[0m Milions \n'
      'Translator Optimizer Parameters:\u001b[1m{}\u001b[0m  \n'
      '\n'
      'Fold(s) to analyze: \u001b[1m{}\u001b[0m \n'
      'Number of Epochs to train: \u001b[1m{}\u001b[0m \n'
      'Batch size: \u001b[1m{}\u001b[0m \n'
      'Number of Samples per Image: \u001b[1m{}\u001b[0m \n'
      'Save - Discard synthetized images: \u001b[1m{}\u001b[0m \n'
      
      'Folder Create: \u001b[1m{}\u001b[0m \n'

      '------------- END RECAP -------------'.format(num_params_D/1000000,D_optim,D_optimizer_parameters,labels,modeltype, num_params_T/1000000, T_optimizer_parameters,folds,epoch_number,b_size, samplesnumber,saveimage, mainfolder,))
#--------------------------------------------------------
data = 'GAN_' + modeltype
case_path = [f.path for f in os.scandir(data_path) if f.is_dir()]

tot = time.time()
for k in folds:
    start = time.time()
    print('Getting the fold \u001b[1m{}\u001b[0m'.format(k))
    # case_paths,test_case_paths =get_datax(case_path.copy(),k)
    case_paths, test_case_paths = splitdata( case_path.copy(), k )
    print("Loading training dataset with \u001b[1m{}\u001b[0m images in fold \u001b[1m{}\u001b[0m...".format(len(case_paths),k))
    a = time.time()
    dataset = nl.parallel_load(load_func=load_case_brats, arguments=case_paths, num_workers=12)
    b = time.time()
    dataset_train, dataset_val = nl.split_list(dataset, fraction=0.8)  # Split images into train and validation
    print('Training dataset with \u001b[1m{}\u001b[0m train and \u001b[1m{}\u001b[0m val images'.format(len(dataset_train), len(dataset_val)))
    print('needed \u001b[1m{:.2f} seconds\u001b[0m to load and split the images'.format(b-a))

    step = 1
    samples = samplesnumber
    c = time.time()
    train_patch_set = get_patch_gen(dataset_train, step, samples, normalize_opt= 'none')
    val_patch_set = get_patch_gen(dataset_val, step, samples, normalize_opt= 'none')

    train_gen = nl.generator.make_generator(set=train_patch_set, batch_size=b_size, shuffle=True)
    val_gen = nl.generator.make_generator(set=val_patch_set, batch_size=b_size, shuffle=True)
    d = time.time()
    print("Train and val patch generators with {} and {} patches".format(len(train_patch_set), len(val_patch_set)))
    print('Needed {:.2f} minutes to generate {} patches. '.format((d-c)/60, samplesnumber))
    loss_to_use = nl.loss.LossWrapper( torch.nn.BCEWithLogitsLoss(),
                                       preprocess_fn=lambda out, tgt: (
                                       out,  tgt.float()) )
    name = data
    epoch_num = epoch_number
    name2 = '_net.pt'
    fullname = name +'_'+str(epoch_num)+'ep_'+str(k)+'f'+ name2
    trainer = utils.Trainer(
        filepath= checkpoints_path+fullname,
        filepath2=None,
        matching_loss=None,
        optimizer_D_2=None,
        optimizer_T_2=None,
        max_epochs=epoch_num,
        feature_loss = nn.L1Loss(),
        loss_func=loss_to_use,
        optimizer_D=D_optim,
        optimizer_T=torch.optim.Adam,
        optimizer_D_opts=D_optimizer_parameters,
        optimizer_T_opts=T_optimizer_parameters,
        labels=labels,
        train_metrics={'l1': nn.L1Loss()},
        val_metrics={'l1': nn.L1Loss()},
        b_size = b_size,
        plugins=[
            # utils.VisdomLogger(env=modeltype),
            utils.EarlyStopping( metric_name='loss', mode='min', patience=5 ),
            utils.ProgressBarGAN(log_path),
            utils.Logger( log_path + 'train_log.csv' )],
        device='cuda' ,
        save='best',
        metric_name='loss',
        mode='min')
    trainer.train_gan(model_D, model_T, train_gen, val_gen)
    model_version = checkpoints_path + fullname
    na = name + '_' + str( epoch_num )
    image_version = results_path + na
    print('Evaluating')
    if saveimage == 1:
        synth_metrics = evaluate_model( test_case_paths, model_version,image_version,metrics_path, k, save_img=True )
    else:
        synth_metrics = evaluate_model( test_case_paths, model_version, image_version, metrics_path, k, save_img=False )
    # print( synth_metrics )

    print( '\u001b[1m Process done\u001b[0m! Check the results!' )
    end = time.time()
    print('\nFold \u001b[1m{}\u001b[0m needed in total \u001b[1m{:2f} Minutes\u001b[0m to be evaluated! '.format(k, (end-start)/60))

realend = time.time()
print('\nTo evaluate the dataset of {} folds, we needed \u001b[1m{:.2f} Hours\u001b[0m. Good Bye!'.format(len(folds), (realend-start)/900))

    #------------OLD THINGS------------------------------------------------------------------------------
    # model_T = models.UNet_LRes(in_channel=2, n_classes=1)
    # model_T = mod.Unet(input_size=2, output_size=1)
    # model_T = models.UNet(in_channel=1, n_classes=1)
    # model_T = models.ResUNet(in_channel=2, n_classes=1)
    # model_T = models.Thief(in_channel=2, n_classes=1)
    # model_T = models.ResUNet_LRes(in_channel=2,n_classes=1)
    # model_T = mod.Generatori(g_input_dim=2, g_output_dim=1)
    # model_D = mod.Discriminator(input_size=1, output_size=1)
    # model_D = mod.DiscriminatorBody(k =16, outputsize=1,inputsize=1 )
    # model_D = mod.Discriminatore( output_size=1,input_size=1 )
    # model_D = mod.AnotherDisc( 1,1 )
    # model_D = mod.Discriminatori(1)
    # model_D = mod.NLayerDiscriminator(1,20,3)
    # model_D = mod.PixelDiscriminator(1,1)
    # loss_to_use = nl.loss.LossWrapper(torch.nn.CrossEntropyLoss(),
    #           preprocess_fn= lambda out, tgt: (
    #           out, torch.argmax(F.softmax(tgt,dim =1),dim =1).long()))

    # torch.argmax( F.softmax( tgt, dim=1 ), dim=1 ).float()) )
    # loss_to_use = nn.MSELoss()