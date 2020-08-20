import os

import torch
from torch.nn import functional as F

import synthunet.Functions.utils as utils
import niclib as nl
import numpy as np
import synthunet.Functions.models  as mod
import synthunet.Functions.models_new  as models
import torch.nn as nn
import time
import synthunet.Functions.preutils as pre
from synthunet.Functions.dataprep import preparedata, load_case_whm,load_case_brats, get_datax, get_patch_gen,get_patch_gen_3input,splitdata,get_patch_gen_3input2d
from synthunet.Functions.test import evaluate_model,evaluate_cyclegan

# ------------------------------------------------------------------------------
print( 'Initializing Options...\n' )
dset = pre.get_dataset()

modeltype, model_T_1, num_params_T, T_optimizer_parameters = pre.build_Generator()

model_D_1, D_optim, D_optimizer_parameters,  num_params_D,labels = pre.build_Discriminator()

modeltype_2, model_T_2, num_params_T_2, T_2_optimizer_parameters = pre.build_Generator()
model_D_2, D_optim_2, D_optimizer_parameters_2, num_params_D_2,_ = pre.build_Discriminator()
b_size,flag, epoch_number, folds, samplesnumber, saveimage, lam = pre.build_Backbone()

data = 'CycleGAN_' + modeltype+ '_'+ modeltype_2
data_path, checkpoints_path, results_path, metrics_path, log_path, mainfolder = preparedata( dset )
summary = log_path + 'summary.txt' #name of the txt plus the directory
f = open( summary, 'a' )
f.write( 'Recap: \n'
         'Discriminator Number of Parameters:' + str( num_params_D / 1000000 ) + ' Milions \n'
         'Discriminator Optimizer:' + str(D_optim ) + '\n'
         'Discriminator Optimizer parameters:' + repr( D_optimizer_parameters ) + '\n'
         'Hard - Soft Labels:' + str(labels ) + '\n'
         '\n'
         'Translator_1 Selected:' + modeltype + ' \n'
         'Translator Number of Parameters: ' + str(num_params_T / 1000000 ) + ' Milions \n'
         'Translator Optimizer Parameters:' + repr( T_optimizer_parameters ) + ' \n'
         '\n'
         'Translator_2 Selected:' + modeltype_2 + ' \n'
         'Translator Number of Parameters: ' + str(num_params_T_2 / 1000000 ) + ' Milions \n'
         'Translator Optimizer Parameters:' + repr( T_2_optimizer_parameters ) + ' \n'
         '\n'
         'Fold(s) to analyze: ' + str(folds ) + ' \n'
         'Number of Epochs to train: ' + str( epoch_number ) + ' \n'
         'Batch size: ' + str( b_size ) + '\n'
         'Number of Samples per Image: ' + str(samplesnumber ) + ' \n'
         ) #things to write
f.close()
nl.print_big('RECAP')
print(
       'Discriminator Optimizer: \u001b[1m{}\u001b[0m \n'
       'Translator 1 Selected: \u001b[1m{}\u001b[0m \n'
       'Translator 2 Selected: \u001b[1m{}\u001b[0m \n'
       'Fold(s) to analyze: \u001b[1m{}\u001b[0m \n'
       'Number of Epochs to train: \u001b[1m{}\u001b[0m \n'
       'Batch size: \u001b[1m{}\u001b[0m \n'
       'Number of Samples per Image: \u001b[1m{}\u001b[0m \n'
       'Save - Discard synthetized images: \u001b[1m{}\u001b[0m \n'
       'Folder Create: \u001b[1m{}\u001b[0m \n'.format(D_optim,
                                                       modeltype,
                                                       modeltype_2,
                                                       folds,epoch_number, b_size, samplesnumber, saveimage, mainfolder, ) )
nl.print_big('END RECAP')
# --------------------------------------------------------

case_path = [f.path for f in os.scandir( data_path ) if f.is_dir()]
flag = [flag]
tot = time.time()
for k in folds:
    start = time.time()
    print( 'Getting the fold \u001b[1m{}\u001b[0m'.format( k ) )
    if dset == 0:
        case_paths, test_case_paths = get_datax( case_path.copy(), k )
        print( "Loading training dataset with \u001b[1m{}\u001b[0m images in fold \u001b[1m{}\u001b[0m...".format(
            len( case_paths ), k ) )
        a = time.time()
        dataset = nl.parallel_load( load_func=load_case_whm, arguments=case_paths, num_workers=12 )

    else:
        case_paths, test_case_paths = splitdata( case_path.copy(), k )
        print( "Loading training dataset with \u001b[1m{}\u001b[0m images in fold \u001b[1m{}\u001b[0m...".format(
            len( case_paths ), k ) )
        a = time.time()
        dataset = nl.parallel_load( load_func=load_case_brats, arguments=case_paths, num_workers=12 )
    # dataset = nl.parallel_load( load_func=load_case, arguments=[[case, flag] for case in case_paths], num_workers=12 )
    b = time.time()
    dataset_train, dataset_val = nl.split_list( dataset, fraction=0.8 )  # Split images into train and validation
    print( 'Training dataset with \u001b[1m{}\u001b[0m train and \u001b[1m{}\u001b[0m val images'.format(
        len( dataset_train ), len( dataset_val ) ) )
    print( 'needed \u001b[1m{:.2f} seconds\u001b[0m to load and split the images'.format( b - a ) )

    step = 1
    samples = samplesnumber
    c = time.time()
    # print('Sampling With the mask :)')
    # if modeltype =='Unet2D' or '6' or '7' or 'ResUnet_2D':
    #     train_patch_set = get_patch_gen_3input2d( dataset_train, step, samples, normalize_opt='none' )
    #     val_patch_set = get_patch_gen_3input2d( dataset_val, step, samples, normalize_opt='none' )
    # else:
    #     train_patch_set = get_patch_gen_3input( dataset_train, step, samples, normalize_opt='none' )
    #     val_patch_set = get_patch_gen_3input( dataset_val, step, samples, normalize_opt='none' )
    # # assert np.sum(train_patch_set) > 0
    train_patch_set = get_patch_gen_3input( dataset_train, step, samples, normalize_opt='none' )
    val_patch_set = get_patch_gen_3input( dataset_val, step, samples, normalize_opt='none' )
    train_gen = nl.generator.make_generator( set=train_patch_set, batch_size=b_size, shuffle=True )
    val_gen = nl.generator.make_generator( set=val_patch_set, batch_size=b_size, shuffle=True )
    d = time.time()
    print(
        "Train and val patch generators with {} and {} patches".format( len( train_patch_set ), len( val_patch_set ) ) )
    print( 'Needed {:.2f} minutes to generate {} patches. '.format( (d - c) / 60, samplesnumber ) )
    loss_to_use = nl.loss.LossWrapper( torch.nn.BCEWithLogitsLoss(),
                                       preprocess_fn=lambda out, tgt: (
                                           out, tgt.float()) )
    name = data
    epoch_num = epoch_number
    name2 = '_net.pt'
    fullname = name + '_' + str( epoch_num ) + 'ep_' + str( k ) + 'f' + name2
    fullname_2 = name + '_' + str( epoch_num ) + 'ep_' + str( k ) + 'f' +'_cycle'+ name2
    trainer = utils.Trainer(
        filepath=checkpoints_path + fullname,
        filepath2=checkpoints_path + fullname_2,
        max_epochs=epoch_num,
        feature_loss=nn.L1Loss(),
        matching_loss = nn.MSELoss(),
        loss_func=loss_to_use,
        optimizer_D=D_optim,
        lam = lam,
        optimizer_D_2=D_optim_2,
        optimizer_T=torch.optim.Adam,
        optimizer_T_2=torch.optim.Adam,
        optimizer_D_opts=D_optimizer_parameters,
        optimizer_D_2_opts=D_optimizer_parameters_2,
        optimizer_T_opts=T_optimizer_parameters,
        optimizer_T_2_opts=T_2_optimizer_parameters,
        labels=labels,
        train_metrics={'l1': nn.L1Loss()},
        val_metrics={'l1': nn.L1Loss()},
        b_size=b_size,
        plugins=[
            utils.VisdomLogger( env=modeltype ),
            # utils.EarlyStopping( metric_name='loss', mode='min', patience=5 ),
            utils.ProgressBarGAN(log_path),
            utils.Logger( log_path + 'train_log.csv' )],
        device='cuda',
        save='best',
        metric_name='loss',
        mode='min' )
    trainer.train_cyclegan( model_D_1,model_D_2, model_T_1,model_T_2, train_gen, val_gen )
    model_version = checkpoints_path + fullname
    model_2_version = checkpoints_path + fullname_2
    na = name + '_' + str( epoch_num )
    image_version = results_path + na
    print( 'Evaluating' )
    if saveimage == 1:
        # synth_metrics = evaluate_cyclegan( test_case_paths,flag, model_version,model_2_version, image_version, metrics_path,
        #                                    k, save_img = True, intermediate_img = True )
        print( 'predicting with sampling step = 10,10,10' )
        evaluate_cyclegan(test_case_paths,dset,  model_version,model_2_version, image_version, metrics_path,k, 1,
                          save_img = True)
        # print( 'predicting with sampling step = 16,16,16' )
        # evaluate_cyclegan( test_case_paths, dset, model_version, model_2_version, image_version, metrics_path, k, 2,
        #                    save_img=True )
        # evaluate_cyclegan( test_case_paths, dset, model_version, model_2_version, image_version, metrics_path, k, 5,
        #                    save_img=True )

    else:
        print('predicting with sampling step = 4,4,4')
        evaluate_cyclegan( test_case_paths, dset,  model_version, model_2_version, image_version, metrics_path, k,1,
                           save_img=False )
        print( 'predicting with sampling step = 16,16,16' )
        evaluate_cyclegan( test_case_paths, dset,  model_version, model_2_version, image_version, metrics_path, k,2,
                           save_img=False )
        evaluate_cyclegan( test_case_paths, dset,  model_version, model_2_version, image_version, metrics_path, k,5,
                           save_img=False )

    print( '\u001b[1m Process done\u001b[0m! Check the results!' )
    end = time.time()
    print( '\nFold \u001b[1m{}\u001b[0m needed in total \u001b[1m{:2f} Minutes\u001b[0m to be evaluated! '.format( k, (
                end - start) / 60 ) )

realend = time.time()
print(
    '\nTo evaluate the dataset of {} folds, we needed \u001b[1m{:.2f} Hours\u001b[0m.'.format( len( folds ),
    (realend - tot) / 3600 ) )

nl.print_big('Good Bye!')