import os
import torch
from synthunet.Functions.test import evaluate_model
import niclib as nl
import synthunet.Functions.models  as mod
import torch.nn as nn
from synthunet.Functions.dataprep import preparedata, load_case_brats, get_patch_gen, splitdata,get_datax,load_case_whm


data = 'resunet_flair-t1_'
data_path,checkpoints_path, results_path, metrics_path, log_path, mainfolder = preparedata(1, data)
case_path = [f.path for f in os.scandir(data_path) if f.is_dir()]

folds = [0,1,2,3]
# folds = [0]
for k in folds:
    print('Getting the fold {}'.format(k))
    case_paths,test_case_paths =splitdata(case_path.copy(),k)
    print("Loading training dataset with {} images in fold {}...".format(len(case_paths),k))

    dataset = nl.parallel_load(load_func=load_case_brats, arguments=case_paths, num_workers=12)
    dataset_train, dataset_val = nl.split_list(dataset, fraction=0.8)  # Split images into train and validation
    print('Training dataset with {} train and {} val images'.format(len(dataset_train), len(dataset_val)))

    step = 1
    samples = 768
    train_patch_set = get_patch_gen(dataset_train, step, samples, normalize_opt= 'none')
    val_patch_set = get_patch_gen(dataset_val, step, samples, normalize_opt= 'none')

    train_gen = nl.generator.make_generator(set=train_patch_set, batch_size=32, shuffle=True)
    val_gen = nl.generator.make_generator(set=val_patch_set, batch_size=32, shuffle=True)
    print("Train and val patch generators with {} and {} patches".format(len(train_patch_set), len(val_patch_set)))

    model = mod.ResUNet(inputsize=1, outputsize=1, k=16)
    # model_T = mod.ResUnette( in_classes=1, out_classes=1 )
    # model = mod.Unet(input_size=1, output_size=1)

    name = data
    epoch_num = 3
    name2 = '_net.pt'
    fullname = name +'_'+str(epoch_num)+'ep_'+str(k)+'f_'+ name2
    trainer = nl.net.train.Trainer(
        max_epochs=epoch_num,
        loss_func=nn.L1Loss(),
        optimizer=torch.optim.Adam,
        optimizer_opts={'lr': 0.0001},
        train_metrics={'l1': nn.L1Loss()},
        val_metrics={'l1': nn.L1Loss()},
        plugins=[
            nl.net.train.ProgressBar(),
            nl.net.train.ModelCheckpoint(checkpoints_path + fullname, save='best', metric_name='loss', mode='min'),
            nl.net.train.EarlyStopping(metric_name='loss', mode='min', patience=3),
            nl.net.train.Logger(log_path + 'train_log.csv')],
        device='cuda')

    trainer.train(model, train_gen, val_gen)
    model_version = checkpoints_path + fullname
    image_version = results_path + name
    synth_metrics = evaluate_model( test_case_paths, model_version,image_version,metrics_path, k,
                                    save_img=True )
    # print( synth_metrics )

    print( 'Process done! Check the results!' )

#%%

#----------ONLY FOR TESTING AFTER TRAINING---------------------------
import os

from synthunet.Functions.test import evaluate_model
from synthunet.Functions.dataprep import preparedata, get_datax

PREFIX = '/home/pierpaolo/home/pierpaolo/synthunet/'
# data = 'b'
# data_path, checkpoints_path, results_path, metrics_path, log_path = preparedata(data)
# case_path = [f.path for f in os.scandir(data_path) if f.is_dir()]
# k = 0
# name = 'ResUnet'
# epoch_num = 5
# name2 = '_net.pt'
# fullname = name +'_'+str(epoch_num)+'ep_'+str(k)+'f_'+ name2
#
# print('Getting the fold {}'.format(k))
# case_paths,test_case_paths = get_datax(case_path.copy(),k)
# model_version = PREFIX + checkpoints_path + fullname
# image_version = results_path + name
#
# synth_metrics = evaluate_model( test_case_paths, model_version, PREFIX + image_version, PREFIX + metrics_path, k, save_img=True )
# print(synth_metrics)
#
# print('Process done! Check the results!')

#%% GAN TRIAL
#
# import os
# import torch
# from synthunet.Functions.test import evaluate_model
# import niclib as nl
# import synthunet.Functions.models  as mod
# import torch.nn as nn
# from synthunet.Functions.dataprep import preparedata, load_case,get_datax, get_patch_gen
#
#
# data = 'b'
# data_path, checkpoints_path, results_path, metrics_path, log_path = preparedata(data)
# case_path = [f.path for f in os.scandir(data_path) if f.is_dir()]
#
# folds = [0,1,2,3]
# # folds = [3]
# for k in folds:
#     print('Getting the fold {}'.format(k))
#     case_paths,test_case_paths =get_datax(case_path.copy(),k)
#     print("Loading training dataset with {} images in fold {}...".format(len(case_paths),k))
#
#     dataset = nl.parallel_load(load_func=load_case, arguments=case_paths, num_workers=12)
#     dataset_train, dataset_val = nl.split_list(dataset, fraction=0.8)  # Split images into train and validation
#     print('Training dataset with {} train and {} val images'.format(len(dataset_train), len(dataset_val)))
#
#     step = 1
#     samples = 1000
#     train_patch_set = get_patch_gen(dataset_train, step, samples, normalize_opt= 'none')
#     val_patch_set = get_patch_gen(dataset_val, step, samples, normalize_opt= 'none')
#
#     train_gen = nl.generator.make_generator(set=train_patch_set, batch_size=32, shuffle=True)
#     val_gen = nl.generator.make_generator(set=val_patch_set, batch_size=32, shuffle=True)
#     print("Train and val patch generators with {} and {} patches".format(len(train_patch_set), len(val_patch_set)))
#
#     generator = mod.ResUNet(inputsize=2, outputsize=1, k=16)
#     discriminator = mod.Discriminator(input_size=1,output_size=1)
#
#     optimizer_G = torch.optim.Adadelta()
#     optimizer_D = torch.optim.Adadelta()
#     #model = mod.Unet(input_size=1, output_size=1)
#     name = 'Unet_single'
#     epoch_num = 20
#     name2 = '_net.pt'
#     fullname = name +'_'+str(epoch_num)+'ep_'+str(k)+'f_'+ name2
#     adversarial_loss = torch.nn.BCELoss()
#     for epoch in range(epoch_num):
#
#         optimizer_G.zero_grad()
#         #pass the imput img
#         z=0
#         valid_gt =0
#         real_imgs = 0
#         fake =0
#
#
#         #generate img
#         gen_imgs = generator(z)
#         # Loss measures generator's ability to fool the discriminator
#         g_loss = adversarial_loss( discriminator( gen_imgs ), valid_gt )
#
#         g_loss.backward()
#         optimizer_G.step()
#
#         # ---------------------
#         #  Train Discriminator
#         # ---------------------
#
#         optimizer_D.zero_grad()
#
#         # Measure discriminator's ability to classify real from generated samples
#         real_loss = adversarial_loss( discriminator( real_imgs ), valid_gt )
#         fake_loss = adversarial_loss( discriminator( gen_imgs.detach() ), fake )
#         d_loss = (real_loss + fake_loss) / 2
#
#         d_loss.backward()
#         optimizer_D.step()
#
#         print(
#             "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
#             % (epoch, opt.n_epochs, i, len( dataloader ), d_loss.item(), g_loss.item())
#         )
#
#     model_version = checkpoints_path + fullname
#     image_version = results_path + name
#     synth_metrics = evaluate_model( test_case_paths, model_version,image_version,metrics_path, k,
#                                     save_img=False )
#     print( synth_metrics )
#
#     print( 'Process done! Check the results!' )
