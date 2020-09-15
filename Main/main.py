import os
import torch
import niclib as nl
from synthunet.Functions.test import evaluate_model
import Functions.options as opt
import synthunet.Functions.models  as mod
import torch.nn as nn
import Functions.preprocessing as prep


data_path,checkpoints_path, results_path, metrics_path, log_path, mainfolder = prep.preparedata()
case_path = [f.path for f in os.scandir(data_path) if f.is_dir()]
modeltype,model,num_params, G_optimizer_parameters = opt.build_Generator()
folds = [0,1,2,3]
# folds = [0]
for k in folds:
    print('Getting the fold {}'.format(k))
    case_paths,test_case_paths =prep.get_datax(case_path.copy(),k)
    print("Loading training dataset with {} images in fold {}...".format(len(case_paths),k))

    dataset = nl.parallel_load(load_func=prep.load_case_whm, arguments=case_paths, num_workers=12)
    dataset_train, dataset_val = nl.split_list(dataset, fraction=0.8)  # Split images into train and validation
    print('Training dataset with {} train and {} val images'.format(len(dataset_train), len(dataset_val)))

    step = 1
    samples = 2304
    train_patch_set = prep.get_patch_gen(dataset_train, step, samples, normalize_opt= 'none')
    val_patch_set = prep.get_patch_gen(dataset_val, step, samples, normalize_opt= 'none')

    train_gen = nl.generator.make_generator(set=train_patch_set, batch_size=32, shuffle=True)
    val_gen = nl.generator.make_generator(set=val_patch_set, batch_size=32, shuffle=True)
    print("Train and val patch generators with {} and {} patches".format(len(train_patch_set), len(val_patch_set)))


    name = modeltype
    epoch_num = 20
    name2 = 'net.pt'
    fullname = name +'_'+str(epoch_num)+'ep_'+str(k)+'f_'+ name2
    trainer = nl.net.train.Trainer(
        max_epochs=epoch_num,
        loss_func=nn.L1Loss(),
        optimizer=torch.optim.Adam,
        optimizer_opts={'lr': G_optimizer_parameters},
        train_metrics={'l1': nn.L1Loss()},
        val_metrics={'l1': nn.L1Loss()},
        plugins=[
            nl.net.train.ProgressBar(),
            nl.net.train.ModelCheckpoint(checkpoints_path + fullname, save='best', metric_name='loss', mode='min'),
            nl.net.train.EarlyStopping(metric_name='loss', mode='min', patience=5),
            nl.net.train.Logger(log_path + 'train_log.csv')],
        device='cuda')
    trainer.train(model, train_gen, val_gen)
    model_version = checkpoints_path + fullname
    image_version = results_path + name
    synth_metrics = evaluate_model( test_case_paths, model_version,image_version,metrics_path, k,
                                    save_img=True )

    print( 'Process done! Check the results!' )
