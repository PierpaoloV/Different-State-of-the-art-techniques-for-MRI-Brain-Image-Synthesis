import numpy as np
import os
import nibabel as nib
import torch
import niclib as nl
import synthunet.Functions.models  as mod
import torch.nn as nn
import synthunet.Functions.dataprep as prep
import tensorflow as tf
from matplotlib import pyplot as plt
import pdb

import visdom
from skimage.exposure import match_histograms
from niclib.utils import print_progress_bar, RemainingTimeEstimator, save_nifti
from niclib.generator import make_generator
from niclib.generator.patch import PatchSet, sample_centers_uniform, _get_patch_slice

class PatchTester:
    """Forward pass a volume through the given network using uniformly sampled patches. After a patch is predicted, it
    is accumulated by averaging back in a common space.

    :param patch_shape: tuple (X, Y, Z) with the input patch shape of the model.
    :param patch_out_shape: (default: None) shape of the network forward passed patch, if None it is assumed to be of
        the same shape as ``patch_shape``.
    :param extraction_step: tuple (X, Y, Z) with the extraction step to uniformly sample patches.
    :param str normalize: either 'none', 'patch' or 'image'.
    :param activation: (default: None) the output activation after the forward pass.
    :param int batch_size: (default: 32) batch size for prediction, bigger batch sizes can speed up prediction if
        gpu utilization (NOT gpu memory) is under 100%.
    """

    def __init__(self, patch_shape, extraction_step, normalize, activation=None, batch_size=32, patch_out_shape=None):
        self.in_shape = patch_shape
        self.out_shape = patch_shape if patch_out_shape is None else patch_out_shape
        self.extraction_step = extraction_step
        self.normalize = normalize
        self.bs = batch_size
        self.activation=activation

        assert len(extraction_step) == 3, 'Please give extraction step as (X, Y, Z)'
        assert len(self.in_shape) == len(self.out_shape) ==  4, 'Please give shapes as (CH, X, Y, Z)'

        self.num_ch_out = self.out_shape[0]

    def predict(self, model, x, mask=None, device='cuda'):
        """ Predict the given volume ``x`` using the provided ``model``.

        :param torch.nn.Module model: The trained torch model.
        :param x: the input volume with shape (CH, X, Y, Z) to predict.
        :param mask: (default: None) a binary array of the same shape as x that defines the ROI for patch extraction.
        :param str device: the torch device identifier.
        :return: The accumulated outputs of the network as an array of the same shape as x.
        """
        assert len(x.shape) == 4, 'Please give image with shape (CH, X, Y, Z)'

        # Create patch generator with known patch center locations.
        x_centers = nl.generator.sample_centers_uniform(x[0], self.in_shape[1:], self.extraction_step, mask=mask)
        x_slices = nl.generator.patch._get_patch_slice(x_centers, self.in_shape[1:])
        patch_gen = nl.generator.make_generator(
            prep.PatchSet([x], self.in_shape[1:], None, self.normalize, centers=[x_centers]), self.bs, shuffle=False)

        # Put accumulation in torch (GPU accelerated :D)
        voting_img = torch.zeros((self.num_ch_out,) + x[0].shape, device=device).float()
        counting_img = torch.zeros_like(voting_img).float()

        # Perform inference and accumulate results in torch (GPU accelerated :D (if device is cuda))
        model.eval()
        model.to(device)
        with torch.no_grad():
            rta = nl.utils.RemainingTimeEstimator(len(patch_gen))

            for n, (x_patch, x_slice) in enumerate(zip(patch_gen, x_slices)):
                x_patch = x_patch.to(device)

                y_pred = model(x_patch)


                if self.activation is not None:
                    y_pred = self.activation(y_pred)

                batch_slices = x_slices[self.bs * n:self.bs * (n + 1)]
                for predicted_patch, patch_slice in zip(y_pred, batch_slices):
                    voting_img[patch_slice] += predicted_patch
                    counting_img[patch_slice] += torch.ones_like(predicted_patch)
                    # counting_img += predicted_patch > 0
                nl.utils.print_progress_bar(self.bs * n, self.bs * len(patch_gen), suffix="patches predicted - ETA: {}".format(rta.update(n)))
            nl.utils.print_progress_bar(self.bs * len(patch_gen), self.bs * len(patch_gen), suffix="patches predicted - ETA: {}".format(rta.elapsed_time()))

        counting_img[counting_img == 0.0] = 1.0  # Avoid division by 0
        predicted_volume = torch.div(voting_img, counting_img).detach().cpu().numpy()
        return predicted_volume



def evaluate_model(test_case_paths, model_version, image_version, metrics_path,fold, save_img = True):
    unet_trained = torch.load( model_version )
    v = visdom.Visdom()
    predictor = nl.net.test.PatchTester(
        patch_shape=(1, 32, 32, 32),
        patch_out_shape=(1, 32, 32, 32),
        extraction_step=(6, 6, 6),
        normalize='none',
        activation=None )

    dataset_test = nl.parallel_load( load_func=prep.load_case_brats, arguments=test_case_paths, num_workers=12 )
    # dataset_test2 = nl.parallel_load( load_func=load_case2, arguments=test_case_paths, num_workers=12 )
    synthetic_images = []
    original_images = []
    metric = []
    ids =[]
    mse =[]
    std_ssim = []
    std_mse = []
    ssim =[]
    snr = []
    for n, case in enumerate( dataset_test ):
        # b = np.squeeze(case['flair_unprocessed'])
        b = case['domain_2_unprocessed']
        # Predict image with the predictor and store
        print( "Synthetising image {}".format( n ) )
        synth_img = predictor.predict( unet_trained, case['domain_1'] )
        na = 'Image '+case['id']

        # Add the input and output images to a list for metrics computation
        synthetic_images.append( synth_img )
        # synth_img_2 = match_histograms( synth_img, b )

        # v.image( np.squeeze( synth_img[:, :, 90], axis=0 ), win=na, env='Images' )
        if save_img:
            nl.save_nifti(
                filepath=image_version + '_img_{}_synthetic.nii.gz'.format( case['id'] ),
                volume=np.squeeze( synth_img, axis=0 ),  # Remove single channel dimension for storage
                reference=case['nifti']
                # channel_handling='split'
            )
            # nl.save_nifti(
            #     filepath=image_version + '_img_{}_synthetic_unmatched.nii.gz'.format( case['id'] ),
            #     volume=np.squeeze( synth_img_2, axis=0 ),  # Remove single channel dimension for storage
            #     reference=case['nifti']
            #     # channel_handling='split'
            # )
        ids.append(case['id'])
        original_images.append( case['domain_2'] )

    # Compute the metric
        print("Calculating metrics for image {}".format( n ) )
        synth_metrics = nl.metrics.compute_metrics( outputs=synthetic_images,
                                                    targets=original_images,
                                                    ids= ids,
                                                    metrics={
                                                     'mse': nl.metrics.mse,
                                                     'ssim': nl.metrics.ssim,
                                                     #'psnr': tf.image.psnr(max_val =1.0),
                                                 } )
        metric.append(synth_metrics)
        mse.append(synth_metrics[n]['mse'])
        ssim.append( synth_metrics[n]['ssim'] )
        # for (a,b) in zip(synthetic_images, original_images):
        #     tmp = tf.image.psnr(a, b, max_val =1.0)
        #     snr.append(tmp)
        #     print(tmp+'\n')
        #snr.append( synth_metrics[n]['psnr'] )

    std_ssim = np.std(ssim)
    std_mse = np.std(mse)
    avgmse = np.average(mse)
    mini = np.min(mse)
    maxi = np.max(mse)
    mins = np.min(ssim)
    maxs = np.max(ssim)
    avgssim = np.average(ssim)
    # avgsnr = np.average(snr)
    # maxsnr = np.max(snr)
    # minsnr = np.min(snr)
    avg = [{'Avg_MSE': avgmse, 'Min_MSE': mini, 'Max_MSE': maxi,'St.Dev_MSE': std_mse, 'Avg_SSIM': avgssim, 'Min_SSIM': mins,
             'Max_SSIM': maxs,'St.Dev_SSID':std_ssim}]#,'Avg_PSNR': avgsnr, 'Max_PSNR': maxsnr, 'Min_PSNR': minsnr }]
    nameavg = 'average_metrics_fold'+str(fold)
    namesave = 'synthetisation_metrics_fold'+str(fold)
    endnamea =namesave+'.csv'
    endnameb = nameavg+'.csv'


    nl.save_to_csv( metrics_path + endnamea, synth_metrics )
    nl.save_to_csv( metrics_path + endnameb, avg )

    return synth_metrics

def evaluate_cyclegan(test_case_paths,dset,  model_version,model_2_version, image_version, metrics_path,fold,predict = 1, save_img = True):
    
    translator_1 = torch.load( model_version )
    translator_2 = torch.load( model_2_version )
    if dset ==0:
        dataset_test = nl.parallel_load( load_func=prep.load_case_whm, arguments=test_case_paths, num_workers=12 )
    # dataset = nl.parallel_load( load_func=load_case, arguments=[[case, x] for case, x in zip( test_case_paths, flag )],
    #                             num_workers=12 )
    else:
        dataset_test = nl.parallel_load( load_func=prep.load_case_brats, arguments=test_case_paths, num_workers=12 )


    if predict ==1:
        synthetic_imgs, cyclic_imgs, inverse_1, inverse_2, ids, original_imgs, t1_imgs = synthetize_img(translator_1,
                                                                                                        translator_2,
                                                                                                        dataset_test,
                                                                                                        image_version,
                                                                                                        save_img,
                                                                                                        predict)
    else:
        synthetic_imgs, cyclic_imgs, inverse_1, inverse_2, ids, original_imgs, t1_imgs = synthetize_img( translator_1,
                                                                                                         translator_2,
                                                                                                         dataset_test,
                                                                                                         image_version,
                                                                                                         save_img,
                                                                                                         predict )
    new_synth = matchandsave(synthetic_imgs,'synthetic', dataset_test, 'domain_2_unprocessed', image_version,ids)
    new_cyclic = matchandsave(cyclic_imgs,'inverse', dataset_test, 'domain_1_unprocessed', image_version,ids)
    new_inv1 = matchandsave(inverse_1,'F(G(x))', dataset_test, 'domain_1_unprocessed', image_version,ids)
    new_inv2 = matchandsave(inverse_2,'G(F(y))', dataset_test, 'domain_2_unprocessed', image_version,ids)

    get_metrics(dataset_test,fold,metrics_path,ids,'T1_to_Flair_{}'.format(predict), new_synth, t1_imgs)
    get_metrics(dataset_test,fold,metrics_path,ids,'Flair_to_T1_{}'.format(predict), new_cyclic, original_imgs)
    get_metrics(dataset_test,fold,metrics_path,ids,'Cycle_consistent_T1_to_T1_{}'.format(predict), new_inv1, t1_imgs)
    get_metrics(dataset_test,fold,metrics_path,ids,'Cycle_consistent_Flair_to_Flair_{}'.format(predict), new_inv2, original_imgs)

    return

def synthetize_img(model_1, model_2, dataset,image_version, save_img = True, predict =1 ):
    original_imgs, synthetic_imgs, cyclic_imgs, decyclic_imgs, ids, t1_imgs,inverse_1, inverse_2 = [],[],[],[],[],[],[],[]
    if predict == 1:

        predictor = nl.net.test.PatchTester(
            patch_shape=( 1, 32,32,32),
            extraction_step=(8,8,8),
            normalize='none',
            activation=None )
        # np.squeeze(predictor.in_shape)
        # np.squeeze(predictor.out_shape)
    elif predict == 2:
        predictor = PatchTester(
            patch_shape=(1, 1, 32, 32),
            extraction_step=(1, 16, 16),
            normalize='none',
            activation=None )
    else:
        predictor = PatchTester(
            patch_shape=(1, 32, 32, 32),
            patch_out_shape=(1, 32, 32, 32),
            extraction_step=(2, 2, 2),
            normalize='none',
            activation=None )
    for n, case in enumerate( dataset ):
        print( "Synthetising image {}".format( n ) )
        # pdb.set_trace()
        synth_img = predictor.predict( model_1, case['domain_1'] )
        desynt_img = predictor.predict(model_2, synth_img)
        consistent_image = predictor.predict(model_2, case['domain_2'])
        decon_img = predictor.predict( model_1, consistent_image )
        na = 'Image '+case['id']

        # Add the input and output images to a list for metrics computation
        synthetic_imgs.append( synth_img )
        cyclic_imgs.append(consistent_image)
        inverse_1.append(desynt_img)
        inverse_2.append(decon_img)
        ids.append(case['id'])
        original_imgs.append( case['domain_2'] )
        t1_imgs.append(case['domain_1'])
        #
        # synth_img_2 = match_histograms( synth_img, case['domain_2_unprocessed'] )
        # consistent_image_2 = match_histograms(consistent_image, case['domain_1_unprocessed'])
        # desynt_img_2 = match_histograms(desynt_img, case['domain_1_unprocessed'])
        # decon_img_2 = match_histograms(decon_img,case['domain_2_unprocessed'])
        # # synth_img = synth_img.astype(float)
        # if save_img:
        #     nl.save_nifti(
        #         filepath=image_version + '_img_{}_psample_{}_synthetic.nii.gz'.format( case['id'], predict ),
        #         volume=np.squeeze( synth_img_2, axis=0 ),  # Remove single channel dimension for storage
        #         reference=case['nifti']
        #         # channel_handling='split'
        #     )
        #     nl.save_nifti(
        #         filepath=image_version + '_img_{}_psample_{}_F(G(x)).nii.gz'.format( case['id'], predict  ),
        #         volume=np.squeeze( desynt_img_2, axis=0 ),  # Remove single channel dimension for storage
        #         reference=case['nifti']
        #         # channel_handling='split'
        #     )
        #     nl.save_nifti(
        #         filepath=image_version + '_img_{}_psample_{}_G(F(y)).nii.gz'.format( case['id'], predict  ),
        #         volume=np.squeeze( decon_img_2, axis=0 ),  # Remove single channel dimension for storage
        #         reference=case['nifti']
        #         # channel_handling='split'
        #     )
        #
        #     nl.save_nifti(
        #         filepath=image_version + '_img_{}_psample_{}_inverse.nii.gz'.format( case['id'], predict  ),
        #         volume=np.squeeze( consistent_image_2, axis=0 ),  # Remove single channel dimension for storage
        #         reference=case['nifti']
        #         # channel_handling='split'
        #     )

    return synthetic_imgs,cyclic_imgs, inverse_1, inverse_2, ids, original_imgs, t1_imgs

def get_metrics(dataset, fold, metrics_path,ids,name, domain_1, domain_2):
   #For normal synthetic
    metric, mse, ssim, snr, = [],[],[],[]
    for n, case in enumerate( dataset ):
        print("Calculating metrics for image {}".format( n ) )
        synth_metrics = nl.metrics.compute_metrics( outputs=domain_1,
                                                    targets=domain_2,
                                                    ids= ids,
                                                    metrics={
                                                     'mse': nl.metrics.mse,
                                                     'ssim': nl.metrics.ssim
                                                     # 'psnr': nl.metrics.psnr,
                                                 } )
        metric.append(synth_metrics)
        mse.append(synth_metrics[n]['mse'])
        ssim.append( synth_metrics[n]['ssim'] )
        # snr.append( synth_metrics[n]['psnr'] )

    std_ssim = np.std( ssim )
    std_mse = np.std( mse )
    avgmse = np.average( mse )
    mini = np.min( mse )
    maxi = np.max( mse )
    mins = np.min( ssim )
    maxs = np.max( ssim )
    avgssim = np.average( ssim )
    # avgsnr = np.average( snr )
    # maxsnr = np.max( snr )
    # minsnr = np.min( snr )

    avg = [{'Avg_MSE': avgmse, 'Min_MSE': mini, 'Max_MSE': maxi, 'St.Dev_MSE': std_mse, 'Avg_SSIM': avgssim,
            'Min_SSIM': mins,
            'Max_SSIM': maxs, 'St.Dev_SSID': std_ssim}]#, 'Avg_PSNR': avgsnr, 'Max_PSNR': maxsnr, 'Min_PSNR': minsnr}]

    nameavg = 'Average_'+name +'_fold ' +str( fold )
    namesave = 'Imagewise_'+name+'_fold ' + str( fold )
    endnamea = namesave + '.csv'
    endnameb = nameavg + '.csv'

    nl.save_to_csv( metrics_path + endnamea, synth_metrics )
    nl.save_to_csv( metrics_path + endnameb, avg )
    return

def matchandsave(synt_image, name, dataset, casename, image_version, ids):
    imgs = []
    for n, case in enumerate(dataset):
        print('Matching and saving image {}'.format(n))
        synth_img_2 = match_histograms(synt_image[n], case[casename] )
        # consistent_image_2 = match_histograms( consistent_image, case['domain_1_unprocessed'] )
        # desynt_img_2 = match_histograms( desynt_img, case['domain_1_unprocessed'] )
        # decon_img_2 = match_histograms( decon_img, case['domain_2_unprocessed'] )
        #
        nl.save_nifti(
            filepath=image_version +name+ '_img_{}_.nii.gz'.format(ids[n]),
            volume=np.squeeze( synth_img_2, axis=0 ),  # Remove single channel dimension for storage
            reference=case['nifti']
            # channel_handling='split'
        )
        nl.save_nifti(
            filepath=image_version + name + '_img_{}_original.nii.gz'.format( ids[n] ),
            volume=np.squeeze( synt_image[n], axis=0 ),  # Remove single channel dimension for storage
            reference=case['nifti']
            # channel_handling='split'
        )
        imgs.append(synth_img_2)
    return imgs