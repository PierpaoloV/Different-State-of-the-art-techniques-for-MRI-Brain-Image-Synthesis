import nibabel as nib
import os
import numpy as np
import torch
import niclib as nl
import synthunet.Functions.preutils as pre
import copy
import itertools
from abc import ABC, abstractmethod
from typing import List
from torch.utils.data import Dataset as TorchDataset
from sklearn.model_selection import KFold

def scannerlist(data_path):
    case_folders = sorted( [f.path for f in os.scandir(data_path) if f.is_dir()] )
    dims = []
    for n, cf in enumerate( case_folders ):
        dims.append(nib.load(os.path.join(cf, '3DT1_brain.nii.gz')).get_data().shape)
        #dims.append( np.around( nib.load( os.path.join( cf, '3DT1_brain.nii.gz' ) ).header['pixdim'], decimals=3 ) )

    unique_dims = []
    for d in dims:
        # Check if existing in unique dims
        for ud in unique_dims:
            if np.array_equal( d, ud ):
                break
        else:
            unique_dims.append( d )
    return unique_dims


def preparedata(data =1, name = 'unet'):
    day = nl.get_timestamp(time_format='%d_%m_%Y')
    if data == 1:
        challenge = 'Brats2018'
        data_path = '/home/pierpaolo/Datasets/Brats18TrainingData'

    elif data == 0:
        challenge = 'WMH20173D'
        data_path = '/home/pierpaolo/Datasets/WMH20173D'
    else:
        return

    mainfolder = nl.make_dir(day + '_' + str(challenge)+name)
    checkpoints_path = nl.make_dir(os.path.join(mainfolder, 'checkpoints/'))
    results_path = nl.make_dir(os.path.join(mainfolder, 'results/'))
    metrics_path = nl.make_dir(os.path.join(mainfolder, 'metrics/'))
    log_path = nl.make_dir(os.path.join(mainfolder, 'log/'))

    return data_path,checkpoints_path, results_path, metrics_path, log_path, mainfolder

def load_case_whm(case_paths, flag = 0):
    # Dictionary containing at first the img and the gt
    t1_nifti = nib.load(os.path.join(case_paths, '3DT1_brain.nii.gz'))
    t1_img = (t1_nifti.get_data())
    t1_img = nl.data.clip_percentile(t1_img, [0.0, 99.99])  # Clip to ignore bright extrema
    t1_img = nl.data.adjust_range(t1_img, [0.0, 1.0])  # Adjust range to 0-1 for Sigmoid activation
    t1_img = np.expand_dims( t1_nifti.get_data(), axis=0 )  # Add single channel modality
    ob = t1_img

    gt_nifti = nib.load(os.path.join(case_paths, '3DFLAIR_brain.nii.gz'))
    gt_img = np.expand_dims(gt_nifti.get_data(), axis=0)
    gt_img = nl.data.clip_percentile(gt_img, [0.0, 99.99])  # Clip to ignore bright extrema
    gt_img = nl.data.adjust_range(gt_img, [0.0, 1.0])  # Adjust range to 0-1 for Sigmoid activation

    gt_nifti2 = nib.load(os.path.join(case_paths, '3DFLAIR_brain.nii.gz'))
    gt_match = np.expand_dims(gt_nifti2.get_data(), axis=0)
    # gt_img =gt_nifti.get_data()
    gt_nifti3 = nib.load(os.path.join(case_paths, '3DT1_brain.nii.gz'))
    t1_match = np.expand_dims(gt_nifti3.get_data(), axis=0)


    return {'id': case_paths.split('/')[-1], 'nifti': t1_nifti, 'domain_1': ob,  'domain_2': gt_img,
            'domain_2_unprocessed': gt_match, 'domain_1_unprocessed': t1_match}

def load_case_brats(case_paths, flag = 0):
    # Dictionary containing at first the img and the gt
    a = nl.get_filename(str(case_paths))
    name = a.replace( "\']", "" )
    name = name.replace( "/", "" )
    t1_nifti = nib.load(os.path.join(case_paths, name+'_flair.nii.gz'))
    t1_img = (t1_nifti.get_data())
    t1_img = nl.data.clip_percentile(t1_img, [0.0, 99.99])  # Clip to ignore bright extrema
    t1_img = nl.data.adjust_range(t1_img, [0.0, 1.0])  # Adjust range to 0-1 for Sigmoid activation
    t1_img = np.expand_dims( t1_nifti.get_data(), axis=0 )  # Add single channel modality
    ob = t1_img
 # Adjust range to 0-1 for Sigmoid activation

    t1_nifti2 = nib.load(os.path.join(case_paths, name+'_flair.nii.gz'))
    t1_img2 = (t1_nifti2.get_data())
    t1_img2 = np.expand_dims( t1_nifti2.get_data(), axis=0 )

    gt_nifti = nib.load(os.path.join(case_paths, name+'_t1.nii.gz'))
    gt_img = np.expand_dims(gt_nifti.get_data(), axis=0)
    gt_img = nl.data.clip_percentile(gt_img, [0.0, 99.99])  # Clip to ignore bright extrema
    gt_img = nl.data.adjust_range(gt_img, [0.0, 1.0])  # Adjust range to 0-1 for Sigmoid activation

    gt_nifti2 = nib.load(os.path.join(case_paths, name+'_t1.nii.gz'))
    gt_img2 = np.expand_dims(gt_nifti2.get_data(), axis=0)
    return {'id': case_paths.split('/')[-1], 'nifti': t1_nifti, 'domain_1': ob,  'domain_2': gt_img,
            'domain_2_unprocessed': gt_img2, 'domain_1_unprocessed': t1_img2}

def splitdata(data,k ):
    train = []
    asd = data.copy()
    asd = sorted(asd,key=lambda x: int("".join([i for i in x if i.isdigit()])))
    num = len(data)
    if k ==0:
        a = asd[0:int(num/5)]
        # a = asd[0:3]
        test = a
        train = np.asarray( [i for i in data if i not in test] )
    elif k == 1:
        a = asd[int(num/5):2*int(num/5)]
        test = a
        train = np.asarray( [i for i in data if i not in test] )
    elif k == 2:
        a = asd[2*int(num/5):3*int(num/5)]
        test = a
        train = np.asarray( [i for i in data if i not in test] )
    elif k == 3:
        a = asd[3*int(num/5):4*int(num/5)]
        test = a
        train = np.asarray( [i for i in data if i not in test] )
    elif k == 4:
        a = asd[4*int(num/5):]
        test = a
        train = np.asarray( [i for i in data if i not in test] )
    return train, test
def get_datax(data,fold):
    asd = data.copy()
    asd = sorted(asd,key=lambda x: int("".join([i for i in x if i.isdigit()])))
    if fold == 0:
        a = asd[0:5]
        b = asd[20:25]
        c = asd[40:45]
        test = a+b+c

        train = np.asarray([i for i in data if i not in test])
    elif fold ==1:
        a = asd[5:10]
        b = asd[25:30]
        c = asd[45:50]
        test = a+b+c

        train = np.asarray( [i for i in data if i not in test] )
    elif fold == 2:
        a = asd[10:15]
        b = asd[30:35]
        c = asd[50:55]
        test = a+b+c

        train = np.asarray( [i for i in data if i not in test] )
    else:
        a = asd[15:20]
        b = asd[35:40]
        c = asd[55:]
        test = a+b+c

        train = np.asarray( [i for i in data if i not in test] )

    return train,test


def get_patch_gen(data, step, samples, normalize_opt= 'none'):
    #to change to 2D i need to change patch_shape to either (1,32,32) or (32,32,1)
    patch_set = nl.generator.ZipSet( [
        nl.generator.PatchSet(
            images=[case['domain_1'] for case in data],
            patch_shape=(32, 32, 32),
            normalize=normalize_opt,
            sampling=nl.generator.UniformSampling( (step, step, step), num_patches=samples * len( data ) ) ),
        nl.generator.PatchSet(
            images=[case['domain_2'] for case in data],
            patch_shape=(32, 32, 32),
            normalize=normalize_opt,
            sampling=nl.generator.UniformSampling( (step, step, step), num_patches=samples * len( data ) ) )] )
    return patch_set

def get_patch_gen_3input(data, step, samples, normalize_opt= 'none'):
    #to change to 2D i need to change patch_shape to either (1,32,32) or (32,32,1)
    patch_set = nl.generator.ZipSet( [
        nl.generator.PatchSet(
            images=[case['domain_1'] for case in data],
            patch_shape=(32,32,32),
            normalize=normalize_opt,
            sampling=nl.generator.UniformSampling( (step, step, step), num_patches=samples * len( data ))),#,
                                                   #masks=[case['mask'] for case in data]) ),
        nl.generator.PatchSet(
            images=[case['domain_2'] for case in data],
            patch_shape=(32,32,32),
            normalize=normalize_opt,
            sampling=nl.generator.UniformSampling( (step, step, step), num_patches=samples * len( data ))),# ,
                                                  # masks=[case['mask'] for case in data]) ),
        # nl.generator.PatchSet(
        #     images=[case['domain_1_single'] for case in data],
        #     patch_shape=(32,32,32),
        #     normalize=normalize_opt,
        #     sampling=pre.UniformSampling( (step, step, step), num_patches=samples * len( data ))),#,
                                                   #masks=[case['mask'] for case in data]) )
        # nl.generator.PatchSet(
        #     images=[case['lesion'] for case in data],
        #     patch_shape=(32, 32, 32),
        #     normalize=normalize_opt,
        #     sampling=pre.UniformSampling( (step, step, step), num_patches=samples * len( data ) ) )  # ,
        # # masks=[case['mask'] for case in data]) )
    ] )
    return patch_set

def _norm_patch(x):
    channel_means = np.mean(x, axis=(1, 2, 3), keepdims=True)
    channel_stds = np.std(x, axis=(1, 2, 3), keepdims=True)
    return np.divide(x - channel_means, channel_stds)

def _get_patch_slice(center, patch_shape):
    """
    :param center: a tuple or list of (x,y,z) tuples
    :param tuple patch_shape: (x,y,z) tuple with arr dimensions
    :return: a tuple (channel_slice, x_slice, y_slice, z_slice) or a list of them
    """
    if not isinstance(center, list): center = [center]
    span = [[int(np.ceil(dim / 2.0)), int(np.floor(dim / 2.0))] for dim in patch_shape]
    patch_slices = \
        [(slice(None),) + tuple(slice(cdim - s[0], cdim + s[1]) for cdim, s in zip(c, span)) for c in center]
    return patch_slices if len(patch_slices) > 1 else patch_slices[0]
class PatchSet(TorchDataset):
    """
    Creates a torch dataset that returns patches extracted from images either from predefined extraction centers or
    using a predefined patch sampling strategy.

    :param List[np.ndarray] images: list of images with shape (CH, X, Y, Z)
    :param PatchSampling sampling: An object of type PatchSampling defining the patch sampling strategy.
    :param normalize: one of ``'none'``, ``'patch'``, ``'image'``.
    :param dtype: the desired output data type (default: torch.float)
    :param List[List[tuple]] centers: (optional) a list containing a list of centers for each provided image.
        If provided it ignores the given sampling and directly uses the centers to extract the patches.

    :Example:

    # >>> images = [np.ones((1, 100,100,100)) for _ in range(10)]
    # >>> patch_set = PatchSet(
    # >>>     images, patch_shape=(2, 1, 1), sampling=UniformSampling(step=(10, 10, 10)), normalize='none')
    # >>> print(len(patch_set))
    # 10000
    # >>> print(patch_set[0])
    tensor([[[[1.]],
         [[1.]]]])
    """

    def  __init__(self, images, patch_shape, sampling, normalize, dtype=torch.float, centers=None):
        assert all([img.ndim == 4 for img in images]), 'Images must be numpy ndarrays with dimensions (C, X, Y, Z)'
        assert len(patch_shape) == 3, 'len({}) != 3'.format(patch_shape)
        assert normalize in ['none', 'patch', 'image']
        if centers is None: assert isinstance(sampling, nl.generator.PatchSampling)
        if centers is not None: assert len(centers) == len(images)

        self.images, self.dtype = images, dtype

        # Build all instructions according to centers and normalize
        self.instructions = []
        images_centers = sampling.sample_centers(images, patch_shape) if centers is None else centers

        for image_idx, image_centers in enumerate(images_centers):
            # Compute normalize function for this image's patches
            if normalize == 'patch':
                norm_func = _norm_patch
            elif normalize == 'image': # Update norm_func with the statistics of the image
                means = np.mean(self.images[image_idx], axis=(1,2,3), keepdims=True, dtype=np.float64)
                stds = np.std(self.images[image_idx], axis=(1,2,3), keepdims=True, dtype=np.float64)

                # BY PROVIDING MEANS AND STDS AS ARGUMENTS WITH DEFAULT VALUES, WE MAKE A COPY of their values inside
                # norm_func. If not, the means and stds would be of the last stored value (last image's statistics)
                # leading to incorrect results
                norm_func = lambda x, m=means, s=stds : (x - m) / s
            else:
                # Identity function (normalize == 'none')
                norm_func = lambda x : x

            ## Generate instructions
            self.instructions += [nl.generator.PatchInstruction(
                image_idx, center=center, shape=patch_shape, normalize_function=norm_func) for center in image_centers]

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, index):
        instr = self.instructions[index]
        x_patch = copy.deepcopy(self.images[instr.idx][_get_patch_slice(instr.center, instr.shape)])
        if instr.normalize_function is not None:
            x_patch = instr.normalize_function(x_patch)
        if instr.augment_function is not None:
            x_patch = instr.augment_function(x_patch)
        return torch.tensor(np.squeeze(np.ascontiguousarray(x_patch),axis=0), dtype=self.dtype)


def get_patch_gen_3input2d(data, step, samples, normalize_opt= 'none'):
    #to change to 2D i need to change patch_shape to either (1,32,32) or (32,32,1)
    patch_set = nl.generator.ZipSet( [
        PatchSet(
            images=[case['domain_1'] for case in data],
            patch_shape=(1,32, 32),  #(256,256,192) (256,256,172)
            normalize=normalize_opt,
            sampling=pre.UniformSampling( (step, step, step), num_patches=samples * len( data ))),#,
                                                   #masks=[case['mask'] for case in data]) ),
        PatchSet(
            images=[case['domain_2'] for case in data],
            patch_shape=(1,32, 32),
            normalize=normalize_opt,
            sampling=pre.UniformSampling( (step, step, step), num_patches=samples * len( data ))),# ,
                                                  # masks=[case['mask'] for case in data]) ),
        PatchSet(
            images=[case['domain_1_single'] for case in data],
            patch_shape=(1,32, 32),
            normalize=normalize_opt,
            sampling=pre.UniformSampling( (step, step, step), num_patches=samples * len( data ))),#,
                                                   #masks=[case['mask'] for case in data]) )
        # nl.generator.PatchSet(
        #     images=[case['lesion'] for case in data],
        #     patch_shape=(32, 32, 32),
        #     normalize=normalize_opt,
        #     sampling=pre.UniformSampling( (step, step, step), num_patches=samples * len( data ) ) )  # ,
        # # masks=[case['mask'] for case in data]) )
    ] )
    return patch_set


def get_patch_gen_3input2d2(data, step, samples, normalize_opt= 'none'):
    #to change to 2D i need to change patch_shape to either (1,32,32) or (32,32,1)


    patch_set = nl.generator.ZipSet( [
        PatchSet(
            images=[case['domain_1'] for case in data],
            patch_shape=(1,130, 130),  #(256,256,192) (256,256,172)
            normalize=normalize_opt,
            sampling=pre.UniformSampling( (step, step, step), num_patches=samples * len( data ))),#,
                                                   #masks=[case['mask'] for case in data]) ),
        PatchSet(
            images=[case['domain_2'] for case in data],
            patch_shape=(1,130, 130),
            normalize=normalize_opt,
            sampling=pre.UniformSampling( (step, step, step), num_patches=samples * len( data ))),# ,
                                                  # masks=[case['mask'] for case in data]) ),
        PatchSet(
            images=[case['domain_1_single'] for case in data],
            patch_shape=(1,130, 130),
            normalize=normalize_opt,
            sampling=pre.UniformSampling( (step, step, step), num_patches=samples * len( data ))),#,
                                                   #masks=[case['mask'] for case in data]) )
        # nl.generator.PatchSet(
        #     images=[case['lesion'] for case in data],
        #     patch_shape=(32, 32, 32),
        #     normalize=normalize_opt,
        #     sampling=pre.UniformSampling( (step, step, step), num_patches=samples * len( data ) ) )  # ,
        # # masks=[case['mask'] for case in data]) )
    ] )
    return patch_set

