import nibabel as nib
import os
import numpy as np
import niclib as nl

def get_dataset():
    """
    Simple function for chosing the dataset
    :return: 0 (WMH) , 1(BraTS)
    """
    dset = None
    while dset == None:
        dset = int( input( 'Which dataset ? \n'
                           '0 = WMH\n'
                           '1 = Brats2018\n' ) )
        if dset not in [0, 1]:
            dset = None
            print( 'repeat' )
    return dset

def preparedata():
    """Initializes the directories of the whole project with the main folders.
    :returns main folder of the experiment with subfolders."""
    day = nl.get_timestamp(time_format='%d_%m_%Y')
    data = get_dataset()
    if data == 1:
        challenge = 'Brats2018'
        data_path = '/home/pierpaolo/Datasets/Brats18TrainingData'

    elif data == 0:
        challenge = 'WMH20173D'
        data_path = '/home/pierpaolo/Datasets/WMH20173D'
    else:
        return

    mainfolder = nl.make_dir(day + '_' + str(challenge))
    checkpoints_path = nl.make_dir(os.path.join(mainfolder, 'checkpoints/'))
    results_path = nl.make_dir(os.path.join(mainfolder, 'results/'))
    metrics_path = nl.make_dir(os.path.join(mainfolder, 'metrics/'))
    log_path = nl.make_dir(os.path.join(mainfolder, 'log/'))

    return data_path,checkpoints_path, results_path, metrics_path, log_path, mainfolder

def load_case_whm(case_paths):
    """
    White Matter Hyperintensities challenge dataset.
    Returns a dictionary containing the loaded and preprocessed dataset.
    Each element of the dictionary is composed by ID, Domain_1 image,
    Domain_2 image, Domain_1 image unprocessed, Domain_2 image unprocessed."""
    # Dictionary containing at first the img and the gt
    t1_nifti = nib.load(os.path.join(case_paths, '3DFLAIR_brain.nii.gz'))
    t1_img = (t1_nifti.get_data())
    t1_img = nl.data.clip_percentile(t1_img, [0.0, 99.99])  # Clip to ignore bright extrema
    t1_img = nl.data.adjust_range(t1_img, [0.0, 1.0])  # Adjust range to 0-1 for Sigmoid activation
    t1_img = np.expand_dims( t1_nifti.get_data(), axis=0 )  # Add single channel modality

    les = nib.load(os.path.join(case_paths, '3DWMH_c1.nii.gz'))
    lesi = np.expand_dims(les.get_data(), axis=0)

    ob = np.vstack((t1_img,lesi))

    gt_nifti = nib.load(os.path.join(case_paths, '3DT1_brain.nii.gz'))
    gt_img = np.expand_dims(gt_nifti.get_data(), axis=0)
    gt_img = nl.data.clip_percentile(gt_img, [0.0, 99.99])  # Clip to ignore bright extrema
    gt_img = nl.data.adjust_range(gt_img, [0.0, 1.0])  # Adjust range to 0-1 for Sigmoid activation

    gt_nifti2 = nib.load(os.path.join(case_paths, '3DT1_brain.nii.gz'))
    gt_match = np.expand_dims(gt_nifti2.get_data(), axis=0)
    # gt_img =gt_nifti.get_data()
    gt_nifti3 = nib.load(os.path.join(case_paths, '3DFLAIR_brain.nii.gz'))
    t1_match = np.expand_dims(gt_nifti3.get_data(), axis=0)


    return {'id': case_paths.split('/')[-1], 'nifti': t1_nifti, 'domain_1': ob,  'domain_2': gt_img,
            'domain_2_unprocessed': gt_match, 'domain_1_unprocessed': t1_match}

def load_case_brats(case_paths):
    """
       Brain Tissue Segmentation challenge dataset.
       Returns a dictionary containing the loaded and preprocessed dataset.
       Each element of the dictionary is composed by ID, Domain_1 image,
       Domain_2 image, Domain_1 image unprocessed, Domain_2 image unprocessed."""
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
    """
    Split the BraTS into Train and Test
    :param data: datapath of the dataset
    :param k: current fold
    :return: Train, test lists
    """
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
    """
    Splits WMH Dataset into Train and Test lists
    :param data: The datapath of the dataset
    :param fold: the current fold
    :return: Train, test lists
    """
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
    """
    Split the Train (or test) images into Patches of size 32x32x32
    :param data: Dataset
    :param step: step to do in a given direction from patch to patch
    :param samples: number of samples we ant per volume
    :param normalize_opt: option for applying a normalization to the image ('image')
                        is an option
    :return: set containing the (num_images* samples) patches
    """
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