import os
import numpy as np
import nibabel as nib
import cv2 as cv
import pdb
import matplotlib.pyplot as plt
def image_write(path_A, path_B, path_AB):
    im_A = cv.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_B = cv.imread(path_B, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    # print(im_A.shape)
    # print(im_B.shape)
    # pdb.set_trace()
    im_AB = np.concatenate([im_A, im_B], 1)
    cv.imwrite(path_AB, im_AB)

def adjust_range(img, new_range, old_range=None):
    """Scales the intensity range of the given array to the new range.

    :param img: the input array.
    :param new_range: list or tuple with the minimum and maximum values of the output.
    :param old_range: (optional) list or tuple with the range of the input.
        If not given then the range is computed from the minimum and maximum value of the input.
    :return: the image with intensities in the new range.

    :Example:

    # >>> adjust_range([3, 4, 5], new_range=[0, 10])
    array([ 0., 5.0, 10.])
    # >>> adjust_range([3, 4, 5], new_range=[0, 10], old_range=[0, 100])
    array([ 0.3, 0.4, 0.5])
    """
    assert new_range[0] < new_range[1], 'new_range is not correctly defined'

    new_low, new_high = float( new_range[0] ), float( new_range[1] )
    if old_range is None:
        old_low, old_high = np.min( img ), np.max( img )
        if old_low == old_high:
            raise ValueError( 'Given array is of constant value, range cannot be adjusted!' )
    else:
        old_low, old_high = float( old_range[0] ), float( old_range[1] )

    if not isinstance( img, np.ndarray ):
        img = np.asanyarray( img )

    norm_image = (img - old_low) / (old_high - old_low)  # Put between 0 and 1
    norm_image = new_low + ((new_high - new_low) * norm_image)  # Put in new range
    return norm_image


def clip_percentile(img, percentile, ignore_zeros=False):
    """Clips image values according to the given percentile

    :param img:
    :param percentile: list (low_percentile, high_percentile) where percentile is a number from 0 to 100.
    :param ignore_zeros: (optional) if True, ignores the zero values in the computation of the percentile.
    :return: the clipped image

    :Example:

    # >>> clip_percentile([-1000, -100, -10, 0, 10, 100, 1000], [10, 90])
    array([-460., -100.,  -10.,    0.,   10.,  100.,  460.])
    # >>> clip_percentile([-1000, -100, -10, 0, 10, 100, 1000], [25, 75])
    array([-55., -55., -10.,   0.,  10.,  55.,  55.])

    """
    if not isinstance( img, np.ndarray ):
        img = np.asanyarray( img )

    img_flat = img[np.nonzero( img )] if ignore_zeros else img
    low, high = np.percentile( img_flat, percentile )
    return np.clip( img, low, high )


def load_and_divide(case_paths, patient, typ='3DT1_brain.nii.gz', nam='T1'):
    # Dictionary containing at first the img and the gt
    t1_nifti = nib.load( os.path.join( case_paths, typ ) )
    t1_img = (t1_nifti.get_fdata())
    t1_img = clip_percentile( t1_img, [0.0, 99.99] )  # Clip to ignore bright extrema
    t1_img = adjust_range( t1_img, [0.0, 255.0] )
    #
    t1 = t1_img
    #
    t1_path = os.path.join( patient, nam )
    os.chdir( patient )
    if not os.path.isfile( t1_path ):

        os.mkdir( t1_path )
    os.chdir( t1_path )
    for i in range( t1.shape[2] ):
        tmp = t1[:, :, i]
        #        if np.sum(tmp>0) >0:
        name = 'slice_' + str( i )
        cv.imwrite( name + '.png', tmp )


    print( 'Image_{}_ {} Done!'.format( case_paths, nam ) )
    os.chdir( patient )

def get_datax(data, fold):
    asd = data.copy()
    asd = sorted( asd, key=lambda x: int( "".join( [i for i in x if i.isdigit()] ) ) )
    if fold == 0:
        a = asd[0:5]
        b = asd[20:25]
        c = asd[40:45]
        test = a + b + c

        train = np.asarray( [i for i in data if i not in test] )
        train = train.tolist()
    elif fold == 1:
        a = asd[5:10]
        b = asd[25:30]
        c = asd[45:50]
        test = a + b + c

        train = np.asarray( [i for i in data if i not in test] )
        train = train.tolist()
    elif fold == 2:
        a = asd[10:15]
        b = asd[30:35]
        c = asd[50:55]
        test = a + b + c

        train = np.asarray( [i for i in data if i not in test] )
        train = train.tolist()
    else:
        a = asd[15:20]
        b = asd[35:40]
        c = asd[55:]
        test = a + b + c

        train = np.asarray( [i for i in data if i not in test] )
        train = train.tolist()

    return train, test


def get_fold(case, out):
    name = case[35:]
    fold = os.path.join( out, name )
    if not os.path.isfile( fold ):
        os.mkdir( os.path.join( out, name ) )
    return fold


dataset = "/home/pierpaolo/Datasets/WMH20173D"  # 42 elements
out_dataset = "/home/pierpaolo/Datasets/WMH20172D_pix2pix"
case_path = [f.path for f in os.scandir( dataset ) if f.is_dir()]
folds = [0, 1, 2, 3]
# for k in folds:
#     train, test = get_datax( case_path.copy(), k )
#     data_out = os.path.join( out_dataset, str( k ) )
#
#     os.mkdir( os.path.join( data_out, 'train' ) )
#     os.mkdir( os.path.join( data_out, 'test' ) )
#     train_path = os.path.join( data_out, 'train' )
#     test_path = os.path.join( data_out, 'test' )
#     for i in range( len( train ) ):
#         patient = get_fold( train[i], train_path )
#         load_and_divide( train[i], patient, '3DFLAIR_brain.nii.gz', nam='FLAIR' )
#         load_and_divide( train[i], patient, '3DT1_brain.nii.gz', nam='T1' )
#         os.chdir( out_dataset )
#     for i in range( len( test ) ):
#         patient = get_fold( test[i], test_path )
#         load_and_divide( test[i], patient, '3DFLAIR_brain.nii.gz', nam='FLAIR' )
#         load_and_divide( test[i], patient, '3DT1_brain.nii.gz', nam='T1' )
#         os.chdir( out_dataset )

for k in folds:
    train, test = get_datax( case_path.copy(), k )
    data_out = os.path.join( out_dataset, str( k ) )
    train_path = os.path.join( data_out, 'train' )
    test_path = os.path.join( data_out, 'test' )

    for d in os.listdir( train_path ):
        patient = os.path.join( train_path, d )
        try:
            os.mkdir( os.path.join( patient, 'Concatenate' ))
        except:
            pass
        c = os.path.join( patient, 'Concatenate' )
        t1s, flairs = [], []
        for a in os.listdir( patient ):
            img_a = os.path.join( patient, 'FLAIR' )
            img_b = os.path.join( patient, 'T1' )
        for dirName, subdirList, fileList in os.walk( img_a ):
            for filename in fileList:
                if '.png' in filename.lower():
                    flairs.append( filename )
                    # flairs = flairs[0][2][:-2]
        for dirName, subdirList, fileList in os.walk( img_b ):
            for filename in fileList:
                if'.png' in filename.lower():
                    t1s.append( filename )
                    # t1s = t1s[0][2][:-2]
        number = d
        for i in t1s:
            print('Concatenating ', i)
            name = 'concatenate_' + d + i+'.png'
            patta = os.path.join( img_a, i )
            pattb = os.path.join( img_b, i )
            image_write(patta,pattb,name)

    for d in os.listdir( test_path ):
        patient = os.path.join( test_path, d )
        os.mkdir( os.path.join( patient, 'Concatenate' ))
        c = os.path.join(patient, 'Concatenate')
        t1s, flairs = [], []
        for a in os.listdir( patient ):
            img_a = os.path.join( patient, os.listdir( patient )[1] )
        img_b = os.path.join( patient, os.listdir( patient )[0] )
        for dirName, subdirList, fileList in os.walk( img_a ):
            for filename in fileList:
                if '.png' in filename.lower():
                    flairs.append( filename )
                    # flairs = flairs[0][2][:-2]
        for dirName, subdirList, fileList in os.walk( img_b ):
            for filename in fileList:
                if'.png' in filename.lower():
                    t1s.append( filename )
                    # t1s = t1s[0][2][:-2]
        number = d
        for i in t1s:
            a = os.path.join( img_a, i )
            b = os.path.join( img_b, i )
            image_write(a,b, c)
