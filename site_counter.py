# import modules for handling files
from pathlib import Path
from sys import argv
import csv

# import numpy and skimage modules
import numpy as np
from skimage import filters, morphology  # import filters
#from skimage import morphology
from skimage.measure import label, regionprops
from skimage.exposure import rescale_intensity
from skimage import img_as_float, img_as_uint, img_as_ubyte
import skimage.external.tifffile as tiff
import skimage.io as io


def subtract_median(im, radius):
    # median filtering function
    im_spots = np.zeros(shape=im.shape, dtype=im.dtype)

    for i in range(im.shape[0]):
        im_median = filters.median(im[i, :, :], morphology.disk(radius))
        im_spots[i, :, :] = img_as_uint(
            img_as_float(im[i, :, :]) - img_as_float(im_median))

    # when you're bored, figure out why this does not work
    # for i in range( im.shape[ 0 ] ) :
    #         im_median = filters.median( im[ i , : , : ] , morphology.disk( radius ) )
    #         im_spots[ i , : , : ] = im[ i , : , : ] - im_median

    return im_spots


def erosion(image, n):

    if n == 0:
        n = 1
        print("n set to 1; eroding pixels with no neighbor makes no sense")
    if n > 26:
        n = 26
        print("n set to 26; number of neighbor pixels cannot exceed 26")

    brush = np.array([
        [  # 0
            [
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
        ],
        [  # 1
            [
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
        ],
        [  # 2
            [
                [0, 0, 1],
                [0, 0, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
        ],
        [  # 3
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
        ],
        [  # 4
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
        ],
        [  # 5
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
        ],
        [  # 6
            [
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0]],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
        ],
        [  # 7
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 1, 0]],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
        ],
        [  # 8
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1]],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
        ],
        [  # 9
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
        ],
        [  # 10
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
            [
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
        ],
        [  # 11
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
            [
                [0, 0, 1],
                [0, 1, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
        ],
        [  # 12
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 1, 1],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
        ],
        [  # 13
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1]],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
        ],
        [  # 14
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 1, 0]],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
        ],
        [  # 15
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 0]],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
        ],
        [  # 16
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [1, 1, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
        ],
        [  # 17
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]],
            [
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
        ],
        [  # 18
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]],
            [
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 0]],
        ],
        [  # 19
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]],
            [
                [0, 0, 1],
                [0, 0, 0],
                [0, 0, 0]],
        ],
        [  # 20
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 0, 0]],
        ],
        [  # 21
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]],
        ],
        [  # 22
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 0]],
        ],
        [  # 23
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0]],
        ],
        [  # 24
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 1, 0]],
        ],
        [  # 25
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1]],
        ]
    ])

    eroded_images = np.zeros(shape=image.shape, dtype=image.dtype)

    for j in range(26):
        tmp_image = morphology.binary_erosion(image, brush[j, :, :, :])
        eroded_images = eroded_images + tmp_image
        image_output = img_as_ubyte(eroded_images > 26 - n)

    return image_output


def mask_patches(image, threshold_value=None):
        # make a mask from the image image
    if threshold_value == None:
        threshold_value = filters.threshold_yen(image)

        print(threshold_value)
    # for debug threshold_value = 16000
    print('Yen threshold: 'i + str(threshold_value))
    image_threshold = np.zeros(shape=image.shape, dtype=image.dtype)
    image_threshold[image > threshold_value] = 1

    #image_eroded = erosion( image_threshold , 21 )
    image_eroded = image

    return image_eroded


def cell_area(im, radius=10):
    # initialize empty image
    im_median = np.zeros(shape=im.shape, dtype=im.dtype)
    # fill empty image with median-filtered slices
    for i in range(im.shape[0]):
        im_median[i, :, :] = filters.median(
            im[i, :, :], morphology.disk(radius))
    # maximum project, threshold (otsu) and label binary image
    im_median_max = max_project(im_median)
    im_median_max_threshold = filters.threshold_otsu(im_median_max)
    im_median_max_binary = img_as_ubyte(
        im_median_max > im_median_max_threshold)
    im_labeled = label(im_median_max_binary, return_num=False)
    # measure and return area
    props = regionprops(im_labeled)
    area = props[0].area
    return area


def max_project(im):
    if im.ndim == 3:
        im_max = np.amax(im, 0)
        return im_max
    else:
        print("Error: 3-dimensional stack required")
        return None


def count_patches(path, GFP_pattern='*GFP*', median_radius=10, erosion_n=21, con=2, method='yen'):

        # initialize paths: in/out dirs and output file for numbers
        # using pathlib/Path makes it easier to create folders an manipulate paths than os

    inPath = Path(path)
    outPath = inPath.joinpath(method + str(median_radius) + 'r'
                              + str(erosion_n) + 'n' + str(con) + 'con')
    outPath.mkdir(parents=True, exist_ok=True)
    outCsv = outPath.joinpath("count.csv")

    with outCsv.open('w', newline='') as f:  # initialize a csv file for writing

        # initialize csv writer and write headers
        writer = csv.writer(f, dialect='excel')
        writer.writerow(['Cell', 'Threshold', 'Patches', 'Cross_Area'])

        # iterate over files
        for i in inPath.glob(GFP_pattern):  # glob returns pattern-matching files

            # get the name of image i to modify later
            im_path = outPath.joinpath(i.name)

            # read image
            im = tiff.imread(str(i))

            # remove background with median filtering and save MD image
            im_spots = subtract_median(im, median_radius)
            tiff.imsave(str(im_path).replace(GFP_pattern, '').replace(
                '.tif', '_MD.tif'), img_as_ubyte(rescale_intensity(im_spots, out_range='uint8')))

            # threshold and save image

            if method == 'yen':
                threshold_value = filters.threshold_yen(im_spots)
            elif method == 'otsu':
                threshold_value = filters.threshold_otsu(im_spots)
            else:
                print("method not specified, choose 'yen' or 'otsu'")
            #im_threshold = np.zeros(shape = im.shape , dtype=im.dtype)
            #im_threshold[ im_spots > threshold_value ] = 1

            # threshold
            # bool array doesn't quite work with ImageJ
            im_threshold = img_as_ubyte(im_spots > threshold_value)
            #print( 'Threshold value: '+str(threshold_value) )

            # save
            tiff.imsave(str(im_path).replace(GFP_pattern, '').replace(
                '.tif', '_Thresholded_' + method + '.tif'), im_threshold)

            # erode and save image

            # initialize values for a while loop
            im_check = np.ones(shape=im_threshold.shape,
                               dtype=im_threshold.dtype)
            im_eroded = im_threshold
            loop = 0

            # loop erosion as long as the image is changing
            while np.max(img_as_float(im_check) - img_as_float(im_eroded)) > 0:
                im_check = im_eroded
                im_eroded = erosion(im_eroded, erosion_n)
                loop += 1
                # print('loop number', loop)

            # use label to get eroded image with patch labels and count with total number
            im_eroded, count = label(
                im_eroded, connectivity=con, return_num=True)

            # get out area
            area = cell_area(im)

            # save eroded image and the counts with csv writer
            writer.writerow([i.name.replace('.tif', ''),
                             method, str(count), area])
            tiff.imsave(str(im_path).replace(GFP_pattern, '').replace(
                '.tif', '_Eroded' + '_n' + str(erosion_n) + '.tif'), img_as_ubyte(im_eroded))


# get the path from command line and run counting function
path = argv[1]
count_patches(path, median_radius=5, erosion_n=26, con=2)


# old inputs, maybe will be useful at some point but probably not really?

# mateusz_pathlib ("/Volumes/MarkoKaksonenLab/Mateusz/microscopy/Ede1_mutants_internal/ede1_null/20170831_MKY0654/raw/stk2tif/Z/cells/unbudded", median_radius = 5, erosion_n = 20, con = 2 )

# mateusz_pathlib ("/Volumes/MarkoKaksonenLab/Mateusz/microscopy/Ede1_mutants_internal/wt/20170829_MKY0140/raw/stk2tif/Z/cells/unbudded", median_radius = 5, erosion_n = 20, con = 2 )

# for i in range(10, 13):

#    mateusz_pathlib("/Volumes/MarkoKaksonenLab/Mateusz/microscopy/Ede1_mutants_internal/wt/20170829_MKY0140/raw/stk2tif/Z/cells/unbudded/", median_radius = 5, erosion_n = i, con = 1)
