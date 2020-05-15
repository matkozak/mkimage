# import modules for handling files
import csv
from pathlib import Path
from sys import argv

# import third-party packages
import numpy as np
import skimage as sk
import tifffile as tiff
from skimage.measure import label
from skimage.exposure import rescale_intensity

# import utility functions
from imtools.utility import *

#note to self: abstract the counting function from the writing function

def count_patches(path, GFP_pattern='*GFP*', median_radius=10, erosion_n=3, con=2, method='yen', mask = False, loop = False):

    # initialize paths: in/out dirs and output file for numbers
    # using pathlib/Path makes it easier to create folders an manipulate paths than os

    inPath = Path(path)
    outPath = inPath.joinpath(
        method + '_r' + str(median_radius) + '_n' + str(erosion_n) + '_con' + str(con) + '_mask' * mask + '_loop' * loop)
    outPath.mkdir(parents=True, exist_ok=True)
    # delete this at some point, it's a helper for naming but it's not 
    #experiment = next(i for i in inPath.parts if 'MKY' in i)
    #outCsv = outPath.joinpath(experiment + '_' + method + '_n' + str(erosion_n) + "_count.csv")
    outCsv = outPath.joinpath(method + '_n' + str(erosion_n) + "_count.csv")


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
                '.tif', '_MD.tif'), sk.img_as_ubyte(rescale_intensity(im_spots, out_range='uint8')))

            # threshold and save image
            # filters.threshold methods use stack histograms, not slice histograms
            # we use methods specified in the dictionary at the top
            if mask:
                im_masked = im_spots[mask_cell(im)]
                threshold_value = threshold(im_masked, method)
            else:
                threshold_value = threshold(im_spots, method)

            # threshold
            im_threshold = im_spots > threshold_value
            #print( 'Threshold value: '+str(threshold_value) )

            # save
            # bool array doesn't quite work with ImageJ, hence sk.img_as_ubyte
            tiff.imsave(str(im_path).replace(GFP_pattern, '').replace(
                '.tif', '_Thresholded_' + method + '.tif'), sk.img_as_ubyte(im_threshold))

            # erode and save image

            # pad the image so that eroding can work the edges
            im_eroded = im_threshold.copy()
            
            # initialize values for a while loop
            # if loop is
            if loop:
                im_check = np.ones(shape=im_eroded.shape,
                                   dtype=im_eroded.dtype)
                # loop erosion as long as the image is changing
                while not np.array_equal(im_check, im_eroded):
                    im_check = im_eroded.copy()
                    im_eroded = erode_3d(im_eroded, erosion_n)
                    #loop += 1
                    #print('loop number', loop)
            else:
                im_eroded = erode_3d(im_eroded, erosion_n)
            
            # use label to get eroded image with patch labels and count with total number
            im_eroded, count = label(
                im_eroded, connectivity=con, return_num=True)

            # get out area
            area = cell_area(im)

            # save eroded image and the counts with csv writer
            writer.writerow([i.name.replace('.tif', ''),
                             method, str(count), area])
            tiff.imsave(str(im_path).replace(GFP_pattern, '').replace(
                '.tif', '_Eroded' + '_n' + str(erosion_n) + '.tif'), sk.img_as_uint(im_eroded))

# get the path from command line and run counting function
if __name__ == "__main__": # only executed if ran as script
    path = argv[1]
    method = str(argv[2])
    n = int(argv[3])
    count_patches(path, median_radius = 5, erosion_n = n, con = 2, method = method, mask = True, loop = False)
