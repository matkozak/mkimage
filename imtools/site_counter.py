# import modules for handling files
import csv
from pathlib import Path
from sys import argv

# import our tools for image manipulation
from im_tools import *


def count_patches(path, GFP_pattern='*GFP*', median_radius=10, erosion_n=21, con=2, method='yen'):

    # set up a dictionary for thresholding methods
    thresholding_methods = dict(
        li = filters.threshold_li,
        otsu = filters.threshold_otsu,
        triangle = filters.threshold_triangle,
        yen = filters.threshold_yen
    )

    # check if the method is going to work
    if method not in thresholding_methods.keys():
        print('Specified thresholding method not valid. Choose one of:')
        print(*thresholding_methods.keys(), sep = '\n')
        return

    # initialize paths: in/out dirs and output file for numbers
    # using pathlib/Path makes it easier to create folders an manipulate paths than os

    inPath = Path(path)
    outPath = inPath.joinpath(method + str(median_radius) + 'r'
                              + str(erosion_n) + 'n' + str(con) + 'con')
    outPath.mkdir(parents=True, exist_ok=True)
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
                '.tif', '_MD.tif'), img_as_ubyte(rescale_intensity(im_spots, out_range='uint8')))

            # threshold and save image
            
            # this method from im_tools uses slice histograms
            # consequently the bleaching in lower slices does not affect result
            # unfortunately it generates a lot of salt if there is a slice without a patch
            #im_threshold = threshold_slices(im_spots, method)

            # normal thresholding from filters uses stack histograms
            # we use methods specified in the dictionary at the top
            threshold_value = thresholding_methods[method](im_spots)

            # threshold
            # bool array doesn't quite work with ImageJ, hence img_as_ubyte
            im_threshold = im_spots > threshold_value
            #print( 'Threshold value: '+str(threshold_value) )

            # save
            tiff.imsave(str(im_path).replace(GFP_pattern, '').replace(
                '.tif', '_Thresholded_' + method + '.tif'), img_as_ubyte(im_threshold))

            # erode and save image

            # pad the image so that eroding can work the edges
            im_eroded = np.pad(im_threshold, 1)
            
            # initialize values for a while loop
            im_check = np.ones(shape=im_eroded.shape,
                               dtype=im_eroded.dtype)
            loop = 0

            # loop erosion as long as the image is changing
            while np.max(img_as_float(im_check) - img_as_float(im_eroded)) > 0:
                im_check = im_eroded.copy()
                im_eroded = erode_alternative(im_eroded, erosion_n)
                loop += 1
                print('loop number', loop)

            # un-pad the image
            im_eroded = im_eroded[1:-1, 1:-1, 1:-1]
            
            # use label to get eroded image with patch labels and count with total number
            im_eroded, count = label(
                im_eroded, connectivity=con, return_num=True)

            # get out area
            area = cell_area(im)

            # save eroded image and the counts with csv writer
            writer.writerow([i.name.replace('.tif', ''),
                             method, str(count), area])
            tiff.imsave(str(im_path).replace(GFP_pattern, '').replace(
                '.tif', '_Eroded' + '_n' + str(erosion_n) + '.tif'), img_as_uint(im_eroded))

# get the path from command line and run counting function
if __name__ == "__main__": # only executed if ran as script
    path = argv[1]
    method = str(argv[2])
    n = int(argv[3])
    count_patches(path, median_radius = 5, erosion_n = n, con = 2, method = method)
