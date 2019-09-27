from im_tools import *

# import modules for handling files
from pathlib import Path
from sys import argv
import csv


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
                im_eroded = erode_3d(im_eroded, erosion_n)
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
if __name__ == "__main__": # only executed if ran as script
    path = argv[1]
    count_patches(path, median_radius=5, erosion_n=26, con=2)