# import numpy and skimage modules
import numpy as np
from skimage import filters, morphology  # import filters
#from skimage import morphology
from skimage.measure import label, regionprops
from skimage.exposure import rescale_intensity
from skimage import img_as_float, img_as_uint, img_as_ubyte
import skimage.external.tifffile as tiff
import skimage.io as io


def max_project(im):
    """ Return a maximum Z-projection of a 3D image. """
    if im.ndim == 3:
        im_max = np.amax(im, 0)
        return im_max
    else:
        print("Error: 3-dimensional stack required")
        return None


def median_filter(im, radius):
    """ Median filter a 3D image using brush of given radius. """
    # initialize empty image
    im_median = np.zeros(shape=im.shape, dtype=im.dtype)
    # fill empty image with median-filtered slices
    for i in range(im.shape[0]):
        im_median[i, :, :] = filters.median(
            im[i, :, :], morphology.disk(radius))
    return im_median


def mask_cell(im, radius=10, max=False):
    """
    Return a mask based on thresholded median filtered image.
    To apply mask: im[mask] produces a flat array of masked values. im * mask gives a masked image.
    """
    im_median = median_filter(im, radius)
    # maximum project
    if max:
        im_median = max_project(im_median)
    # threshold (otsu)
    threshold = filters.threshold_li(im_median)
    im_mask = im_median > threshold
    # return masked image
    return im_mask


def subtract_median(im, radius):
    """ Performs median filtering and subtracts the result from original image. """
    im_median = median_filter(im, radius)
    # microscope .tif files are uint16 so subtracting below 0 causes integer overflow
    # casting from to int and back will cause doubling of the positive results
    # float conversion is the best option as far as I know
    im_spots = img_as_uint(img_as_float(im) - img_as_float(im_median))
    return im_spots


def rescale_to_float(a):
    """ Rescale values of a numpy array to span [0,1]. """
    b = (a - np.min(a)) / np.ptp(a)
    return b


def cell_area(im, radius=10):
    """ Return pixel area estimate of cell cross section.
    Only one ROI per image is counted so thresholding has to be unambiguous. """
    im_mask = mask_cell(im, max=True)
    # im_labeled = label(im_mask, return_num=False) might not even be necessary lmao
    # haven't decided whether to completely get rid of it but actually there is no reason to use label()
    #props = regionprops(im_labeled)
    #area = props[0].area
    area = np.sum(im_mask)
    return area


def erode_3d(image, n):
    """
    Performs a three dimensional erosion on binary image. The 3D brush represents all
    possible positions in a cubic array around the eroded pixel while the n parameter
    specifies how many connections a pixel needs to have to be preserved.
    I.e.: n = 1 means that a pixel is eroded unless it is completely surrounded by 1's,
    n = 26 means that the pixel is preserved as long as it has 1 neighbour in 3D.
    """

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

    for i in range(26):
        tmp = morphology.binary_erosion(image, brush[i, :, :, :])
        eroded_images = eroded_images + tmp
        image_output = img_as_ubyte(eroded_images > 26 - n)

    return image_output
