# import numpy and skimage modules
import numpy as np
import skimage as sk
import tifffile as tiff


def max_project(im):
    """ Return a maximum Z-projection of a 3D image. """
    if im.ndim == 3:
        im_max = np.amax(im, 0)
        return im_max
    else:
        print("Error: 3-dimensional stack required")
        return None


def median_filter(im, radius):
    """
    Median filter a 2D/3D image using a circular brush of given radius.
    On a 3D image, each slice is median-filtered separately using a 2D structuring element.
    """
    if len(im.shape) == 2:
        im_median = sk.filters.median(im, sk.morphology.disk(radius))
    elif len(im.shape) == 3:
        # initialize empty image
        im_median = np.zeros(shape=im.shape, dtype=im.dtype)
        # fill empty image with median-filtered slices
        for i in range(im.shape[0]):
            im_median[i, :, :] = sk.filters.median(
                im[i, :, :], sk.morphology.disk(radius))
    else:
        print('Cannot deal with the supplied number of dimensions.')
        return None
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
    threshold = sk.filters.threshold_otsu(im_median)
    im_mask = im_median > threshold
    # return masked image
    return im_mask


def subtract_median(im, radius):
    """ Performs median filtering and subtracts the result from original image. """
    im_median = median_filter(im, radius)
    # microscope .tif files are uint16 so subtracting below 0 causes integer overflow
    # for now using np method to cast to int64, skimage function goes back to image
    im_spots = im.astype(int) - im_median.astype(int)
    return sk.img_as_uint(im_spots)


def rescale_to_float(a):
    """ Rescale values of a numpy array to span [0,1]. """
    b = (a - np.min(a)) / np.ptp(a)
    return b


def cell_area(im, radius=10):
    """ Return pixel area estimate of cell cross section.
    Only one ROI per image is counted so thresholding has to be unambiguous. """
    im_mask = mask_cell(im, max=True)
    area = np.sum(im_mask)
    return area


def erode_3d(im, n):

    # process input image: binarize and pad
    im = (im > 0.5).astype(int)
    im = np.pad(im, 1)

    im_out = im.copy()
    index = np.argwhere(im)

    for i in index:
        z, y, x = i[0], i[1], i[2]
        # calculate the sum of the cube around nonzero pixel
        cube_sum = np.sum(im[z-1:z+2, y-1:y+2, x-1:x+2]) - 1
        # zero pixels below threshold connections
        if cube_sum < n:
            im_out[tuple(i)] = 0

    im_out = im_out[1:-1, 1:-1, 1:-1].astype(int)
    return im_out


def erode_andrea(image, n):
    """
    Performs a three dimensional erosion on binary image. The 3D brush represents all
    possible positions in a cubic array around the eroded pixel while the n parameter
    specifies how many connections a pixel needs to have to be preserved.
    I.e.: n = 26 means that a pixel is eroded unless it is completely surrounded by 1's,
    n = 1 means that the pixel is preserved as long as it has 1 neighbour in 3D.
    """

    if n == 0:
        n = 1
        print("n set to 1; smaller values will not do anything")
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

    image = np.pad(image, 1)
    eroded_images = np.zeros(shape=image.shape, dtype=image.dtype)

    for i in range(26):
        tmp = sk.morphology.binary_erosion(image, brush[i, :, :, :])
        eroded_images += tmp

    image_out = eroded_images >= n
    image_out = image_out[1:-1, 1:-1, 1:-1]

    return image_out

