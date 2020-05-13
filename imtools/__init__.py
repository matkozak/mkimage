# import numpy and skimage modules
import numpy as np
from skimage import filters, morphology  # import filters
#from skimage import morphology
from skimage.measure import label, regionprops
from skimage.exposure import rescale_intensity
from skimage import img_as_float, img_as_uint, img_as_ubyte
import skimage.external.tifffile as tiff
import skimage.io as io