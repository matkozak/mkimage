from im_tools import *

# import modules for handling files
from pathlib import Path
from sys import argv


def batch_mask(path, pattern="*GFP*", rescale=False, return_dict=False, save_arrays=False):
    """
    This function reads all images with a keyword ('GFP' by default) and applies a 3D masking procedure.
    You need to explicitly tell the function whether it should return a dictionary with all pixel values or save each array as a .txt file.
    Rescale option forces each array into [0,1] range.
    """
    # path handling through Pathlib: make output folder within current path
    path_in = Path(path)
    # initialise a dictionary to store results
    pixels = {}

    # actual function: loop over each file with pattern, mask and convert to array
    for i in path_in.glob(pattern):
        im = tiff.imread(str(i))
        # filter out saturated images
        if 65535 in im:
            continue

        im = im[mask_cell(im)]  # mask and select values
        # optional rescaling
        if rescale:
            im = rescale_to_float(im)
        # add dictionary entry with name (no extension) and pixel values
        pixels[i.name.replace('.tif', '')] = im
        # add image name to output folder path

    # output: save each dictionary entry as separate file in a subfolder
    if save_arrays:
        path_out = path_in.joinpath('masked_arrays')  # prepare output path
        f = '%i'  # not quite necessary but the default 18-digit precision means relatively huge files
        if rescale:
            # save rescaled in a subfolder
            path_out = path_out.joinpath('rescaled')
            f = '%f'  # also change format to float
        path_out.mkdir(parents=True, exist_ok=True)
        # save array
        for key, value in pixels.items():
            np.savetxt(str(path_out.joinpath(key))+'.txt', value, fmt=f)
    if return_dict:
        return(pixels)


# paths = [
#    '/Volumes/s-biochem-kaksonen/Mateusz/microscopy/Ede1_mutants_internal/ede1-gfp/20171121_MKY0172/raw/stk2tif/Z/cells/unbudded',
#    '/Volumes/s-biochem-kaksonen/Mateusz/microscopy/Ede1_mutants_internal/ede1-gfp/20171124_MKY3682/raw/stk2tif/Z/cells/unbudded',
#    '/Volumes/s-biochem-kaksonen/Mateusz/microscopy/Ede1_mutants_internal/ede1-gfp/20171128_MKY3685/raw/stk2tif/Z/cells/unbudded',
#    '/Volumes/s-biochem-kaksonen/Mateusz/microscopy/Ede1_mutants_internal/ede1-gfp/20171128_MKY3688/raw/stk2tif/Z/cells/unbudded'
# ]
