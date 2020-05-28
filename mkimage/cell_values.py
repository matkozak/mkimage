# import modules for handling files
import csv
from pathlib import Path
from sys import argv

# import third-party packages
import numpy as np
import tifffile as tiff
from skimage.exposure import rescale_intensity

# import utility functions
from .utility import mask_cell

# %%
def batch_mask(path, pattern="*GFP*",
               rescale=False, save_arrays=False, save_summary=False):
    """
    This function reads all images with a keyword ('GFP' by default) and applies a 3D masking procedure.
    If save_arrays, each array will be saved as a .txt file.
    Summary writes a .csv file with summary statistics per cell (mean, median, sd).
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
            im = rescale_intensity(im, out_range='float')
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

    # output: save a csv file with mean intensity for each cell
    if save_summary:
        path_out = path_in.joinpath("summary.csv")
        with path_out.open('w', newline='') as f:  # initialize a csv file for writing
            # initialize csv writer and write headers
            writer = csv.writer(f, dialect='excel')
            writer.writerow(['cell', 'mean', 'median', 'sd'])
            for key, value in pixels.items():
                writer.writerow([key, round(np.mean(value), 3),
                                 np.median(value), round(np.std(value), 3)])
    # output: return dictionary of masked pixels
    return(pixels)


# get the path from command line and run counting function
if __name__ == "__main__":  # only executed if ran as script
    path = argv[1]
    batch_mask(path, save_summary=True)
