from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import os
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks


def open_xlsx(fname):
    """
    the function read the info file of the experiment
    :param fname: file name
    :return:
    pandas data frame
    """
    data = pd.read_excel(fname)
    data = data.set_index('Parameter')
    return data


def open_rtv(fname):
    """
    the function read the binary file of the fast-frame xrapid camera
    :param fname: file name
    :return:
    numpy array (4,1024,1360)
    4 frames
    """
    file = open(fname, 'rb')
    n = 1024 * 1360
    file_array = np.fromfile(file, dtype='uint16', offset=0x2000, count=n * 4).reshape((4, 1024, 1360))
    ar_right = np.copy(file_array[1::2, :, :1360 // 2])
    ar_left = np.copy(file_array[1::2, :, 1360 // 2:])
    file_array[1::2, :, :1360 // 2] = ar_left
    file_array[1::2, :, 1360 // 2:] = ar_right

    image_array = np.copy(file_array)
    file.close()
    return image_array


def open_folder():
    """
    The function loads the data of experiment from file dialog
    the experiment folder includes:
    'info.xlsx' file with scalar data of experiment
    'before.rtv' bin file with images from xrapid came
    :return:
    dict of data
    """
    folder_name = filedialog.askdirectory(
        initialdir='./example')
    os.chdir(folder_name)
    files_data = dict()
    for fname in os.listdir():
        if fname.split('.')[-1] == 'rtv':
            data = open_rtv(fname)
            if fname.split('.')[0] == 'before':
                files_data['before'] = data
            else:
                files_data['shot'] = data
            continue
        if fname.split('.')[-1] == 'xlsx':
            files_data['info'] = open_xlsx(fname)
            continue
    pass
    return files_data
