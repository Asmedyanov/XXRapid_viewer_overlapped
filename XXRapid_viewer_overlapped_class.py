import numpy as np

# from my_math import *
from my_os import *
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft2, ifft2
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion


def clean_picks(image_array):
    n_image, h_image, w_image = image_array.shape
    ret_array = image_array
    fft_array_abs = np.abs(image_array)
    for i in range(n_image):
        max_filter = maximum_filter(fft_array_abs[i], size=100)
        max_index = np.argwhere(fft_array_abs[i] == max_filter)
        w = 20
        for xy in max_index:
            if xy in np.array([[0, 0], [h_image - 1, 0], [0, w_image - 1], [h_image, w_image]]):
                continue
            x0 = xy[1]
            y0 = xy[0]
            ret_array[i, y0, x0] = 0
            for r in np.arange(1, w):
                for t in np.arange(0, 2.0 * np.pi, np.pi / int(2.0 * np.pi * r)):
                    x = int(x0 + r * np.cos(t))
                    y = int(y0 + r * np.sin(t))
                    if (x < 0):
                        x = 0
                    if (y < 0):
                        y = 0
                    if (y >= h_image):
                        y = h_image - 1
                    if (x >= w_image):
                        x = w_image - 1
                    ret_array[i, y, x] = 0
    return ret_array


def get_norm_array(image_array):
    ret = image_array.astype(float)
    for i in range(image_array.shape[0]):
        ret[i] -= ret[i].min()
        ret[i] /= ret[i].max()
    return ret


class XXRapid_viewer_overlapped_class:
    def __init__(self):
        self.data_dict = open_folder()
        self.curdir = os.curdir
        self.sort_data_dict()
        self.image_preprocessing()
        self.overlap_images()

    def save_all_images(self, name):
        fig, ax = plt.subplots(2, 4)
        fig.set_size_inches(11.7, 8.3)
        for i in range(4):
            ax[0, i].imshow(self.before_array[i])
            '''ax[0, i].set_title(
                f'shutters {int(self.starts[::2][i] * 1000)} ns and {int(self.starts[1::2][i] * 1000)} ns')'''
            ax[1, i].imshow(self.shot_array[i])
        plt.tight_layout()
        fig.savefig(name)
        plt.close()

    def save_all_images_specter(self, name):
        fig, ax = plt.subplots(2, 4)
        fig.set_size_inches(11.7, 8.3)
        for i in range(4):
            show_array = np.abs(self.before_array_specter[i]) + 1.0e-5
            show_array = np.where(show_array > 1.0e-3, np.log(show_array), 1.0e-3)
            ax[0, i].imshow(show_array)
            '''ax[0, i].set_title(f'from {int(self.starts[i] * 1000)} ns to {int(self.stops[i] * 1000)} ns')'''
            show_array = np.abs(self.shot_array_specter[i]) + 1.0e-5
            show_array = np.where(show_array > 1.0e-3, np.log(show_array), 1.0e-3)
            ax[1, i].imshow(show_array)
        plt.tight_layout()
        fig.savefig(name)
        plt.close()

    def image_preprocessing(self):
        os.makedirs('common', exist_ok=True)
        # common preprocessing
        self.save_all_images('common/0.original.png')
        self.shape = self.before_array.shape
        self.framecount, self.frameheight, self.framewidth = self.shape
        self.before_array = get_norm_array(self.before_array)
        self.shot_array = get_norm_array(self.shot_array)
        self.save_all_images('common/1.normed.png')

        self.before_array_specter = fft2(self.before_array)
        self.shot_array_specter = fft2(self.shot_array)
        self.save_all_images_specter('common/2.fft2_original.png')

        self.before_array_specter = clean_picks(self.before_array_specter)
        self.shot_array_specter = clean_picks(self.shot_array_specter)
        self.save_all_images_specter('common/3.fft_cut.png')

        self.before_array = np.abs(ifft2(self.before_array_specter))
        self.shot_array = np.abs(ifft2(self.shot_array_specter))
        self.save_all_images('common/4.filtered.png')

    def sort_data_dict(self):
        """
        The function distributes the experiment data dictionary
        :return:
        """
        self.dy = self.data_dict['info']['Value']['dy']
        self.dx = self.data_dict['info']['Value']['dx']
        '''self.h_foil = self.data_dict['info']['Value']['Thickness']
        self.waist = self.data_dict['info']['Value']['Waist']
        self.w_foil = self.data_dict['info']['Value']['Width']
        self.l_foil = self.data_dict['info']['Value']['Length']
        self.w_front = self.data_dict['info']['Value']['w_front']
        self.w_smooth = self.data_dict['info']['Value']['w_smooth']
        self.sequence = np.array(self.data_dict['info']['Value']['Sequence'].split(','), dtype='int')
        self.density = self.data_dict['info']['Value']['Density']
        self.mass = self.density * self.h_foil * self.l_foil * 0.5 * (self.w_foil + self.waist) * 1.0e-3
        # self.shot_name, self.before_array, self.shot_array, self.peak_times, self.wf_time, self.current = open_images()'''
        self.before_array = self.data_dict['before']
        self.shot_array = self.data_dict['shot']
        '''self.peak_times = self.data_dict['waveform']['peaks']
        self.wf_time = self.data_dict['waveform']['time']
        self.wf_time_power = self.data_dict['waveform']['time_power']
        self.current = self.data_dict['waveform']['current']
        self.power = self.data_dict['waveform']['power']
        self.resistance = self.data_dict['waveform']['resistance']
        self.u_res = self.data_dict['waveform']['u_resistive']
        self.energy = self.data_dict['waveform']['energy']
        self.starts = self.peak_times[::2]
        self.starts = np.flip(self.starts[self.sequence])
        self.stops = self.peak_times[1::2]'''

    def show_original(self):
        fig, ax = plt.subplots(self.shot_array.shape[0], self.shot_array.shape[1])
        fig.set_layout_engine(layout='tight')
        try:
            for i in range(self.shot_array.shape[1]):
                ax[i].imshow(self.shot_array[0, i], cmap='gray_r')
        except:
            for i in range(self.shot_array.shape[0]):
                for j in range(self.shot_array.shape[1]):
                    ax[i, j].imshow(self.shot_array[i, j], cmap='gray_r')
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()

    def overlap_images(self):
        self.overlapped_image = np.abs(self.shot_array - self.before_array)
        extent = [0,
                  self.framewidth * self.dx,
                  self.frameheight * self.dx,
                  0]
        min_non_zerro = self.overlapped_image[np.nonzero(self.overlapped_image)].min()
        self.overlapped_image = np.where(self.overlapped_image <= 0, min_non_zerro, self.overlapped_image)
        self.overlapped_image = np.where(self.overlapped_image >= self.before_array.max(), self.before_array.max(),
                                         self.overlapped_image)
        self.overlapped_image -= self.overlapped_image.min()
        self.overlapped_image /= self.overlapped_image.max()
        fig, ax = plt.subplots(ncols=2, nrows=2)
        fig.set_layout_engine(layout='tight')
        '''for i in range(self.overlapped_image.shape[0]):'''
        ax[0, 0].imshow(self.overlapped_image[0], cmap='gray_r', extent=extent)
        ax[0, 0].set_ylabel('Image 1, mm')
        ax[0, 0].grid()
        ax[0, 1].imshow(self.overlapped_image[1], cmap='gray_r', extent=extent)
        ax[0, 1].set_ylabel('Image 2, mm')
        ax[0, 1].grid()
        ax[1, 0].imshow(self.overlapped_image[2], cmap='gray_r', extent=extent)
        ax[1, 0].set_ylabel('Image 3, mm')
        ax[1, 0].grid()
        ax[1, 1].imshow(self.overlapped_image[3], cmap='gray_r', extent=extent)
        ax[1, 1].set_ylabel('Image 4, mm')
        ax[1, 1].grid()
        plt.savefig('common/overlapped.png')
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        plt.show()
        pass
