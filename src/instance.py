import matplotlib
# matplotlib.use("Qt4Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import skimage.io as io
import glob
import os
import time
import matplotlib

import matplotlib.pyplot as plt
from skimage.feature import hog
import random
from skimage import color, exposure
from skimage.feature import daisy
import numpy as np
from  skimage.transform import resize
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

from sklearn.svm import NuSVC

country_map = {'China':0, 'Japan':1, 'Malaysia':2, 'Singapore':3, 'South_Korea':4}

class Instance:
    
    g_id = 0    
    def __init__(self, file_path, country = None, year = None, id=-1):
        if (id == -1):
            self.id = Instance.g_id
            Instance.g_id += 1
        self.file_path = file_path
        self.country = country
        self.year = year
    def load(self):
        self.image = io.imread(self.file_path)
        self.image = resize(self.image, (100, 100))
        if  self.image.shape[0] <= 0:
            raise 'error in loading' + self.file_path
    def free(self):
        self.image = None
    def generate_hog(self):
        gray_image = color.rgb2gray(self.image)

        return hog(gray_image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)
    def generate_daisy(self):
        gray_image = color.rgb2gray(self.image)
        return daisy(gray_image, step=180, radius=32, rings=2, histograms=6,
                         orientations=8, visualize=True)
    def generate_color_histogram(self):
        rgb = np.split(self.image, 3, 2)
        
        r_histo = np.histogram(rgb[0], 15, density=True)
        g_histo = np.histogram(rgb[1], 15, density=True)
        b_histo = np.histogram(rgb[2], 15, density=True)
        
        return np.concatenate((r_histo[0], g_histo[0], b_histo[0]))        
