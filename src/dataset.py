import skimage.io as io
import glob
import os
import time
import matplotlib.pyplot as plt
from skimage.feature import hog
import random
from skimage import color, exposure
from skimage.feature import daisy
import numpy as np


class Instance:
    
    g_id = 0    
    def __init__(self, file_path, country, year, id=-1):
        if (id == -1):
            self.id = Instance.g_id
            Instance.g_id += 1
        self.file_path = file_path
    def load(self):
        self.image = io.imread(self.file_path)
    
    def generate_hog(self):
        gray_image = color.rgb2gray(self.image)

        return hog(gray_image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)
    def generate_daisy(self):
        gray_image = color.rgb2gray(self.image)
        return daisy(gray_image, step=180, radius=58, rings=2, histograms=6,
                         orientations=8, visualize=True)
    def generate_color_histogram(self):
        rgb = np.split(self.image, 3, 2)
        
        r_histo = np.histogram(rgb[0], 15, density=True)
        g_histo = np.histogram(rgb[1], 15, density=True)
        b_histo = np.histogram(rgb[2], 15, density=True)
        
        return np.concatenate((r_histo[0], g_histo[0], b_histo[0]))        
class Dataset:
    
    
    
    def __init__(self, dataset_folder, countries, years):
       
        self.dataset_folder = dataset_folder
        self.countries = countries
        self.years = years
        
        
    def generate_date(self):
        self.instances = []
        for country in self.countries:
            for year in self.years:
                files = glob.glob(self.dataset_folder + os.path.sep + country + os.path.sep + year + os.path.sep + "*.*")
                for file in files:
                    self.instances.append(Instance(file, country, year))
        
        print('Total # of instances:' + str(len(self.instances)))
        
#                     plt.imshow(image)
#                     plt.show()

def main():
    dataset = Dataset('./dataset', ['China', 'Japan', 'Malaysia', 'Singapore', 'South_Korea'], ['2010', '2011', '2012', '2013', '2014', '2015' ])
    
    dataset.generate_date()
    
    rIndex = random.randint(0, len(dataset.instances))
    instance = dataset.instances[rIndex]
    instance.load()
    
    fd_color_histogram = instance.generate_color_histogram()
    print(fd_color_histogram.shape)
    
    fd, hog_image = instance.generate_hog()
    
    # Visualizing HOG
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
# 
#     ax1.axis('off')
#     ax1.imshow(instance.image, cmap=plt.cm.gray)
#     ax1.set_title('Input image')
#     ax1.set_adjustable('box-forced')
#     
#     # Rescale histogram for better display
#     hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
#     
#     ax2.axis('off')
#     ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
#     ax2.set_title('Histogram of Oriented Gradients')
#     ax1.set_adjustable('box-forced')
#     plt.show()
    
    
    daisy_fd, daisy_image = instance.generate_daisy()
    
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(daisy_image)
    descs_num = daisy_fd.shape[0] * daisy_fd.shape[1]
    ax.set_title('%i DAISY descriptors extracted:' % descs_num)
    plt.show()
if __name__ == '__main__':
    main()    
