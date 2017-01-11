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

from instance import Instance

country_map = {'China':0, 'Japan':1, 'Malaysia':2, 'Singapore':3, 'South_Korea':4}

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
        

        print('Shuffling instances')        
        rr = range(len(self.instances))
        np.random.shuffle(rr)
        self.instances = np.take(self.instances, rr, axis=0)        
        
        print('Generating training instances')
        self.training_instances = []
        for i in xrange((len(self.instances) * 2) / 3):
            self.training_instances.append(self.instances[i])    

        print('Generating testing instances')
        self.testing_instances = []
        for i in range((len(self.instances) * 2) / 3, len(self.instances)):
            self.testing_instances.append(self.instances[i])        
        
        print('Done.')
    def get_training_data_country(self):        
        X = []
        y = []
        print('Generate training data for country classification')
        for instance in self.training_instances:
            instance.load()
            # append color histogram
            features = instance.generate_color_histogram()
            # append HOG
            features = np.concatenate((features, instance.generate_hog()[0]))
#             features = instance.generate_hog()[0]
            # append DAISY
            #features = np.concatenate((features, instance.generate_daisy()[0]))  #                       
            X.append(features)
            y.append(country_map[instance.country.strip()])
            instance.free()
        X = np.asanyarray(X)
        y = np.asanyarray(y)
        return X, y        

    def get_testing_data_country(self):        
        X = []
        y = []
        print('Generate testing data for country classification')
        for instance in self.testing_instances:
            instance.load()
            # append color histogram
            features = instance.generate_color_histogram()
            # append HOG
            features = np.concatenate((features, instance.generate_hog()[0]))
            # append DAISY
#             features = np.concatenate((features, instance.generate_daisy()[0]))  #  
#             features = instance.generate_hog()[0]                    
            X.append(features)
            y.append(country_map[instance.country.strip()])
            instance.free()
        X = np.asanyarray(X)
        y = np.asanyarray(y)
        return X, y        
    
    
    def get_training_data_year(self):
        X = []
        y = []
                
#    def get_testing_data(self):
    
        
#                     plt.imshow(image)
#                     plt.show()



#this is just to test and visualize some images
def main():
    rIndex = random.randint(0, len(dataset.instances))
    rIndex = 431
    print(rIndex)
    instance = dataset.instances[rIndex]
    instance.load()
#     
    fd_color_histogram = instance.generate_color_histogram()
#     print(fd_color_histogram.shape)
#     
    fd, hog_image = instance.generate_hog()
#     
#     # Visualizing HOG
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(instance.image, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')
#     # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
     
    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()
#     
         
    daisy_fd, daisy_image = instance.generate_daisy()
     
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(daisy_image)
    descs_num = daisy_fd.shape[0] * daisy_fd.shape[1]
    ax.set_title('%i DAISY descriptors extracted:' % descs_num)
    plt.show()

if __name__ == '__main__':

    matplotlib.use("gtk")
    main()    


    
