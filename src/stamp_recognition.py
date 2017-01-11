import matplotlib
# matplotlib.use("Qt4Agg")

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

from instance import Instance
from dataset import Dataset


class StampRecognition:

    def plot_confusion_matrix(self, cm, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        
    def train_test_logistic_regression(self, X_train, y_train, X_test, y_test):    
        print('Training Logistic Regression Classifier')
        logistic_regression = LogisticRegression()
        logistic_regression.fit(X_train, y_train)
        
        print('Testing Logistic Regression Classifier')
        y_pred = logistic_regression.predict(X_test)
    
        cm = confusion_matrix(y_test, y_pred)
        
        print(cm)
    
    def train_test_SVM(self, X_train, y_train, X_test, y_test): 
        print('Training SVM Classifier')   
        svm_classifier = NuSVC()
        svm_classifier.fit(X_train, y_train) 
    
        print('Testing SVM Classifier')
        y_pred = svm_classifier.predict(X_test)
        
        print(y_pred.shape)
        cm = confusion_matrix(y_test, y_pred)        
        print(cm)

def main():
    
    stamp_recogntion = StampRecognition()
    dataset = Dataset('./dataset', ['China', 'Japan', 'Malaysia', 'Singapore', 'South_Korea'], ['2010', '2011', '2012', '2013', '2014', '2015' ])
    
    dataset.generate_date()
    
    
#     train samples
    X_train , y_train = dataset.get_training_data_country()
    
#     test samples
    X_test , y_test = dataset.get_testing_data_country()
    
    stamp_recogntion.train_test_logistic_regression(X_train, y_train, X_test, y_test)
    
    stamp_recogntion.train_test_SVM(X_train, y_train, X_test, y_test)  
    
    
      
#     plot_confusion_matrix(cm)
    
    
    return
    
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
