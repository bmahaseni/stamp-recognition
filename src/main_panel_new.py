from Tkinter import *
from PIL import Image, ImageTk
from dataset import Dataset
import ttk
import time

from tkFileDialog   import askopenfilename      
import tkMessageBox
import matplotlib
# matplotlib.use("Qt4Agg")
from shutil import copyfile

import skimage.io as io
import glob
import os
import time
import matplotlib
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
from sklearn.externals import joblib
from tornado.gen import Task


country_map = {'China':0, 'Japan':1, 'Malaysia':2, 'Singapore':3, 'South_Korea':4, 'Not_Stamp': 5}

from instance import Instance
from dataset import Dataset
import pickle



dataset = Dataset('./dataset', ['China', 'Japan', 'Malaysia', 'Singapore', 'South_Korea', 'Not_Stamp'], ['2010', '2011', '2012', '2013', '2014', '2015'])

dataset.generate_date()


class ResultDialog:
    def __init__(self, parent, result):
        
        global loaded_model
        global loaded_model_name
        top = self.top = Toplevel(parent)
        top.geometry("%dx%d%+d%+d" % (500, 400, 250, 125))
        T = Text(top, height=10, width=30)
        T.pack()
        
        T.insert(END, result)

class ImageDialog:
    def __init__(self, parent, image_name):
        self.rotate = 0
        self.add = False
        global loaded_model
        global loaded_model_name
        self.image_name = image_name
        top = self.top = Toplevel(parent)
        top.geometry("%dx%d%+d%+d" % (500, 400, 250, 125))
        self.top = top
        image = Image.open(image_name)
        image.thumbnail((200, 200))
        photo = ImageTk.PhotoImage(image)
        self.label = Label(top, image=photo, text='', compound=LEFT)
        self.label.image = photo  # keep a reference!
        self.label.pack()
        
        b = Button(top, text="Test", command=self.test_image)
        b.pack(pady=5)
        b = Button(top, text="Rotate", command=self.rotate_image)
        b.pack(pady=5)
        label = Label(top, text='Loaded Country Model Name: ' + loaded_country_model_name.split('/')[-1], compound=LEFT)
        label.pack()
        label = Label(top, text='Loaded Year Model Name: ' + loaded_year_model_name.split('/')[-1], compound=LEFT)
        label.pack()

        self.result = Label(self.top, text='', compound=LEFT)
        self.result.pack()
        
        self.image_instance = Instance(file_path=image_name, id=0)
        
    def test_image(self):

        print 'test'		
        self.image_instance.load()
        # append color histogram
        features = self.image_instance.generate_color_histogram()
        # append HOG
        features = np.concatenate((features, self.image_instance.generate_hog()[0]))
#             features = instance.generate_hog()[0]
        # append DAISY
        # features = np.concatenate((features, instance.generate_daisy()[0]))  #                       
        _pred_country = loaded_country_model.predict(features)
        _pred_year = loaded_year_model.predict(features)
        country_code = {0: 'China', 1:'Japan', 2:'Malaysia', 3:'Singapore', 4:'South_Korea', 5: 'Unkown'}
        year_code = {0: '2010', 1:'2011', 2:'2012', 3:'2013', 4:'2014', 5:'2015', 6: '-1000'}
        if country_code == 5:
            self.result['text'] = 'Country:' + country_code[int(_pred_country)]
        else:
            self.result['text'] = 'Country:' + country_code[int(_pred_country)] + ' Year:' + year_code[int(_pred_year)]
        if self.add == False:
            b = Button(self.top, text="Add", command=self.add_image_dataset)
            b.pack(pady=5)
            self.add = True
        self.country = country_code[int(_pred_country)]
        self.year = year_code[int(_pred_year)]

    def rotate_image(self):

        print 'rotate'
        self.rotate = (self.rotate + 1) % 4
        print self.image_name
        image = Image.open(self.image_name)
        image.thumbnail((200, 200))
        for i in xrange(self.rotate):
            image = image.rotate(90)
        print self.rotate
        photo = ImageTk.PhotoImage(image)
        self.label.configure(image = photo)
        self.label.image = photo
        app.update_idletasks()


    def add_image_dataset(self):
        copyfile(self.image_instance.file_path, dataset.dataset_folder + '/' + self.country + '/' + self.year + '/' + self.image_instance.file_path.split('/')[-1].strip())
        self.top.destroy()

class TrainDialog:

    def __init__(self, parent):
        global new_model_name
        global algorithm

        top = self.top = Toplevel(parent)

        Label(top, text="New Model Name").pack()

        
        self.e = Entry(top)
        self.e.pack(padx=5)
        Label(top, text="-----------------------------------------------------").pack()
        Label(top, text="Character").pack()
        ##############################################################################
        #### Country or Year?
        MODES = [
            ("Country", "0"),
            ("Year", "1")
            ]
    
        v = StringVar()
        v.set("0")  # initialize
        i = 0
        for text, mode in MODES:    
            if i == 0:
                b = Radiobutton(top, text=text,
                        variable=v, value=mode, command=self.set_country)
            else:
                b = Radiobutton(top, text=text,
                        variable=v, value=mode, command=self.set_year)
    
            b.pack(anchor=W)
            i += 1

        Label(top, text="-----------------------------------------------------").pack()
        Label(top, text="Algorithm").pack()
        ##############################################################################
        #### Algorithm?
    
        MODES = [
            ("Logistic Regression", "0"),
            ("SVM", "1")
            ]
    
        v = StringVar()
        v.set("0")  # initialize
        i = 0
        for text, mode in MODES:    
            if i == 0:
                b = Radiobutton(top, text=text,
                        variable=v, value=mode, command=self.set_lr)
            else:
                b = Radiobutton(top, text=text,
                        variable=v, value=mode, command=self.set_svm)
    
            b.pack(anchor=W)
            i += 1

        b = Button(top, text="Start", command=self.train_model)
        b.pack(pady=5)

    
    def set_lr(self):
        global algorithm
        algorithm = 0
    def set_svm(self):
        global algorithm
        algorithm = 1
        
    def set_country(self):
        global character
        character = 0
    def set_year(self):
        global character
        character = 1
        
        
    def train_model(self):
        global algorithm
        global character
        if self.e.get().strip() == '':
            return 
        new_model_name = self.e.get()
        print "mode name is", self.e.get()
        print "algorithm name is", algorithm
        #     train samples
        global number_training_samples
        s = None
        if character == 0:            
            X_train , y_train = dataset.get_training_data_country()
            number_training_samples = len(y_train)
            if algorithm == 0:
                s = train_logistic_regression_country(X_train , y_train)
            else:
                s = train_SVM_country(X_train , y_train)
        else:
            print dataset
            X_train , y_train = dataset.get_training_data_year()             
            number_training_samples = len(y_train)
            if algorithm == 0:
                s = train_logistic_regression_year(X_train , y_train)
            else:
                s = train_SVM_year(X_train , y_train)
            
        joblib.dump(s, new_model_name + '.pkl') 
        self.top.destroy()


def train_logistic_regression_country(X_train, y_train):    
    print('Training Logistic Regression Classifier')
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, y_train)
    s = pickle.dumps(logistic_regression)
    return s
def train_SVM_country(X_train, y_train): 
    print('Training SVM Classifier')   
    svm_classifier = NuSVC()
    svm_classifier.fit(X_train, y_train) 
    s = pickle.dumps(svm_classifier)
    return s


def train_logistic_regression_year(X_train, y_train):    
    print('Training Logistic Regression Classifier')
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, y_train)
    s = pickle.dumps(logistic_regression)
    return s
def train_SVM_year(X_train, y_train): 
    print('Training SVM Classifier')   
    svm_classifier = NuSVC()
    svm_classifier.fit(X_train, y_train) 
    s = pickle.dumps(svm_classifier)
    return s

def load_country_model():
    model_file = askopenfilename()
    global loaded_country_model_name
    loaded_country_model_name = model_file
    global loaded_country_model
    loaded_country_model = pickle.loads(joblib.load(model_file))
def load_year_model():
    model_file = askopenfilename()
    global loaded_year_model_name
    loaded_year_model_name = model_file
    global loaded_year_model
    loaded_year_model = pickle.loads(joblib.load(model_file))

def open_file():
    global loaded_country_model
    global loaded_year_model
    if loaded_year_model == None or loaded_country_model == None:
        tkMessageBox.showwarning(
            "Open file",
            "Model file is not selected"
        )
        return
    image_file = askopenfilename()
    print image_file
    d = ImageDialog(app.frame, image_file)

    app.frame.wait_window(d.top)




def pad_image(image):
    max_dim = max(image.size)
    ratio = max_dim / 100
    image = image.resize((image.size[0] / ratio, image.size[1] / ratio))
    old_size = image.size
    new_size = (100, 100)
    new_im = Image.new("RGB", new_size)  # # luckily, this is already black!
    new_im.paste(image, ((new_size[0] - old_size[0]) / 2,
    	              (new_size[1] - old_size[1]) / 2))
    return new_im
def load_dataset():
    app.frame.pack()
    app.message_label.config(text='')
    app.console_message.config(text='')

    global index
    global stamp_labels
    for stamp_label in stamp_labels:
        stamp_label.destroy()
    index = 0
    for i in xrange(15):
        image = Image.open(dataset.instances[i].file_path)
        image = pad_image(image)
        # image.thumbnail((100, 100))
        photo = ImageTk.PhotoImage(image)
        text_str = 'Country:' + dataset.instances[i].country + ' Year:' + dataset.instances[i].year
        label = Label(app.frame, image=photo, text=text_str, compound=LEFT)
        label.image = photo  # keep a reference!
        label.grid(row=15 + i / 3, column=i % 3)
        stamp_labels.append(label)

def train():
    global stamp_labels
    for stamp_label in stamp_labels:
        stamp_label.config(text='', image='')
    stamp_labels = []

    app.message_label.pack()
    app.console_message.pack()
    app.update_idletasks()

    global number_training_samples

    start_time = time.time()
    d = TrainDialog(app.frame)
    app.frame.wait_window(d.top)
    end_time = time.time()
    prediction_time = end_time - start_time
    app.message_label.config(text='Message')
    app.console_message.config(text='Total #examples:' + str(number_training_samples) + '\n' + 'Processing time:' + str(prediction_time))
    app.update_idletasks()

def evaluate_country():
    global stamp_labels
    print len(stamp_labels)
    for stamp_label in stamp_labels:
        stamp_label.config(text='', image='')
    stamp_labels = []
    app.message_label.config(text='')
    app.console_message.config(text='')

    app.message_label.pack()
    app.console_message.pack()
    app.update_idletasks()

    model_file = askopenfilename()
    print model_file
    start_time = time.time()

    model = pickle.loads(joblib.load(model_file))
    app.progress.config(text='Please Wait ...')
    app.update_idletasks()
    X_test , y_test = dataset.get_testing_data_country()
    end_time = time.time()
    data_load_time = end_time - start_time
    start_time = end_time
    y_pred = model.predict(X_test)
    end_time = time.time()
    prediction_time = end_time - start_time

    cm = confusion_matrix(y_test, y_pred)
    
    
    country_code = {0: 'China', 1:'Japan', 2:'Malaysia', 3:'Singapore', 4:'South_Korea', 5: 'Unkown'}
    year_code = {0: '2010', 1:'2011', 2:'2012', 3:'2013', 4:'2014', 5:'2015', 6: '-1000'}
    s = 'China'.ljust(15) + '\t' + 'Japan'.ljust(15) + '\t' + 'Malaysia'.ljust(15) + '\t' + 'Singapore'.ljust(15) + '\t' + 'South_Korea'.ljust(15) + '\n'
    correct = 0.0
    total = 0.0
    for i in xrange(len(country_code) - 1):
        s += country_code[i].ljust(15) + '\t'
        for j in xrange(len(country_code) - 1):
            s += str(cm[i][j]).ljust(15) + '\t'
            if i == j:
                correct += cm[i][j]
            total += cm[i][j]
        s += '\n'
    
    s += 'Accuracy: ' + str(correct / total) 
    s += '\n\n'
    s += '-----------------\n'
    app.progress.config(text='')
    # d = ResultDialog(app.frame, cm)
    app.message_label.config(text='Result')
    s += 'Data loading time : ' + str(data_load_time) + '\n'
    s += 'Prediction time : ' + str(prediction_time) + '\n'
    s += 'Total Processing time : ' + str(prediction_time + data_load_time) + '\n'
    app.console_message.config(text=s)
    app.update_idletasks()
    print s
#     app.frame.wait_window(d.top)


def evaluate_year():
    global stamp_labels
    for stamp_label in stamp_labels:
        stamp_label.config(text='', image='')
    stamp_labels = []
    app.message_label.config(text='')
    app.console_message.config(text='')

    app.message_label.pack()
    app.console_message.pack()
    app.update_idletasks()

    model_file = askopenfilename()
    print model_file
    model = pickle.loads(joblib.load(model_file))
    app.progress.config(text='Please Wait ...')
    app.update_idletasks()

    start_time = time.time()
    X_test , y_test = dataset.get_testing_data_year()
    end_time = time.time()
    data_load_time = end_time - start_time
    start_time = end_time

    y_pred = model.predict(X_test)
    end_time = time.time()
    prediction_time = end_time - start_time

    cm = confusion_matrix(y_test, y_pred)

    country_code = {0: 'China', 1:'Japan', 2:'Malaysia', 3:'Singapore', 4:'South_Korea', 5: 'Unkown'}
    year_code = {0: '2010', 1:'2011', 2:'2012', 3:'2013', 4:'2014', 5:'2015', 6: '-1000'}
    s = '2010'.ljust(8) + '\t' + '2011'.ljust(8) + '\t' + '2012'.ljust(8) + '\t' + '2013'.ljust(8) + '\t' + '2014'.ljust(8) + '\t' + '2015'.ljust(8) + '\n'
    correct = 0.0
    total = 0.0

    for i in xrange(len(year_code) - 1):
        s += year_code[i].ljust(8) + '\t'
        for j in xrange(len(year_code) - 1):
            s += str(cm[i][j]).ljust(8) + '\t'
            if i == j:
                correct += cm[i][j]
            total += cm[i][j]

        s += '\n'
    
    s += 'Accuracy: ' + str(correct / total)
    s += '\n\n'
    s += '-----------------\n'

    
#     d = ResultDialog(app.frame, cm)
    
#     app.frame.wait_window(d.top)
    app.progress.config(text='')
    # d = ResultDialog(app.frame, cm)
    app.message_label.config(text='Message')
    s += 'Data loading time : ' + str(data_load_time) + '\n'
    s += 'Prediction time : ' + str(prediction_time) + '\n'
    s += 'Total Processing time : ' + str(prediction_time + data_load_time) + '\n'
    print s
    app.console_message.config(text=s)
    app.update_idletasks()


    
def About():
    print "This is a simple example of a menu"


def next_page():
    if len(app.message_label['text'].strip()) != 0:
        app.message_label.config(text='')
        app.console_message.config(text='')
    global index
    if index * 15 < len(dataset.instances):	
        index += 1
        global stamp_labels
        for stamp_label in stamp_labels:
            stamp_label.destroy()
        stamp_labels = []
        for i in xrange(15):
            image = Image.open(dataset.instances[index * 15 + i].file_path)
            image = pad_image(image)
            # image.thumbnail((100, 100))
            photo = ImageTk.PhotoImage(image)
            text_str = 'Country:' + dataset.instances[i].country + ' Year:' + dataset.instances[i].year
            label = Label(app.frame, image=photo, text=text_str, compound=LEFT)
            label.image = photo  # keep a reference!
            label.grid(row=15 + i / 3, column=i % 3)
            stamp_labels.append(label)
def previous_page():
    if len(app.message_label['text'].strip()) != 0:
        app.message_label.config(text='')
        app.console_message.config(text='')
    global index
    if index != 0:
        index -= 1
        global stamp_labels
        for stamp_label in stamp_labels:
            stamp_label.destroy()
        stamp_labels = []
        for i in xrange(15):
            image = Image.open(dataset.instances[index * 15 + i].file_path)
            image = pad_image(image)
            # image.thumbnail((100, 100))
            photo = ImageTk.PhotoImage(image)
            text_str = 'Country:' + dataset.instances[i].country + ' Year:' + dataset.instances[i].year
            label = Label(app.frame, image=photo, text=text_str, compound=LEFT)
            label.image = photo  # keep a reference!
            label.grid(row=15 + i / 3, column=i % 3)
            stamp_labels.append(label)

class StampApp(Tk):

    def __init__(self, *args, **kwargs):
        self.root = Tk.__init__(self, *args, **kwargs)
        self.label = Label(text="")
        self.label.pack()
        self.label = Label(text="")
        self.label.pack()
        b = Button(text="previous", command=previous_page)
        b.pack()
        self.label = Label(text="")
        self.label.pack()
        
        self.frame = Frame(self.root)
        self.frame.pack()
        self.label = Label(text="")
        self.label.pack()
        b = Button(text="next", command=next_page)
        b.pack()

        self.label = Label(text="------------------------")
        self.label.pack()

        self.progress = Label(text='')
        self.progress.pack()
        
        
        # self.progress.pack_forget()
        self.message_label = Label(text="")
        self.message_label.pack()

        self.console_message = Label(text="")
        self.console_message.pack()


# global variables
index = 0
stamp_labels = []
loaded_country_model = None
loaded_country_model_name = ''
loaded_year_model = None
loaded_year_model_name = ''

algorithm = 0  # 0)logisitic regression 1) SVM
character = 0
#





app = StampApp()


number_training_samples = 0

menu = Menu(app.frame)
app.config(menu=menu)

###########################################
filemenu = Menu(menu)
menu.add_cascade(label="File", menu=filemenu)
filemenu.add_command(label="Load Country Model", command=load_country_model)
filemenu.add_command(label="Load Year Model", command=load_year_model)
filemenu.add_command(label="Evaluate an Image", command=open_file)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=app.quit)

###########################################
dataset_menue = Menu(menu)
menu.add_cascade(label="Dataset", menu=dataset_menue)
dataset_menue.add_command(label="Show Dataset Images", command=load_dataset)
dataset_menue.add_command(label="Train", command=train)
dataset_menue.add_command(label="Evaluate Country Character", command=evaluate_country)
dataset_menue.add_command(label="Evaluate Year Character", command=evaluate_year)
#########################################3#
helpmenu = Menu(menu)
menu.add_cascade(label="Help", menu=helpmenu)
helpmenu.add_command(label="About...", command=About)

app.wm_title("Stamp Recognition Panel")

app.geometry('1200x700')




app.mainloop()

# root.mainloop()
