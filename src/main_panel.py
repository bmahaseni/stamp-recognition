from Tkinter import *
from PIL import Image, ImageTk
from dataset import Dataset

from tkFileDialog   import askopenfilename      

dataset = Dataset('./dataset', ['China', 'Japan', 'Malaysia', 'Singapore', 'South_Korea'], ['2010', '2011', '2012', '2013', '2014', '2015' ])

dataset.generate_date()




class VerticalScrolledFrame(Frame):
    """A pure Tkinter scrollable frame that actually works!
    * Use the 'interior' attribute to place widgets inside the scrollable frame
    * Construct and pack/place/grid normally
    * This frame only allows vertical scrolling

    """
    def __init__(self, parent, *args, **kw):
        Frame.__init__(self, parent, *args, **kw)            

        # create a canvas object and a vertical scrollbar for scrolling it
        vscrollbar = Scrollbar(self, orient=VERTICAL)
        vscrollbar.pack(fill=Y, side=RIGHT, expand=TRUE)
        canvas = Canvas(self, bd=0, highlightthickness=0,
                        yscrollcommand=vscrollbar.set)
        canvas.pack(side=LEFT, fill=BOTH, expand=TRUE)
        vscrollbar.config(command=canvas.yview)

        # reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)
        canvas.config(scrollregion='0 0 %s %s' % (100, 100))


        # create a frame inside the canvas which will be scrolled with it
        self.interior = interior = Frame(canvas)
        interior_id = canvas.create_window(0, 0, window=interior,
                                           anchor=NW)

        # track changes to the canvas and frame width and sync them,
        # also updating the scrollbar
        def _configure_interior(event):
            # update the scrollbars to match the size of the inner frame
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            canvas.config(scrollregion="0 0 %s %s" % size)
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the canvas's width to fit the inner frame
                canvas.config(width=interior.winfo_reqwidth())
        interior.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the inner frame's width to fill the canvas
                canvas.itemconfigure(interior_id, width=canvas.winfo_width())
        canvas.bind('<Configure>', _configure_canvas)

def load_model():
    model_file = askopenfilename()
    print model_file

def open_file():
    image_file = askopenfilename()
    print image_file

def load_dataset():
    for i in xrange(5):
        image = Image.open(dataset.instances[i].file_path)
        image.thumbnail((100, 100), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        text_str = '\tCountry:' + dataset.instances[i].country + ' Year:' + dataset.instances[i].year + '\t\t'
        label = Label(app.frame.interior, image=photo, text=text_str, compound=LEFT)
        label.image = photo  # keep a reference!
        label.grid(row= i/2, column=i%2)

def train():
    scrollbar = Scrollbar(root)
    scrollbar.pack(side=RIGHT, fill=Y)
    
    listbox = Listbox(root, yscrollcommand=scrollbar.set)
    for i in range(20):
        image = Image.open(dataset.instances[i].file_path)
        image.thumbnail((98, 98), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        text_str = 'Country:' + dataset.instances[i].country + ' Year:' + dataset.instances[i].year
        label = Label(image=photo, text=text_str, compound=LEFT)
        label.image = photo  # keep a reference!
        listbox.insert(END, label)
    listbox.pack(side=LEFT, fill=BOTH)
    
    scrollbar.config(command=listbox.yview)


def train_logistic_regression():
        print('Training Logistic Regression Classifier')
        logistic_regression = LogisticRegression()
        logistic_regression.fit(X_train, y_train)

	c = ProgressDialog(t, title="Please wait...",
                  type="infinite",
                      width=20,
                      stop="Stop",
                      textvariable=progmsg,
                      variable=progval,
                      command=lambda: c.destroy()
                      )
def update_progress():
         progval.set(2)
         c.after(20, update_progress)


def evaluate():
    scrollbar = Scrollbar(root)
    scrollbar.pack(side=RIGHT, fill=Y)
    
    listbox = Listbox(root, yscrollcommand=scrollbar.set)
    for i in range(20):
        image = Image.open(dataset.instances[i].file_path)
        image.thumbnail((98, 98), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        text_str = 'Country:' + dataset.instances[i].country + ' Year:' + dataset.instances[i].year
        label = Label(image=photo, text=text_str, compound=LEFT)
        label.image = photo  # keep a reference!
        #label.pack()
        listbox.insert(END, label)
    listbox.pack(side=LEFT, fill=BOTH)
    
    scrollbar.config(command=listbox.yview)



def About():
    print "This is a simple example of a menu"



class StampApp(Tk):
	def __init__(self, *args, **kwargs):
	    root = Tk.__init__(self, *args, **kwargs)
	    self.label = Label(text="")
	    self.label.pack()
	    self.label = Label(text="")
	    self.label.pack()
	    self.frame = VerticalScrolledFrame(root)
	    self.frame.pack()

app = StampApp()


menu = Menu(app.frame)
app.config(menu=menu)

###########################################
filemenu = Menu(menu)
menu.add_cascade(label="File", menu=filemenu)
filemenu.add_command(label="Load Model", command=load_model)
filemenu.add_command(label="Open Image", command=open_file)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=app.quit)

###########################################
dataset_menue = Menu(menu)
menu.add_cascade(label="Dataset", menu=dataset_menue)
dataset_menue.add_command(label="Import Images", command=load_dataset)
dataset_menue.add_command(label="Train", command=train)
dataset_menue.add_command(label="Evaluate", command=evaluate)
#########################################3#
helpmenu = Menu(menu)
menu.add_cascade(label="Help", menu=helpmenu)
helpmenu.add_command(label="About...", command=About)

app.wm_title("Stamp Recognition Panel")

app.geometry('1200x700')




app.mainloop()

#root.mainloop()
