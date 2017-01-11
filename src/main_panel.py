from Tkinter import *
from PIL import Image, ImageTk
from dataset import Dataset
dataset = Dataset('./dataset', ['China', 'Japan', 'Malaysia', 'Singapore', 'South_Korea'], ['2010', '2011', '2012', '2013', '2014', '2015' ])

dataset.generate_date()




root = Tk()
root.wm_title("Stamp Recognition Panel")

root.geometry('400x600')

def callback():
    print "Add!"

b = Button(root, text="Add", command=callback)
b.pack()


b = Button(root, text="re-Train", command=callback)
b.pack()

for i in xrange(20):

    image = Image.open(dataset.instances[i].file_path)
    image.thumbnail((98, 98), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)
    text_str = 'Country:' + dataset.instances[i].country + ' Year:' + dataset.instances[i].year
    label = Label(image=photo, text=text_str,compound=LEFT)
    label.image = photo # keep a reference!
    label.pack()


root.mainloop()