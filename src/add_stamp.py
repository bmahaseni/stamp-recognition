from Tkinter import *
import csv

class Window:       
    def __init__(self, master):     
        self.filename=""
        csvfile=Label(root, text="File").grid(row=1, column=0)
        bar=Entry(master).grid(row=1, column=1) 
    
        #Buttons  
        y=7
        self.cbutton= Button(root, text="OK", command=self.process_csv)
        y+=1
        self.cbutton.grid(row=10, column=3, sticky = W + E)
        self.bbutton= Button(root, text="Browse", command=self.browsecsv)
        self.bbutton.grid(row=1, column=3)
    
    def browsecsv(self):
        from tkFileDialog import askopenfilename
    
        Tk().withdraw() 
        self.filename = askopenfilename()
    
    def process_csv(self):
        if self.filename:
            with open(self.filename, 'rb') as csvfile:
                logreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                rownum=0
    
                for row in logreader:    
                    NumColumns = len(row)        
                    rownum += 1
    
                Matrix = [[0 for x in xrange(NumColumns)] for x in xrange(rownum)] 

root = Tk()
root.wm_title("Open Stamp")

window=Window(root)
root.mainloop()  