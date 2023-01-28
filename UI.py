import tkinter as tk
from tkinter import filedialog 
from time import sleep
import os

class SimpleUI:
    def __init__(self):
        # create the root window
        self.path = ''
        self.root = tk.Tk()
        self.root.title('Select a Folder for Inference')
        self.root.resizable(False, False)
        self.root.geometry('300x150')
        tk.Label(self.root, text="Drill Bit Classifier", font= ('Helvetica 14 bold')).pack()
        self.root.wm_attributes('-toolwindow', 'True')

    def select_path(self):
        self.path = filedialog.askdirectory(
            title="Select directory")
        if (self.path!=''):
            self.root.destroy()

    def choose_directory(self):
        # open button
        tk.Button(
            self.root,
            text='Open a Folder',
            command=self.select_path
        ).pack(expand=True)
        tk.mainloop()
        
        if self.path == '':
            sleep(2)
            return self.choose_directory()
        else:
            return self.path.replace('/', os.sep)

