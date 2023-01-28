import csv  
import os
import random

# class takes a directory and assign all images in each same sub-directory with one label, split them into a training set and a test set, save them in two .csv files
class Annotator:
    def __init__(self, dir, train_ratio=0.8):
        self.directory = dir
        self.train_ratio = train_ratio

    def split_filename_list(self, filelist):
        random.shuffle(filelist)
        train_len = int(len(filelist)*self.train_ratio)
        test_len = len(filelist) - train_len
        train_list = []
        test_list = []
        for i in range(train_len):
            train_list.append(filelist[i])
        
        for j in range(test_len):
            test_list.append(filelist[j+train_len])

        return train_list, test_list

    def combine_same_size(self):
        combined_filelist = []
        dirlist = os.listdir(self.directory)
        dirlist = [s for s in dirlist if 'x' in s]

        while(len(dirlist)!=0):
            dir1 = dirlist.pop()
            filelist1 = os.listdir(dir1) 
            filelist1 = [os.path.join(dir1,s) for s in filelist1]

            idx = dir1.index('x')
            size = dir1[idx-2:idx+3]

            dir2 = [s for s in dirlist if size in s][0]
            filelist2 = os.listdir(dir2)
            filelist2 = [os.path.join(dir2,s) for s in filelist2]
            dirlist.remove(dir2)
            combined_filelist.append((filelist1+filelist2, size))

        return combined_filelist

    def generate_annotation(self):
        # use two file handles
        # store names of images into a list, shuffle, assign ratio of them to one handle, and the rest into another
        curdir = os.getcwd()
        os.chdir(self.directory)
        combined_filelist = self.combine_same_size()

        # remove existing files
        if (os.path.exists('train.csv')):
            os.remove('train.csv')
        if (os.path.exists('test.csv')):
            os.remove('test.csv')
        if (os.path.exists('map.csv')):
            os.remove('map.csv')

        # define file handles
        f_train = open('train.csv', 'a', encoding='UTF8', newline='')
        f_test = open('test.csv', 'a', encoding='UTF8', newline='')
        f_map = open("map.csv", "a")

        writer_map = csv.writer(f_map)
        for i in range(len(combined_filelist)):
            filelist = combined_filelist[i][0]
            train_list, test_list = self.split_filename_list(filelist)
            
            for train in train_list:
                writer = csv.writer(f_train)
                row = [train, i]
                # write the data
                writer.writerow(row)

            for test in test_list:
                writer = csv.writer(f_test)
                row = [test, i]
                # write the data
                writer.writerow(row)
            
            row = [combined_filelist[i][1], i]
            # write the mapping
            writer_map.writerow(row)

        f_train.close()
        f_test.close()
        f_map.close()
        os.chdir(curdir)
