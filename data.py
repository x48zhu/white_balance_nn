
import csv
import numpy as np

data_dir = '/Users/xi/Downloads/data/white_balance'

def load_data(file):
    csv_file = open(data_dir + file, 'rb')
    csv_file_object = csv.reader(csv_file)
    _y = []
    for row in csv_file_object:
        _y.append(row[-2:])

def pre_process():
    # resize image?
    # convert to LU space

def hyp_net_pre_process(X):
    # extract mean