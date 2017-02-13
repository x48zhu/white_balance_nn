import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np

def load_data(data_set):
    features = []
    data = []
    # Load features and labels
    csv_file = open(os.path.join(data_dir, data_set + '.csv'), 'r')
    csv_file_object = csv.reader(csv_file)

    for row in csv_file_object:
        features.append(row[:8])
        data.append(row[-2:])
    return features, data