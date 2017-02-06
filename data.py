import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from constants import *

def _pre_process_image(image):
    # Convert to UV space
    # (log(r/b), log(g/b)) ???
    image = np.stack((np.ma.log(image[:,:,0] / image[:,:,2]), 
                      np.ma.log(image[:,:,1] / image[:,:,2]))).T
    return image

def load_data(data_set):
    # Load images
    print("Loading images...")
    images = []
    img_dir = os.path.join(data_dir, data_set, 'MAT')
    for image_file in os.listdir(img_dir):
        # image = cv2.imread(os.path.join(img_dir, image_file))
#         image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
#         image = _pre_process_image(image)
        image = (scipy.io.loadmat(os.path.join(img_dir, image_file)))['img']
        images.append(image)
    
    # Load labels
    print("Loading labels...")
    csv_file = open(os.path.join(data_dir, data_set + '.csv'), 'r')
    csv_file_object = csv.reader(csv_file)
    data = []
    for row in csv_file_object:
        data.append(row[-2:])
    return images, data
    
def hyp_net_pre_process(X):
    # extract mean
    pass
