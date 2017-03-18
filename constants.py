"""
Constants used in white balance neural network
"""

import os

root_dir = '/Users/xi/Downloads/data/white_balance/'
data_dir = os.path.join(root_dir, 'data')
log_dir = os.path.join(root_dir, 'log')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
check_ptr_dir = os.path.join(root_dir, 'save')
if not os.path.exists(check_ptr_dir):
    os.makedirs(check_ptr_dir)
    
IMAGE_SIZE = [1460, 2193, 3]
PATCH_SIZE = [47, 47, 3]

# the size of the data set using under debug mode
DEBUG_DATA_SIZE = 5

