"""
Constants used in white balance neural network
"""

import os

data_dir = '/Users/xi/Downloads/data/white_balance/data'
log_dir = '/Users/xi/Downloads/data/white_balance/simple/log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
