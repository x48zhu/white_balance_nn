"""
Constants used in white balance neural network
"""

data_dir = '/Users/xi/Downloads/data/white_balance'
log_dir = '/Users/xi/Downloads/data/white_balance/log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
check_ptr_dir = '/Users/xi/Downloads/data/white_balance/save'
if not os.path.exists(check_ptr_dir):
    os.makedirs(check_ptr_dir)
