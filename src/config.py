import os
import random
import numpy as np

# Cố định Seed chuẩn môn học DTA301
RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"-> Đã thiết lập RANDOM_SEED = {seed}")

# Định nghĩa các đường dẫn thư mục
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')