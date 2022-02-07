import numpy as np
from enum import Enum
import struct
import os

class Params:
    MNIST_DIR = './Lab2/data'
    TRAIN_IMAGE = 'train-images.idx3-ubyte'
    TRAIN_LABEL = 'train-labels.idx1-ubyte'
    TEST_IMAGE = 't10k-images.idx3-ubyte'
    TEST_LABEL = 't10k-labels.idx1-ubyte'

class ReadDataType(Enum):
    image = 0
    label = 1

class DataLoader:
    def load_raw_data(self, file_dir, read_type):
        bin_file = open(file_dir, 'rb')
        bin_data = bin_file.read()
        bin_file.close()

        if (read_type == ReadDataType.image):
            fmt_header = '>iiii'
            _, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data)
        else:
            fmt_header = '>ii'
            _, num_images = struct.unpack_from(fmt_header, bin_data, 0)
            num_rows = 1
            num_cols = 1
        
        ret_size = num_images * num_rows * num_cols
        fmt_data = '>' + str(ret_size) + 'B'
        ret_data = struct.unpack_from(fmt_data, bin_data, struct.calcsize(fmt_header))
        ret_data = np.reshape(ret_data, [num_images, num_rows * num_cols])

        return ret_data
    
    def load_all_data(self):
        train_images = self.load_raw_data(file_dir=os.path.join(Params.MNIST_DIR, Params.TRAIN_IMAGE), read_type=ReadDataType.image)
        train_labels = self.load_raw_data(file_dir=os.path.join(Params.MNIST_DIR, Params.TRAIN_LABEL), read_type=ReadDataType.label)
        test_images = self.load_raw_data(file_dir=os.path.join(Params.MNIST_DIR, Params.TEST_IMAGE), read_type=ReadDataType.image)
        test_labels = self.load_raw_data(file_dir=os.path.join(Params.MNIST_DIR, Params.TEST_LABEL), read_type=ReadDataType.label)
        self.train_data = np.append(train_images, train_labels, axis=1)
        self.test_data = np.append(test_images, test_labels, axis=1)
