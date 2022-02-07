import numpy as np
from enum import Enum
import struct
import os

import pycnml

class Params:
    MNIST_DIR = './Lab2/data'
    TEST_IMAGE = 't10k-images.idx3-ubyte'
    TEST_LABEL = 't10k-labels.idx1-ubyte'

class ReadDataType(Enum):
    image = 0
    label = 1

class DlpModel:
    def __init__(
        self,
        batch_size = 10000,
        input_size = 784,
        hidden1 = 32,
        hidden2 = 16,
        output_size = 10,
        quant_param_path = '',
        train_model_path = ''
    ) -> None:
        self.net = pycnml.CnmlNet()
        self.input_quant_params = []
        self.filter_quant_params = []
        self.quant_param_path = quant_param_path
        self.train_model_path = train_model_path

        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_size = output_size

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

    def load_data(self):
        test_images = self.load_raw_data(file_dir=os.path.join(Params.MNIST_DIR, Params.TEST_IMAGE), read_type=ReadDataType.image)
        test_labels = self.load_raw_data(file_dir=os.path.join(Params.MNIST_DIR, Params.TEST_LABEL), read_type=ReadDataType.label)
        self.test_data = np.append(test_images, test_labels, axis=1)

    def build_model(self):
        params = np.load(self.quant_param_path)
        input_params = params['input']
        filter_params = params['filter']

        for i in range(0, len(input_params), 2):
            self.input_quant_params.append(pycnml.QuantParam(int(input_params[i]), float(input_params[i + 1])))

        for i in range(0, len(filter_params), 2):
            self.filter_quant_params.append(pycnml.QuantParam(int(filter_params[i]), float(filter_params[i + 1])))

        self.net.setInputShape(self.batch_size, self.input_size, 1, 1)
        self.net.createMlpLayer('fcl1', self.hidden1, self.input_quant_params[0])
        self.net.createReLuLayer('rl1')
        self.net.createMlpLayer('fcl2', self.hidden2, self.input_quant_params[1])
        self.net.createReLuLayer('rl2')
        self.net.createMlpLayer('fcl3', self.output_size, self.input_quant_params[2])
        self.net.createSoftmaxLayer('sll', axis=1)
        
    def load_model(self):
        params = np.load(self.train_model_path, allow_pickle=True).item()
        w1 = np.transpose(params['w1'], [1, 0]).flatten().astype(np.float64)
        b1 = params['b1'].flatten().astype(np.float64)
        w2 = np.transpose(params['w2'], [1, 0]).flatten().astype(np.float64)
        b2 = params['b2'].flatten().astype(np.float64)
        w3 = np.transpose(params['w3'], [1, 0]).flatten().astype(np.float64)
        b3 = params['b3'].flatten().astype(np.float64)
        self.net.loadParams(0, w1, b1, self.filter_quant_params[0])
        self.net.loadParams(2, w2, b2, self.filter_quant_params[1])
        self.net.loadParams(4, w3, b3, self.filter_quant_params[2])

    def evaluate(self):
        pred_results = np.zeros([self.test_data.shape[0]])
        for index in range(self.test_data.shape[0] // self.batch_size):
            batch_image = self.test_data[index * self.batch_size : (index + 1) * self.batch_size, :-1].flatten().tolist()
            self.net.setInputData(batch_image)
            self.net.forward()
            prob = np.array(self.net.getOutputData()).reshape([self.batch_size, self.output_size])
            pred_label = np.argmax(prob, axis=1)
            pred_results[index * self.batch_size : (index + 1) * self.batch_size] = pred_label
            
        accuracy = np.mean(pred_results == self.test_data[:, -1])
        print('accuacy %.6f' % accuracy)



if __name__ == '__main__':
    h1, h2 = 32, 16
    model = DlpModel(
        hidden1=h1,
        hidden2=h2,
        quant_param_path='./Lab2/data/mnist_mlp_quant_param.npz',
        train_model_path='./Lab2/data/mnist_mlp_train_model.npy'
    )
    model.load_data()
    model.build_model()
    model.load_model()
    model.evaluate()
