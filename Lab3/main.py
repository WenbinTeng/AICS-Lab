from DataLoader import *
from ConvolutionLayer import *
from MaxPoolingLayer import *
from FlattenLayer import *
from Lab2.FullyConnectedLayer import FullyConnectedLayer
from Lab2.ReLULayer import ReLULayer
from Lab2.SoftmaxLossLayer import SoftmaxLossLayer

import numpy as np

class Params:
    PARAM_DIR = './Lab3/data/vgg19_train_model.mat'
    IMAGE_DIR = './Lab3/data/brown-tabby-cat.jpg'

class Model:
    def __init__(self) -> None:
        pass

    def build_model(self):
        self.layers = {}
        self.layers['cl11'] = ConvolutionLayer(3, 3, 64, 1, 1)
        self.layers['rl11'] = ReLULayer()
        self.layers['cl12'] = ConvolutionLayer(3, 64, 64, 1, 1)
        self.layers['rl12'] = ReLULayer()
        self.layers['mpl1'] = MaxPoolingLayer(2, 2)
        self.layers['cl21'] = ConvolutionLayer(3, 64, 128, 1, 1)
        self.layers['rl21'] = ReLULayer()
        self.layers['cl22'] = ConvolutionLayer(3, 128, 128, 1, 1)
        self.layers['rl22'] = ReLULayer()
        self.layers['mpl2'] = MaxPoolingLayer(2, 2)
        self.layers['cl31'] = ConvolutionLayer(3, 128, 256, 1, 1)
        self.layers['rl31'] = ReLULayer()
        self.layers['cl32'] = ConvolutionLayer(3, 256, 256, 1, 1)
        self.layers['rl32'] = ReLULayer()
        self.layers['cl33'] = ConvolutionLayer(3, 256, 256, 1, 1)
        self.layers['rl33'] = ReLULayer()
        self.layers['cl34'] = ConvolutionLayer(3, 256, 256, 1, 1)
        self.layers['rl34'] = ReLULayer()
        self.layers['mpl3'] = MaxPoolingLayer(2, 2)
        self.layers['cl41'] = ConvolutionLayer(3, 256, 512, 1, 1)
        self.layers['rl41'] = ReLULayer()
        self.layers['cl42'] = ConvolutionLayer(3, 512, 512, 1, 1)
        self.layers['rl42'] = ReLULayer()
        self.layers['cl43'] = ConvolutionLayer(3, 512, 512, 1, 1)
        self.layers['rl43'] = ReLULayer()
        self.layers['cl44'] = ConvolutionLayer(3, 512, 512, 1, 1)
        self.layers['rl44'] = ReLULayer()
        self.layers['mpl4'] = MaxPoolingLayer(2, 2)
        self.layers['cl51'] = ConvolutionLayer(3, 512, 512, 1, 1)
        self.layers['rl51'] = ReLULayer()
        self.layers['cl52'] = ConvolutionLayer(3, 512, 512, 1, 1)
        self.layers['rl52'] = ReLULayer()
        self.layers['cl53'] = ConvolutionLayer(3, 512, 512, 1, 1)
        self.layers['rl53'] = ReLULayer()
        self.layers['cl54'] = ConvolutionLayer(3, 512, 512, 1, 1)
        self.layers['rl54'] = ReLULayer()
        self.layers['mpl5'] = MaxPoolingLayer(2, 2)
        self.layers['fl']   = FlattenLayer([512, 7, 7], [512 * 7 * 7])
        self.layers['fcl6'] = FullyConnectedLayer(25088, 4096)
        self.layers['rl6']  = ReLULayer()
        self.layers['fcl7'] = FullyConnectedLayer(4096, 4096)
        self.layers['rl7']  = ReLULayer()
        self.layers['fcl8'] = FullyConnectedLayer(4096, 1000)
        self.layers['sll']  = SoftmaxLossLayer()

        self.data_loader = DataLoader(Params.PARAM_DIR)

    def load_model(self, param_dir):
        param = scipy.io.loadmat(param_dir)
        layer = list(self.layers.keys())
        for index in range(len(layer)):
            if 'fcl' in layer[index]:
                weight, bias = param['layers'][0][index - 1][0][0][0][0]
                weight = np.reshape(weight, [weight.shape[0] * weight.shape[1] * weight.shape[2], weight.shape[3]])
                bias = np.reshape(bias, -1)
                self.layers[layer[index]].load_param(weight, bias)
            elif 'cl' in layer[index]:
                weight, bias = param['layers'][0][index][0][0][0][0]
                weight = np.transpose(weight, [2, 0, 1, 3])
                bias = np.reshape(bias, -1)
                self.layers[layer[index]].load_param(weight, bias)

    def load_image(self, image_dir, h, w):
        return self.data_loader.load_image(image_dir, h, w)

    def save_image(self, image_dir, image):
        self.data_loader.save_image(image_dir, image)

    def forward(self, input_data):
        data = input_data
        for layer in self.layers.values():
            data = layer.forward(data)
        return data

    def evaluate(self, input_data):
        prob = self.forward(input_data)
        pred = np.argmax(prob[0])
        print('id=%d, prob=%d' % (pred, prob[0, pred]))

if __name__ == '__main__':
    model = Model()
    model.build_model()
    model.load_model(Params.PARAM_DIR)
    image = model.load_image(Params.IMAGE_DIR, 224, 224)
    model.evaluate(image)
