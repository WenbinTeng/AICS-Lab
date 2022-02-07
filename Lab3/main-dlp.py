import numpy as np
import scipy
import scipy.io
from PIL import Image

import pycnml

class DlpModel:
    def __init__(self, quant_param_path, train_model_path) -> None:
        self.net = pycnml.CnmlNet()
        self.image_mean = np.array([123.680, 116.779, 103.939])
        self.input_quant_params = []
        self.filter_quant_params = []
        self.quant_param_path = quant_param_path
        self.train_model_path = train_model_path

    def build_model(self):
        params = np.load(self.quant_param_path)
        input_params = params['input']
        filter_params = params['filter']

        for i in range(0, len(input_params), 2):
            self.input_quant_params.append(pycnml.QuantParam(int(input_params[i]), float(input_params[i + 1])))

        for i in range(0, len(filter_params), 2):
            self.filter_quant_params.append(pycnml.QuantParam(int(filter_params[i]), float(filter_params[i + 1])))
    
        self.net.setInputShape(1, 3, 224, 224)
        self.net.createConvLayer('cl11', 64, 3, 1, 1, 1, self.input_quant_params[0])
        self.net.createReLuLayer('rl11')
        self.net.createConvLayer('cl12', 64, 3, 1, 1, 1, self.input_quant_params[1])
        self.net.createReLuLayer('rl12')
        self.net.createPoolingLayer('mpl1', 2, 2)
        self.net.createConvLayer('cl21', 128, 3, 1, 1, 1, self.input_quant_params[2])
        self.net.createReLuLayer('rl21')
        self.net.createConvLayer('cl22', 128, 3, 1, 1, 1, self.input_quant_params[3])
        self.net.createReLuLayer('rl22')
        self.net.createPoolingLayer('mpl2', 2, 2)
        self.net.createConvLayer('cl31', 256, 3, 1, 1, 1, self.input_quant_params[4])
        self.net.createReLuLayer('rl31')
        self.net.createConvLayer('cl32', 256, 3, 1, 1, 1, self.input_quant_params[5])
        self.net.createReLuLayer('rl32')
        self.net.createConvLayer('cl33', 256, 3, 1, 1, 1, self.input_quant_params[6])
        self.net.createReLuLayer('rl33')
        self.net.createConvLayer('cl34', 256, 3, 1, 1, 1, self.input_quant_params[7])
        self.net.createReLuLayer('rl34')
        self.net.createPoolingLayer('mpl3', 2, 2)
        self.net.createConvLayer('cl41', 512, 3, 1, 1, 1, self.input_quant_params[8])
        self.net.createReLuLayer('rl41')
        self.net.createConvLayer('cl42', 512, 3, 1, 1, 1, self.input_quant_params[9])
        self.net.createReLuLayer('rl42')
        self.net.createConvLayer('cl43', 512, 3, 1, 1, 1, self.input_quant_params[10])
        self.net.createReLuLayer('rl43')
        self.net.createConvLayer('cl44', 512, 3, 1, 1, 1, self.input_quant_params[11])
        self.net.createReLuLayer('rl44')
        self.net.createPoolingLayer('mpl4', 2, 2)
        self.net.createConvLayer('cl51', 512, 3, 1, 1, 1, self.input_quant_params[12])
        self.net.createReLuLayer('rl51')
        self.net.createConvLayer('cl52', 512, 3, 1, 1, 1, self.input_quant_params[13])
        self.net.createReLuLayer('rl52')
        self.net.createConvLayer('cl53', 512, 3, 1, 1, 1, self.input_quant_params[14])
        self.net.createReLuLayer('rl53')
        self.net.createConvLayer('cl54', 512, 3, 1, 1, 1, self.input_quant_params[15])
        self.net.createReLuLayer('rl54')
        self.net.createPoolingLayer('mpl5', 2, 2)
        self.net.createFlattenLayer('fl', (1, 25088, 1, 1))
        self.net.createMlpLayer('fcl6', 4096, self.input_quant_params[16])
        self.net.createReLuLayer('rl6')
        self.net.createMlpLayer('fcl7', 4096, self.input_quant_params[17])
        self.net.createReLuLayer('rl7')
        self.net.createMlpLayer('fcl8', 1000, self.input_quant_params[18])
        self.net.createSoftmaxLayer('sll', 1)
    
    def load_model(self):
        param = scipy.io.loadmat(self.train_model_path)
        count = 0
        for index in range(self.net.size()):
            if 'fcl' in self.net.getLayerName(index):
                weight, bias = param['layers'][0][index - 1][0][0][0][0]
                weight = np.transpose(np.reshape(weight, [weight.shape[0] * weight.shape[1] * weight.shape[2], weight.shape[3]]), [1, 0]).flatten().astype(np.float64)
                bias = np.reshape(bias, -1).astype(np.float64)
                self.net.loadParams(index, weight, bias, self.filter_quant_params[count])
                count += 1
            elif 'cl' in self.net.getLayerName(index):
                weight, bias = param['layers'][0][index][0][0][0][0]
                weight = np.transpose(weight, [3, 2, 0, 1]).flatten().astype(np.float64)
                bias = np.reshape(bias, -1).astype(np.float64)
                self.net.loadParams(index, weight, bias, self.filter_quant_params[count])
                count += 1

    def load_image(self, image_dir, h, w):
        input_image = np.array(Image.open(image_dir).convert('RGB').resize((h, w))).astype(np.float32)
        input_image = input_image - self.image_mean
        input_image = np.reshape(input_image, [1] + list(input_image.shape))
        input_image = np.transpose(input_image, [0, 3, 1, 2])
        self.input_data = input_image.flatten().astype(np.float64)
        self.net.setInputData(self.input_data)
    
    def evaluate(self):
        self.net.forward()
        label = np.argmax(self.net.getOutputData())
        print('label: %d' % (label))

if __name__ == '__main__':
    model = DlpModel(
        quant_param_path='./Lab3/data/vgg19_quant_param.npz',
        train_model_path='./Lab3/data/vgg19_train_model.mat'
    )
    model.build_model()
    model.load_model()
    model.load_image('./Lab3/data/brown-tabby-cat.jpg', 224, 224)
    model.evaluate()
