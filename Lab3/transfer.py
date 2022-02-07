from DataLoader import *
from ConvolutionLayer import *
from ReLULayer import *
from MaxPoolingLayer import *
from FlattenLayer import *
from ContentLossLayer import *
from StyleLossLayer import *
from AdamOptimizer import *

import numpy as np

class InputParams:
    MODEL_PARAMS_DIR = './Lab3/data/vgg19_train_model.mat'
    STYLE_IMAGE_DIR = './Lab3/data/starry-night.png'
    CONTENT_IMAGE_DIR= './Lab3/data/venice-city.png'
    SAVE_DIR = './Lab3/data/transfered-image.png'
    IMAGE_HEIGHT, IMAGE_WIDTH = 100, 100

class ModelParams:
    CONTENT_LOSS_LAYERS = ['rl33']
    STYLE_LOSS_LAYERS = ['rl11', 'rl21', 'rl31', 'rl41', 'rl51']
    NOISE = 0.5
    ALPHA, BETA = 1, 500
    TRAIN_STEP = 2001
    LEARNING_RATE = 1.0

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

        self.content_loss_layer = ContentLossLayer()
        self.style_loss_layer = StyleLossLayer()
        self.adam_optimizer = AdamOptimizer(ModelParams.LEARNING_RATE, [InputParams.IMAGE_HEIGHT, InputParams.IMAGE_WIDTH])

        self.data_loader = DataLoader(InputParams.MODEL_PARAMS_DIR)

    def load_model(self, param_dir):
        param = scipy.io.loadmat(param_dir)
        layer = list(self.layers.keys())
        for index in range(len(layer)):
            if 'cl' in layer[index]:
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

    def backward(self, top_diff, layer_name):
        diff = top_diff
        for index in range(list.index(list(self.layers.keys()), layer_name), -1, -1):
            diff = list(self.layers.values())[index].backward(diff)
        return diff
    
    def transfer(self, content_image, style_image):

        transfer_image = np.random.uniform(-20, 20, content_image.shape) * ModelParams.NOISE + content_image * (1 - ModelParams.NOISE)

        model.forward(content_image)
        content_layers = {}
        for layer_name in ModelParams.CONTENT_LOSS_LAYERS:
            content_layers[layer_name] = model.layers[layer_name].get_forward_data()

        model.forward(style_image)
        style_layers = {}
        for layer_name in ModelParams.STYLE_LOSS_LAYERS:
            style_layers[layer_name] = model.layers[layer_name].get_forward_data()

        for step in range(ModelParams.TRAIN_STEP):

            content_loss = np.array([])
            style_loss = np.array([])
            content_diff = np.zeros(transfer_image.shape)
            style_diff = np.zeros(transfer_image.shape)
            
            model.forward(transfer_image)
            transfer_layers = {}
            for layer_name in ModelParams.CONTENT_LOSS_LAYERS + ModelParams.STYLE_LOSS_LAYERS:
                transfer_layers[layer_name] = model.layers[layer_name].get_forward_data()
            
            for layer_name in ModelParams.CONTENT_LOSS_LAYERS:
                loss = model.content_loss_layer.forward(transfer_layers[layer_name], content_layers[layer_name])
                content_loss = np.append(content_loss, loss)
                dloss = model.content_loss_layer.backWard()
                content_diff += model.backward(dloss, layer_name) / len(ModelParams.CONTENT_LOSS_LAYERS)

            for layer_name in ModelParams.STYLE_LOSS_LAYERS:
                loss = model.style_loss_layer.forward(transfer_layers[layer_name], style_layers[layer_name])
                style_loss = np.append(style_loss, loss)
                dloss = model.style_loss_layer.backWard()
                style_diff += model.backward(dloss, layer_name) / len(ModelParams.STYLE_LOSS_LAYERS)
            
            total_loss = ModelParams.ALPHA * np.mean(content_loss) + ModelParams.BETA * np.mean(style_loss)
            image_diff = ModelParams.ALPHA *         content_diff  + ModelParams.BETA *         style_diff
            
            transfer_image = model.adam_optimizer.update_param(transfer_image, image_diff)

            print('step=%d loss=%.6f' % (step, total_loss))

if __name__ == '__main__':
    model = Model()
    model.build_model()
    model.load_model(InputParams.MODEL_PARAMS_DIR)
    content_image = model.load_image(InputParams.CONTENT_IMAGE_DIR, InputParams.IMAGE_HEIGHT, InputParams.IMAGE_WIDTH)
    style_image = model.load_image(InputParams.STYLE_IMAGE_DIR, InputParams.IMAGE_HEIGHT, InputParams.IMAGE_WIDTH)
    transfer_image = model.transfer(content_image, style_image)
    model.save_image(InputParams.SAVE_DIR, transfer_image)
