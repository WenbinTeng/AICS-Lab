from DataLoader import *
from FullyConnectedLayer import *
from ReLULayer import *
from SoftmaxLossLayer import *

import numpy as np

class Model:
    def __init__(
        self,
        batch_size = 100,
        input_size = 784,
        hidden1 = 32,
        hidden2 = 16,
        output_size = 10,
        learning_rate = 0.01,
        max_epoch = 2,
        print_iter = 100
    ) -> None:
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.print_iter = print_iter

    def load_data(self):
        loader = DataLoader()
        loader.load_all_data()
        self.train_data = loader.train_data
        self.test_data = loader.test_data

    def shuffle_data(self):
        np.random.shuffle(self.train_data)
        np.random.shuffle(self.test_data)

    def build_model(self):
        self.fcl1 = FullyConnectedLayer(self.input_size, self.hidden1)
        self.fcl2 = FullyConnectedLayer(self.hidden1, self.hidden2)
        self.fcl3 = FullyConnectedLayer(self.hidden2, self.output_size)
        self.rl1 = ReLULayer()
        self.rl2 = ReLULayer()
        self.sll = SoftmaxLossLayer()

    def forward(self, input_data):
        fcl1_out = self.fcl1.forward(input_data)
        rl1_out = self.rl1.forward(fcl1_out)
        fcl2_out = self.fcl2.forward(rl1_out)
        rl2_out = self.rl2.forward(fcl2_out)
        fcl3_out = self.fcl3.forward(rl2_out)
        sll_out = self.sll.forward(fcl3_out)
        return sll_out

    def backward(self):
        d_loss = self.sll.backward()
        d_fcl3 = self.fcl3.backward(d_loss)
        d_rl2 = self.rl2.backward(d_fcl3)
        d_fcl2 = self.fcl2.backward(d_rl2)
        d_rl1 = self.rl1.backward(d_fcl2)
        _ = self.fcl1.backward(d_rl1)

    def update(self, learning_rate):
        self.fcl1.update_param(learning_rate)
        self.fcl2.update_param(learning_rate)
        self.fcl3.update_param(learning_rate)

    def save_model(self, param_dir):
        param_dic = {}
        param_dic['w1'], param_dic['b1'] = self.fcl1.save_param()
        param_dic['w2'], param_dic['b2'] = self.fcl2.save_param()
        param_dic['w3'], param_dic['b3'] = self.fcl3.save_param()
        np.save(param_dir, param_dic)

    def load_model(self, param_dir):
        param_dic = np.load(param_dir, allow_pickle=True).item()
        self.fcl1.load_param(param_dic['w1'], param_dic['b1'])
        self.fcl2.load_param(param_dic['w2'], param_dic['b2'])
        self.fcl3.load_param(param_dic['w3'], param_dic['b3'])

    def train(self):
        for index_epoch in range(self.max_epoch):
            for index_batch in range(self.train_data.shape[0] // self.batch_size):
                batch_image = self.train_data[index_batch * self.batch_size : (index_batch + 1) * self.batch_size, :-1]
                batch_label = self.train_data[index_batch * self.batch_size : (index_batch + 1) * self.batch_size,  -1]
                self.forward(batch_image)
                loss = self.sll.get_loss(batch_label)
                self.backward()
                self.update(self.learning_rate)
                
                if index_batch // self.print_iter == 0:
                    print('epoch %d, batch %d, loss %.6f' % (index_epoch, index_batch, loss))

            self.shuffle_data()

    def evaluate(self):
        pred_results = np.zeros([self.test_data.shape[0]])
        for index in range(self.test_data.shape[0] // self.batch_size):
            batch_image = self.test_data[index * self.batch_size : (index + 1) * self.batch_size, :-1]
            prob = self.forward(batch_image)
            pred_label = np.argmax(prob, axis=1)
            pred_results[index * self.batch_size : (index + 1) * self.batch_size] = pred_label
            
        accuracy = np.mean(pred_results == self.test_data[:, -1])
        print('accuacy %.6f' % accuracy)



if __name__ == '__main__':
    h1, h2, e = 32, 16, 10
    model = Model(hidden1=h1, hidden2=h2, max_epoch=e)
    model.load_data()
    model.build_model()
    model.train()
    model.save_model('./Lab2/data/mnist_mlp_train_model.npy')
    model.load_model('./Lab2/data/mnist_mlp_train_model.npy')
    model.evaluate()
