import numpy as np

class SoftmaxLossLayer(object):
    def forward(self, input_data):
        input_max = np.max(input_data, axis=1, keepdims=True)
        input_exp = np.exp(input_data - input_max)
        self.prob = np.divide(input_exp, np.sum(input_exp, axis=1, keepdims=True))
        return self.prob

    def backward(self):
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff

    def get_loss(self, label):
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss
