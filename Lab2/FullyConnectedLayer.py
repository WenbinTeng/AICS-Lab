import numpy as np

class FullyConnectedLayer(object):
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weight = np.random.normal(loc=0.0, scale=0.01, size=(self.num_inputs, self.num_outputs))
        self.bias = np.zeros([1, self.num_outputs])

    def forward(self, input_data):
        self.input_data = input_data
        self.output_data = np.matmul(self.input_data, self.weight) + self.bias
        return self.output_data

    def backward(self, top_diff):
        self.d_weight = np.matmul(np.transpose(self.input_data), top_diff)
        self.d_bias = np.matmul(np.ones((1, self.input_data.shape[0])), top_diff)
        self.bottom_diff = np.matmul(top_diff, np.transpose(self.weight))
        return self.bottom_diff

    def update_param(self, learning_rate):
        self.weight = self.weight - learning_rate * self.d_weight
        self.bias = self.bias - learning_rate * self.d_bias

    def load_param(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def save_param(self):
        return (self.weight, self.bias)
