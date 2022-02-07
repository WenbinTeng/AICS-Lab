import numpy as np

class ReLULayer(object):
    def forward(self, input_data):
        self.input_data = input_data
        self.output_data = np.where(self.input_data >= 0, input_data, 0)
        return self.output_data
    
    def backward(self, top_diff):
        self.bottom_diff = np.where(self.input_data >= 0, top_diff, 0)
        return self.bottom_diff

    def get_forward_data(self):
        return self.output_data

    def get_backward_diff(self):
        return self.bottom_diff
