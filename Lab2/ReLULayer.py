import numpy as np

class ReLULayer(object):
    def forward(self, input_data):
        self.input_data = input_data
        output_data = np.where(self.input_data >= 0, input_data, 0)
        return output_data
    
    def backward(self, top_diff):
        bottom_diff = np.where(self.input_data >= 0, top_diff, 0)
        return bottom_diff
