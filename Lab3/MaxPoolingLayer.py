import numpy as np

class MaxPoolingLayer(object):
    def __init__(
        self,
        kernel_size,
        stride
    ) -> None:
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input_data):
        self.input_data = input_data
        h_out = (self.input_data.shape[2] - self.kernel_size) // self.stride + 1
        w_out = (self.input_data.shape[3] - self.kernel_size) // self.stride + 1
        self.output_data = np.zeros([self.input_data.shape[0], self.input_data.shape[1], h_out, w_out])
        for index_n in range(self.input_data.shape[0]):
            for index_c in range(self.input_data.shape[1]):
                for index_h in range(h_out):
                    for index_w in range(w_out):
                        hs = index_h * self.stride
                        ws = index_w * self.stride
                        self.output_data[index_n, index_c, index_h, index_w] = np.max(self.input_data[index_n, index_c, hs : hs + self.kernel_size, ws : ws + self.kernel_size])
        
        return self.output_data

    def backward(self, top_diff):
        self.bottom_diff = np.zeros(self.input_data.shape)
        for index_n in range(top_diff.shape[0]):
            for index_c in range(top_diff.shape[1]):
                for index_h in range(top_diff.shape[2]):
                    for index_w in range(top_diff.shape[3]):
                        hs = index_h * self.stride
                        ws = index_w * self.stride
                        max_index = np.unravel_index(np.argmax(self.input_data[index_n, index_c, hs : hs + self.kernel_size, ws : ws + self.kernel_size]), (self.kernel_size, self.kernel_size))
                        self.bottom_diff[index_n, index_c, hs + max_index[0], ws + max_index[1]] = top_diff[index_n, index_c, index_h, index_w]

        return self.bottom_diff

    def get_forward_data(self):
        return self.output_data

    def get_backward_diff(self):
        return self.bottom_diff
