import numpy as np

class ConvolutionLayer(object):
    def __init__(
        self,
        kernel_size,
        channel_input,
        channel_output,
        padding,
        stride,
    ) -> None:
        self.kernel_size = kernel_size
        self.channel_input = channel_input
        self.channel_output = channel_output
        self.padding = padding
        self.stride = stride
        self.weight = np.random.normal(loc=0.0, scale=0.01, size=(self.channel_input, self.kernel_size, self.kernel_size, self.channel_output))
        self.bias = np.zeros([1, self.channel_output])

    def load_param(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def forward(self, input_data):
        self.input_data = input_data
        h = self.input_data.shape[2] + self.padding * 2
        w = self.input_data.shape[3] + self.padding * 2
        self.input_pad = np.zeros([self.input_data.shape[0], self.input_data.shape[1], h, w])
        self.input_pad[:, :, self.padding : self.padding + self.input_data.shape[2], self.padding : self.padding + self.input_data.shape[3]] = self.input_data
        h_out = (h - self.kernel_size) // self.stride + 1
        w_out = (w - self.kernel_size) // self.stride + 1
        self.output_data = np.zeros([self.input_data.shape[0], self.channel_output, h_out, w_out])
        for index_n in range(self.input_data.shape[0]):
            for index_c in range(self.channel_output):
                for index_h in range(h_out):
                    for index_w in range(w_out):
                        hs = index_h * self.stride
                        ws = index_w * self.stride
                        self.output_data[index_n, index_c, index_h, index_w] = np.sum(self.input_pad[index_n, :, hs : hs + self.kernel_size, ws : ws + self.kernel_size] * self.weight[:, :, :, index_c]) + self.bias[index_c]
        
        return self.output_data

    def backward(self, top_diff):
        self.d_weight = np.zeros(self.weight.shape)
        self.d_bias = np.zeros(self.bias.shape)
        self.bottom_diff = np.zeros(self.input_pad.shape)
        for index_n in range(top_diff.shape[0]):
            for index_c in range(top_diff.shape[1]):
                for index_h in range(top_diff.shape[2]):
                    for index_w in range(top_diff.shape[3]):
                        hs = index_h * self.stride
                        ws = index_w * self.stride
                        self.d_weight[:, :, :, index_c] += top_diff[index_n, index_c, index_h, index_w] * self.input_pad[index_n, :, hs : hs + self.kernel_size, ws : ws + self.kernel_size]
                        self.d_bias[index_c] += top_diff[index_n, index_c, index_h, index_w]
                        self.bottom_diff[index_n, :, hs : hs + self.kernel_size, ws : ws + self.kernel_size] += top_diff[index_n, index_c, index_h, index_w] * self.weight[:, :, :, index_c]
        
        self.bottom_diff = self.bottom_diff[:, :, self.padding : self.padding + self.input_data.shape[2], self.padding : self.padding + self.input_data.shape[3]]
        return self.bottom_diff

    def get_forward_data(self):
        return self.output_data
    
    def get_backward_diff(self):
        return self.bottom_diff
