import numpy as np

class StyleLossLayer(object):
    def forward(self, input_layer, style_layer):
        self.input_layer = input_layer
        self.style_layer = style_layer
        self.input_layer_reshape = np.reshape(self.input_layer, [self.input_layer.shape[0], self.input_layer.shape[1], -1])
        self.style_layer_reshape = np.reshape(self.style_layer, [self.style_layer.shape[0], self.style_layer.shape[1], -1])
        self.input_gram = np.array([np.dot(self.input_layer_reshape[index_n, :, :], self.input_layer_reshape[index_n, :, :].T) for index_n in range(input_layer.shape[0])])
        self.style_gram = np.dot(self.style_layer_reshape[0, :, :], self.style_layer_reshape[0, :, :].T)
        n, c, h, w = self.input_layer.shape
        loss = 1.0 / (4 * n * c * c * h * h * w * w) * np.sum(np.square(self.input_gram - self.style_gram))
        return loss

    def backWard(self):
        n, c, h, w = self.input_layer.shape
        bottom_diff = np.zeros([n, c, h * w])
        for index_n in range(n):
            bottom_diff[index_n, :, :] = 1.0 / (n * c * c * h * h * w * w) * np.dot(self.input_gram[index_n, :, :] - self.style_gram, self.input_layer_reshape[index_n, :, :])
        bottom_diff = np.reshape(bottom_diff, self.input_layer.shape)
        return bottom_diff
