import numpy as np

class ContentLossLayer(object):
    def forward(self, input_layer, content_layer):
        self.input_layer = input_layer
        self.content_layer = content_layer
        n, c, h, w = self.input_layer.shape
        loss = 1.0 / (2 * n * c * h * w) * np.sum(np.square(self.input_layer - self.content_layer))
        return loss

    def backWard(self):
        n, c, h, w = self.input_layer.shape
        diff = 1.0 / (n * c * h * w) * (self.input_layer - self.content_layer)
        return diff
