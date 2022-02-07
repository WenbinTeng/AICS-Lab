import numpy as np

class FlattenLayer(object):
    def __init__(
        self,
        input_shape,
        output_shape,
    ) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input_data):
        self.input_data = np.transpose(input_data, [0, 2, 3, 1])
        self.output_data = self.input_data.reshape([self.input_data.shape[0]] + list(self.output_shape))
        return self.output_data
