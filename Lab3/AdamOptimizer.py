import numpy as np

class AdamOptimizer(object):
    def __init__(self, learning_rate, diff_shape) -> None:
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.learning_rate = learning_rate
        self.mt = np.zeros(diff_shape)
        self.vt = np.zeros(diff_shape)
        self.step = 0

    def update_param(self, input_data, grad):
        self.step += 1
        self.mt = self.beta1 * self.mt + (1 - self.beta1) * grad
        self.vt = self.beta2 * self.vt + (1 - self.beta2) * np.square(grad)
        mt_hat = self.mt / (1 - np.power(self.beta1, self.step))
        vt_hat = self.vt / (1 - np.power(self.beta2, self.step))
        output_data = input_data - self.learning_rate * mt_hat / (np.sqrt(vt_hat) + self.eps)
        return output_data
