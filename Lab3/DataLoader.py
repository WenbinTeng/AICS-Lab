import numpy as np
import scipy
import scipy.io
from PIL import Image

class DataLoader:
    def __init__(self, param_dir) -> None:
        self.image_mean = np.mean(scipy.io.loadmat(param_dir)['normalization'][0][0][0], axis=(0, 1))

    def load_image(self, image_dir, h, w):
        input_image = np.array(Image.open(image_dir).convert('RGB').resize((h, w))).astype(np.float32)
        input_image = input_image - self.image_mean
        input_image = np.reshape(input_image, [1] + list(input_image.shape))
        input_image = np.transpose(input_image, [0, 3, 1, 2])
        return input_image

    def save_image(self, image_dir, image):
        image = np.transpose(image, [0, 2, 3, 1])
        image = image[0] + self.image_mean
        image = np.clip(image, 0, 255).astype(np.uint8)
        image = Image.fromarray(image)
        image.save(image_dir)
