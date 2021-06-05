import tensorflow as tf


class ScalePreprocessor:
    def __init__(self, scale=1.0, norm=255.0):
        # store the Red, Green, and Blue channel averages across a training set
        self.scale = scale
        self.norm = norm

    def preprocess(self, image):
        # split the image into its respective Red, Green, and Blue channels
        scaled_image = self.scale * tf.cast(image, tf.float32) / self.norm

        # subtract the means for each channel

        # merge the channels back together and return the image
        return scaled_image