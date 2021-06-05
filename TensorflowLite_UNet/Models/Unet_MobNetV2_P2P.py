from keras import backend as K
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

from IPython.display import clear_output
import matplotlib.pyplot as plt




class Unet_MobNetV2_P2P:
	@staticmethod
	def Unet_model(im_height, im_width, input_channels, output_channels=3, trainable=False):
		"""
		_, im_height, im_width, input_channels = input.shape
		chanDim = -1
		if K.image_data_format() == "channels_first":
			_, input_channels, im_height, im_width = input.shape
			chanDim = 1
		"""

		base_model = tf.keras.applications.MobileNetV2(input_shape=[im_height, im_width, input_channels], weights="imagenet", include_top=False)
		#base_model_1 = tf.keras.applications.
		# Use the activations of these layers
		layer_names = [
			'block_1_expand_relu',  # 64x64
			'block_3_expand_relu',  # 32x32
			'block_6_expand_relu',  # 16x16
			'block_13_expand_relu',  # 8x8
			'block_16_project',  # 4x4
		]
		layers_1 = base_model.layers
		layers = [base_model.get_layer(name).output for name in layer_names]

		# Create the feature extraction model
		down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
		#down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

		down_stack.trainable = trainable

		#return down_stack

	#@staticmethod
	#def Unet_model(im_height, im_width, input_channels, output_channels=3):

		inputShape = [im_height, im_width, input_channels]
		chanDim = -1

		if K.image_data_format() == "channels_first":
			inputShape = [input_channels, im_height, im_width]
			chanDim = 1

		#Stacked layers for decoder
		up_stack = [
			pix2pix.upsample(512, 3),  # 4x4 -> 8x8
			pix2pix.upsample(256, 3),  # 8x8 -> 16x16
			pix2pix.upsample(128, 3),  # 16x16 -> 32x32
			pix2pix.upsample(64, 3),  # 32x32 -> 64x64
		]

		inputs = tf.keras.layers.Input(shape=inputShape)
		x = inputs

		# Downsampling through the model
		skips = down_stack(x)
		#skips = Unet_MobNetV2_P2P.Base_model(x)
		x = skips[-1]
		skips = reversed(skips[:-1])

		# Upsampling and establishing the skip connections
		for up, skip in zip(up_stack, skips):
			x = up(x)
			concat = tf.keras.layers.Concatenate()
			x = concat([x, skip])

		# This is the last layer of the model
		last = tf.keras.layers.Conv2DTranspose(output_channels, 3, strides=2, padding='same')  # 64x64 -> 128x128

		x = last(x)

		return tf.keras.Model(inputs=inputs, outputs=x)