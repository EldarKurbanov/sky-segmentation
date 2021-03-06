import numpy as np
import pandas as pd
import six

from random import randint

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
#import seaborn as sns
#sns.set_style("white")

from sklearn.model_selection import train_test_split

from skimage.transform import resize

from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout,BatchNormalization
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization
from tqdm import tqdm_notebook
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.engine import InputSpec
from keras import backend as K
from keras.regularizers import l2

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.merge import concatenate,add
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D

from keras.layers import Conv2DTranspose
from keras.layers import UpSampling2D
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Add
from keras.layers import ZeroPadding2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense

from keras.engine import get_source_inputs

from keras_applications.imagenet_utils import _obtain_input_shape


from keras.utils import get_file

from keras.losses import binary_crossentropy
#####################################*Pretrained Weights*#################################################################
weights_collection = [
    # ResNet34
    {
        'model': 'base_resnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet34_imagenet_1000.h5',
        'name': 'resnet34_imagenet_1000.h5',
        'md5': '2ac8277412f65e5d047f255bcbd10383',
    },

    {
        'model': 'base_resnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet34_imagenet_1000_no_top.h5',
        'name': 'resnet34_imagenet_1000_no_top.h5',
        'md5': '8caaa0ad39d927cb8ba5385bf945d582',
    }]
########################################################################################################################
class Unet_ResNet34:
#-----------------------------------------------------------------------------------------------------------------------
	@staticmethod
	def handle_block_names_decode(stage):

		conv_name = 'decoder_stage{}_conv'.format(stage)
		bn_name = 'decoder_stage{}_bn'.format(stage)
		relu_name = 'decoder_stage{}_relu'.format(stage)
		up_name = 'decoder_stage{}_upsample'.format(stage)

		return conv_name, bn_name, relu_name, up_name
#-----------------------------------------------------------------------------------------------------------------------
	@staticmethod
	def get_layer_number(model, layer_name):
		"""
	    Help find layer in Keras model by name
	    Args:
	        model: Keras `Model`
	        layer_name: str, name of layer
	    Returns:
	        index of layer
	    Raises:
	        ValueError: if model does not contains layer with such name
		"""
		for i, l in enumerate(model.layers):
			if l.name == layer_name:
				return i
		raise ValueError('No layer with name {} in  model {}.'.format(layer_name, model.name))
#-----------------------------------------------------------------------------------------------------------------------
	@staticmethod
	def Upsample2D_block(filters, stage, kernel_size=(3,3), upsample_rate=(2,2), batchnorm=False, skip=None):

		def layer(input_tensor):

			conv_name, bn_name, relu_name, up_name = Unet_ResNet34.handle_block_names_decode(stage)

			x = UpSampling2D(size=upsample_rate, name=up_name)(input_tensor)

			if skip is not None:
				x = Concatenate()([x, skip])

			x = Conv2D(filters, kernel_size, padding='same', name=conv_name+'1')(x)

			if batchnorm:
				x = BatchNormalization(name=bn_name+'1')(x)
			x = Activation('relu', name=relu_name+'1')(x)

			x = Conv2D(filters, kernel_size, padding='same', name=conv_name+'2')(x)
			if batchnorm:
				x = BatchNormalization(name=bn_name+'2')(x)
			x = Activation('relu', name=relu_name+'2')(x)

			return x

		return layer
#-----------------------------------------------------------------------------------------------------------------------
	@staticmethod
	def Transpose2D_block(filters, stage, kernel_size=(3,3), upsample_rate=(2,2), transpose_kernel_size=(4,4), batchnorm=False, skip=None):

		def layer(input_tensor):

			conv_name, bn_name, relu_name, up_name = Unet_ResNet34.handle_block_names_decode(stage)

			x = Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate, padding='same', name=up_name)(input_tensor)
			if batchnorm:
				x = BatchNormalization(name=bn_name+'1')(x)
			x = Activation('relu', name=relu_name+'1')(x)

			if skip is not None:
				x = Concatenate()([x, skip])

			x = Conv2D(filters, kernel_size, padding='same', name=conv_name+'2')(x)
			if batchnorm:
				x = BatchNormalization(name=bn_name+'2')(x)
			x = Activation('relu', name=relu_name+'2')(x)

			return x

		return layer
#-----------------------------------------------------------------------------------------------------------------------
	@staticmethod
	def build_unet(backbone, classes, last_block_filters, skip_layers, n_upsample_blocks=5, upsample_rates=(2,2,2,2,2),
	               block_type='upsampling', activation='sigmoid', **kwargs):

		input = backbone.input
		x = backbone.output
		print(x)
		if block_type == 'transpose':
			up_block = Unet_ResNet34.Transpose2D_block
		else:
			up_block = Unet_ResNet34.Upsample2D_block

		# convert layer names to indices
		skip_layers = ([Unet_ResNet34.get_layer_number(backbone, l) if isinstance(l, str) else l for l in skip_layers])
		for i in range(n_upsample_blocks):
			# check if there is a skip connection
			if i < len(skip_layers):
				print(backbone.layers[skip_layers[i]])
				print(backbone.layers[skip_layers[i]].output)
				skip = backbone.layers[skip_layers[i]].output
			else:
				skip = None

			up_size = (upsample_rates[i], upsample_rates[i])
			filters = last_block_filters * 2**(n_upsample_blocks-(i+1))

			x = up_block(filters, i, upsample_rate=up_size, skip=skip, **kwargs)(x)

		if classes < 2:
			activation = 'sigmoid'

		x = Conv2D(classes, (3,3), padding='same', name='final_conv')(x)
		x = Activation(activation, name=activation)(x)

		model = Model(input, x, name='u-resnet34')

		return model
#-----------------------------------------------------------------------------------------------------------------------
	@staticmethod
	def handle_block_names(stage, block):
		name_base = 'stage{}_unit{}_'.format(stage + 1, block + 1)
		conv_name = name_base + 'conv'
		bn_name = name_base + 'bn'
		relu_name = name_base + 'relu'
		sc_name = name_base + 'sc'

		return conv_name, bn_name, relu_name, sc_name
#-----------------------------------------------------------------------------------------------------------------------
	@staticmethod
	def get_conv_params(**params):

		default_conv_params = {
			'kernel_initializer': 'glorot_uniform',
			'use_bias': False,
			'padding': 'valid',
		}
		default_conv_params.update(params)

		return default_conv_params
#-----------------------------------------------------------------------------------------------------------------------
	@staticmethod
	def get_bn_params(**params):

		default_bn_params = {
	        'axis': 3,
	        'momentum': 0.99,
	        'epsilon': 2e-5,
	        'center': True,
	        'scale': True,
		}
		default_bn_params.update(params)

		return default_bn_params
#-----------------------------------------------------------------------------------------------------------------------
	@staticmethod
	def basic_identity_block(filters, stage, block):

		def layer(input_tensor):

			conv_params = Unet_ResNet34.get_conv_params()
			bn_params = Unet_ResNet34.get_bn_params()
			conv_name, bn_name, relu_name, sc_name = Unet_ResNet34.handle_block_names(stage, block)

			x = BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
			x = Activation('relu', name=relu_name + '1')(x)
			x = ZeroPadding2D(padding=(1, 1))(x)
			x = Conv2D(filters, (3, 3), name=conv_name + '1', **conv_params)(x)

			x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
			x = Activation('relu', name=relu_name + '2')(x)
			x = ZeroPadding2D(padding=(1, 1))(x)
			x = Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)

			x = Add()([x, input_tensor])

			return x

		return layer
#-----------------------------------------------------------------------------------------------------------------------
	@staticmethod
	def basic_conv_block(filters, stage, block, strides=(2, 2)):

		def layer(input_tensor):

			conv_params = Unet_ResNet34.get_conv_params()
			bn_params = Unet_ResNet34.get_bn_params()
			conv_name, bn_name, relu_name, sc_name = Unet_ResNet34.handle_block_names(stage, block)

			x = BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
			x = Activation('relu', name=relu_name + '1')(x)
			shortcut = x
			x = ZeroPadding2D(padding=(1, 1))(x)
			x = Conv2D(filters, (3, 3), strides=strides, name=conv_name + '1', **conv_params)(x)

			x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
			x = Activation('relu', name=relu_name + '2')(x)
			x = ZeroPadding2D(padding=(1, 1))(x)
			x = Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)

			shortcut = Conv2D(filters, (1, 1), name=sc_name, strides=strides, **conv_params)(shortcut)
			x = Add()([x, shortcut])
			return x

		return layer
#-----------------------------------------------------------------------------------------------------------------------
	@staticmethod
	def conv_block(filters, stage, block, strides=(2, 2)):

		def layer(input_tensor):

			conv_params = Unet_ResNet34.get_conv_params()
			bn_params = Unet_ResNet34.get_bn_params()
			conv_name, bn_name, relu_name, sc_name = Unet_ResNet34.handle_block_names(stage, block)

			x = BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
			x = Activation('relu', name=relu_name + '1')(x)
			shortcut = x
			x = Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(x)

			x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
			x = Activation('relu', name=relu_name + '2')(x)
			x = ZeroPadding2D(padding=(1, 1))(x)
			x = Conv2D(filters, (3, 3), strides=strides, name=conv_name + '2', **conv_params)(x)

			x = BatchNormalization(name=bn_name + '3', **bn_params)(x)
			x = Activation('relu', name=relu_name + '3')(x)
			x = Conv2D(filters * 4, (1, 1), name=conv_name + '3', **conv_params)(x)

			shortcut = Conv2D(filters * 4, (1, 1), name=sc_name, strides=strides, **conv_params)(shortcut)
			x = Add()([x, shortcut])

			return x

		return layer
#-----------------------------------------------------------------------------------------------------------------------
	@staticmethod
	def identity_block(filters, stage, block):

		def layer(input_tensor):

			conv_params = Unet_ResNet34.get_conv_params()
			bn_params = Unet_ResNet34.get_bn_params()
			conv_name, bn_name, relu_name, sc_name = Unet_ResNet34.handle_block_names(stage, block)

			x = BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
			x = Activation('relu', name=relu_name + '1')(x)
			x = Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(x)

			x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
			x = Activation('relu', name=relu_name + '2')(x)
			x = ZeroPadding2D(padding=(1, 1))(x)
			x = Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)

			x = BatchNormalization(name=bn_name + '3', **bn_params)(x)
			x = Activation('relu', name=relu_name + '3')(x)
			x = Conv2D(filters * 4, (1, 1), name=conv_name + '3', **conv_params)(x)

			x = Add()([x, input_tensor])

			return x

		return layer
#-----------------------------------------------------------------------------------------------------------------------
	@staticmethod
	def build_resnet(repetitions=(2, 2, 2, 2), include_top=True, input_tensor=None, input_shape=None, classes=1000,
			block_type='usual'):

		# Determine proper input shape
		input_shape = _obtain_input_shape(input_shape, default_size=224, min_size=197, data_format='channels_last',
		                                  require_flatten=include_top)

		if input_tensor is None:
			img_input = Input(shape=input_shape, name='data')
		else:
			if not K.is_keras_tensor(input_tensor):
				img_input = Input(tensor=input_tensor, shape=input_shape)
			else:
				img_input = input_tensor

		# get parameters for model layers
		no_scale_bn_params = Unet_ResNet34.get_bn_params(scale=False)
		bn_params = Unet_ResNet34.get_bn_params()
		conv_params = Unet_ResNet34.get_conv_params()
		init_filters = 64

		if block_type == 'basic':
			conv_block = Unet_ResNet34.basic_conv_block
			identity_block = Unet_ResNet34.basic_identity_block
		else:
			conv_block = Unet_ResNet34.conv_block
			identity_block = Unet_ResNet34.identity_block

		# resnet bottom
		x = BatchNormalization(name='bn_data', **no_scale_bn_params)(img_input)
		x = ZeroPadding2D(padding=(3, 3))(x)
		x = Conv2D(init_filters, (7, 7), strides=(2, 2), name='conv0', **conv_params)(x)
		x = BatchNormalization(name='bn0', **bn_params)(x)
		x = Activation('relu', name='relu0')(x)
		x = ZeroPadding2D(padding=(1, 1))(x)
		x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pooling0')(x)

		# resnet body
		for stage, rep in enumerate(repetitions):
			for block in range(rep):

				filters = init_filters * (2 ** stage)

				# first block of first stage without strides because we have maxpooling before
				if block == 0 and stage == 0:
					x = conv_block(filters, stage, block, strides=(1, 1))(x)

				elif block == 0:
					x = conv_block(filters, stage, block, strides=(2, 2))(x)

				else:
					x = identity_block(filters, stage, block)(x)

		x = BatchNormalization(name='bn1', **bn_params)(x)
		x = Activation('relu', name='relu1')(x)

		# resnet top
		if include_top:
			x = GlobalAveragePooling2D(name='pool1')(x)
			x = Dense(classes, name='fc1')(x)
			x = Activation('softmax', name='softmax')(x)

		# Ensure that the model takes into account any potential predecessors of `input_tensor`.
		if input_tensor is not None:
			inputs = get_source_inputs(input_tensor)
		else:
			inputs = img_input

		# Create model.
		model = Model(inputs, x, name='base_resnet34')
		print(model.name)

		return model
#-----------------------------------------------------------------------------------------------------------------------
	@staticmethod
	def find_weights(weights_collection, model_name, dataset, include_top):

		w = list(filter(lambda x: x['model'] == model_name, weights_collection))
		w = list(filter(lambda x: x['dataset'] == dataset, w))
		w = list(filter(lambda x: x['include_top'] == include_top, w))

		return w
#-----------------------------------------------------------------------------------------------------------------------
	@staticmethod
	def load_model_weights(weights_collection, model, dataset, classes, include_top):
		weights = Unet_ResNet34.find_weights(weights_collection, model.name, dataset, include_top)

		if weights:
			weights = weights[0]

			if include_top and weights['classes'] != classes:
				raise ValueError('If using `weights` and `include_top`'
				                 ' as true, `classes` should be {}'.format(weights['classes']))

			weights_path = get_file(weights['name'],
			                        weights['url'],
			                        cache_subdir='models',
			                        md5_hash=weights['md5'])

			model.load_weights(weights_path)

		else:
			raise ValueError('There is no weights for such configuration: ' +
			                 'model = {}, dataset = {}, '.format(model.name, dataset) +
			                 'classes = {}, include_top = {}.'.format(classes, include_top))
#-----------------------------------------------------------------------------------------------------------------------
	@staticmethod
	def freeze_model(model):
		for layer in model.layers:
			layer.trainable = False
		return
#-----------------------------------------------------------------------------------------------------------------------
	@staticmethod
	def UResNet34(input_shape=(None, None, 3), classes=1, decoder_filters=16, decoder_block_type='upsampling',
	              encoder_weights=None, input_tensor=None, activation='sigmoid', **kwargs):

		backbone = Unet_ResNet34.build_resnet(input_tensor=input_tensor, input_shape=input_shape, repetitions=(3, 4, 6, 3),
		                        classes=classes, include_top=False, block_type='basic')
		#backbone.
		#backbone.name = 'base_resnet34'

		if encoder_weights == True:
			Unet_ResNet34.load_model_weights(weights_collection=weights_collection, model=backbone, dataset='imagenet',
			                                 classes=1, include_top=False)

		skip_connections = list([129, 74, 37, 5])  # for resnet 34
		model = Unet_ResNet34.build_unet(backbone=backbone, classes=classes, last_block_filters=decoder_filters,
		                   skip_layers=skip_connections, block_type=decoder_block_type, activation=activation, **kwargs)
		#model.name = 'u-resnet34'

		# freeze_model(backbone)

		return model
#-----------------------------------------------------------------------------------------------------------------------
	@staticmethod
	def Seg_model(im_width,im_height, in_channels, encoder_weights=True):

		return Unet_ResNet34.UResNet34(input_shape=(im_width, im_height, in_channels), encoder_weights=encoder_weights)
#-----------------------------------------------------------------------------------------------------------------------
	@staticmethod
	def dice_coef(y_true, y_pred):

		y_true_f = K.flatten(y_true)
		y_pred = K.cast(y_pred, 'float32')
		y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
		intersection = y_true_f * y_pred_f
		score = 2.0 * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))

		return score
#-----------------------------------------------------------------------------------------------------------------------
	@staticmethod
	def dice_loss(y_true, y_pred):

		smooth = 1.0
		y_true_f = K.flatten(y_true)
		y_pred_f = K.flatten(y_pred)
		intersection = y_true_f * y_pred_f
		score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

		return 1.0 - score
#-----------------------------------------------------------------------------------------------------------------------
	@staticmethod
	def bce_dice_loss(y_true, y_pred):

		return binary_crossentropy(y_true, y_pred) + Unet_ResNet34.dice_loss(y_true, y_pred)
#-----------------------------------------------------------------------------------------------------------------------
	@staticmethod
	def bce_logdice_loss(y_true, y_pred):

		return binary_crossentropy(y_true, y_pred) - K.log(1.0 - Unet_ResNet34.dice_loss(y_true, y_pred))
#-----------------------------------------------------------------------------------------------------------------------
	@staticmethod
	def weighted_bce_loss(y_true, y_pred, weight):

		epsilon = 1e-7
		y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
		logit_y_pred = K.log(y_pred / (1.0 - y_pred))
		loss = weight * (logit_y_pred * (1.0 - y_true) + K.log(1. + K.exp(-K.abs(logit_y_pred))) +
		                 K.maximum(-logit_y_pred, 0.))

		return K.sum(loss) / K.sum(weight)
#-----------------------------------------------------------------------------------------------------------------------
	@staticmethod
	def weighted_dice_loss(y_true, y_pred, weight):

		smooth = 1.0
		w, m1, m2 = weight, y_true, y_pred
		intersection = (m1 * m2)
		score = (2.0 * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
		loss = 1.0 - K.sum(score)

		return loss
#-----------------------------------------------------------------------------------------------------------------------
	@staticmethod
	def weighted_bce_dice_loss(y_true, y_pred):

		y_true = K.cast(y_true, 'float32')
		y_pred = K.cast(y_pred, 'float32')
		# if we want to get same size of output, kernel size must be odd
		averaged_mask = K.pool2d(y_true, pool_size=(50, 50), strides=(1, 1), padding='same', pool_mode='avg')
		weight = K.ones_like(averaged_mask)
		w0 = K.sum(weight)
		weight = 5. * K.exp(-5. * K.abs(averaged_mask - 0.5))
		w1 = K.sum(weight)
		weight *= (w0 / w1)
		loss = Unet_ResNet34.weighted_bce_loss(y_true, y_pred, weight) + Unet_ResNet34.dice_loss(y_true, y_pred)

		return loss
#-----------------------------------------------------------------------------------------------------------------------
	"""
	@staticmethod
	def dice_coef(y_true, y_pred, smooth=1):

		intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
		union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])

		return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

	"""
#-----------------------------------------------------------------------------------------------------------------------
	@staticmethod
	def dice_p_bce(in_gt, in_pred):

		return 1e-3 * binary_crossentropy(in_gt, in_pred) - Unet_ResNet34.dice_coef(in_gt, in_pred)
#-----------------------------------------------------------------------------------------------------------------------
	@staticmethod
	def true_positive_rate(y_true, y_pred):

		return K.sum(K.flatten(y_true) * K.flatten(K.round(y_pred))) / (K.sum(y_true) + K.epsilon())
#-----------------------------------------------------------------------------------------------------------------------
	"""
	@staticmethod
	def true_positive_rate(y_true, y_pred):

		return K.sum(K.flatten(y_true) * K.flatten(K.round(y_pred))) / K.sum(y_true)
	"""
#-----------------------------------------------------------------------------------------------------------------------
	@staticmethod
	def model_load(model_load_dir):

		return load_model(model_load_dir)
#-----------------------------------------------------------------------------------------------------------------------
