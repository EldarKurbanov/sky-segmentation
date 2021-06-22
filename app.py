from flask import Flask, url_for
from flask import request
from flask import render_template
import os
from werkzeug.utils import secure_filename
import cv2 as cv
import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2 as cv
import h5py
import argparse
from keras import backend as K
from TensorflowLite_UNet.config import skyseg_config as config
from TensorflowLite_UNet.Preprocessing import MeanPreprocessor, ImageToArrayPreprocessor, ScalePreprocessor
from keras.losses import binary_crossentropy
from TensorflowLite_UNet.Models import UNet_ResNet34
#from TensorflowLite_UNet.Models.UNet_ResNet34 import Unet_ResNet34
from TensorflowLite_UNet.Losses_And_Metrics import bce_logdice_loss, dice_coef, true_positive_rate



###################################Parameters###########################################################################
IMAGE_HEIGHT = 768#256
IMAGE_WIDTH = 768#256
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 3
model = True

def get_mask(fname,out_file,model):
	img = cv.imread(fname)
	r_size = img.shape[:2]
	print("rsize=",r_size)
	out_mask_name = out_file
	print(out_mask_name)
	names = out_file.split(".")
	out_filled_sky = names[-2] + "_filled."+names[-1]
	print(out_filled_sky)

	cv.imwrite("Original_image.jpg", np.asarray(img,dtype=np.uint8))

	input_image = tf.image.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

	#Set preprocessors
	sp = ScalePreprocessor(scale=1.0, norm=255.0)
	iap = ImageToArrayPreprocessor()

	preprocessors = [sp, iap]
	for pr in preprocessors:
		input_image = pr.preprocess(input_image)


	input_image = np.expand_dims(input_image, axis=0)

	img_masks = model.predict(input_image)
	print(img_masks.shape)
	x = 255*img_masks[0]
	im = np.ndarray(x.shape,dtype=np.uint8)
	im[:,:,:] = x[:,:,:]
	im = np.asarray(im,dtype=np.uint8)
	res = img.copy()#255 * input_image[0].copy()

	mask_2 = np.uint8(im)
	mask_2 = np.reshape(mask_2,(mask_2.shape[0],mask_2.shape[1]))
	mask_2 = cv.resize(mask_2,(r_size[1],r_size[0]),interpolation=cv.INTER_CUBIC)


	print(mask_2.shape)
	cond = mask_2 >224 
	print(cond.shape)

	r = res[:,:,2]
	g = res[:,:,1]
	b = res[:,:,0]
	r[cond] = 0
	g[cond] = 0
	b[cond] = 127

	print(res.shape)
	#res = cv.resize(res,(r_size[1],r_size[0]),interpolation=cv.INTER_CUBIC)
	cv.imwrite("."+out_filled_sky,res)
	im = cv.resize(im,(r_size[1],r_size[0]),interpolation=cv.INTER_CUBIC)
	#im = np.ndarray(tf.image.resize(im, r_size),dtype=np.uint8)
	print("mas_size",im.shape[:2])
	cv.imwrite(out_mask_name,im)

def get_mask_0(fname,out_file):
	im= cv.imread(fname)
	gray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
	cv.imwrite(out_file,gray)
	 
		


app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/home/',methods=['GET', 'POST'])
def home():
	if request.method == 'GET':
		
		return render_template("homeform.html")
	else:
		f = request.files['file']
		orig_name = secure_filename(f.filename)
		fname = './static/images/' + orig_name
		mask_name = './static/images/mask_' + orig_name
		f.save(fname)
		get_mask(fname,mask_name,model)
		src = url_for('static', filename='images/' + orig_name)
		mask_src = url_for('static', filename='images/mask_' + orig_name)
		snames = orig_name.split(".")
		filled_src = url_for('static', filename='images/mask_' + snames[-2] +"_filled." + snames[-1])
		data = {'src':src,'mask_src':mask_src,'filled_src':filled_src}
		return render_template("homeform.html",src=data['src'],mask_src=data['mask_src'],filled_src=data['filled_src'])
if __name__ == '__main__':
	########################################################################################################################
	#-----------------------------------------------------------------------------------------------------------------------
	#-----------------------GPU init----------------------------------------------------------------------------------------
	gpus = tf.config.experimental.list_physical_devices('GPU')
	if gpus:
		try:
			for gpu in gpus:

				tf.config.threading.set_inter_op_parallelism_threads = 2048
				tf.config.threading.set_intra_op_parallelism_threads = 2048
				print("Yes")
			tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5012)])
			#tf.config.experimental.set_memory_growth(gpu, True)
		except RuntimeError as e:
			print(e)
	
	model = load_model(os.path.join(config.BEST_WEIGHTS_STORE_DIR, 'seg_model_weights.best_768_log.hdf5'),
                   custom_objects={'loss': bce_logdice_loss}, compile=False)
	model.compile()

	print("Model loaded successfully...")

	app.run(host= '0.0.0.0', port=8000)
