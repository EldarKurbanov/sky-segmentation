import numpy as np
import cv2
import os

class SimpleDatasetLoader:
	def __init__(self, preprocessors=None):
		self.preprocessors = preprocessors

		if self.preprocessors is None:
			self.preprocessors = []


	def load(self, image_paths, verbose=-1):
		data = []
		labels = []

		for index, image_path in enumerate(image_paths):
			# assuming path : /path/to/dataset/{class}/{image}.jpg
			image = cv2.imread(image_path)
			#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			#image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
			label = image_path.split(os.path.sep)[-1][0]

			if self.preprocessors is not None:
				for preprocessor in self.preprocessors:
					image = preprocessor.preprocess(image)

			data.append(image)
			labels.append(label)

			if verbose > 0 and index > 0 and (index + 1) % verbose == 0:
				print(f'[INFO] processed {index + 1}/{len(image_paths)}')

		return (np.array(data), np.array(labels))


