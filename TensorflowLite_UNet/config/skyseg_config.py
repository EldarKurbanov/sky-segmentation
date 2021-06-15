# define the path to the images directory
# path = '/Users/siddhantbansal/Desktop/Python/Personal_Projects/Cats_vs_Dogs/dataset/kaggle_dogs_vs_cats/train'
# since we do not have access to validation data we need to take a number of images from train and test on them
NUM_CLASSES = 2
NUM_VAL_IMAGES = 1250 * NUM_CLASSES
NUM_TEST_IMAGES = 1250 * NUM_CLASSES

# define the file path to output training, validation and testing HDF5 files
#TRAIN_HDF5 = '/home/ubuntu/PycharmProjects/Sky_Segmentation/TensorflowLite_UNet/Dataset/hdf5/train3c.hdf5'

#TRAIN_HDF5 = '/home/ubuntu/PycharmProjects/Sky_Segmentation/TensorflowLite_UNet/Dataset/hdf5/train_3_1c_256_norm.hdf5'
TRAIN_HDF5 = '/code/TensorflowLite_UNet/Dataset/hdf5/train_3_1c_768_norm_full.hdf5'

#VAL_HDF5 = '/home/ubuntu/PycharmProjects/Sky_Segmentation/TensorflowLite_UNet/Dataset/hdf5/val_3_1c_256_norm.hdf5'
VAL_HDF5 = '/code/TensorflowLite_UNet/Dataset/hdf5/val_3_1c_768_norm.hdf5'

#TEST_HDF5 = '/home/ubuntu/PycharmProjects/Sky_Segmentation/TensorflowLite_UNet/Dataset/hdf5/test_3_1c_256_norm.hdf5'
TEST_HDF5 = '/code/TensorflowLite_UNet/Dataset/hdf5/test_3_1c_768_norm.hdf5'

# path to the output model file
MODEL_PATH = '/code/TensorflowLite_UNet/model'

MODEL_ARCHITECT_DIR = '/code/TensorflowLite_UNet/Visualize_architect'

BEST_WEIGHTS_STORE_DIR = '/code/TensorflowLite_UNet/Best_weights'

# define the path to the dataset mean
DATASET_MEAN = '/code/TensorflowLite_UNet/Output/mean_values.json'

# define the path to the output directory used for storing plots, classification_reports etc.
OUTPUT_PATH = '/code/TensorflowLite_UNet/Output'