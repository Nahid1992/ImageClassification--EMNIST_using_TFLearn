import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from scipy import io as spio
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from sklearn.preprocessing import OneHotEncoder
#from tensorflow.examples.tutorials.mnist import input_data

#X,Y,test_x,test_y = emnist.load_data(one_hot=True)

img_width = 28
img_height = 28
nb_classes = 27

def label_process(y):
	label = []
	for index in range(0,y.shape[0]):
		row = np.zeros((nb_classes,), dtype=int)
		row[y[index]] = 1
		row = row.reshape(1,nb_classes)
		label.append(row)
	return label

def loadData():
	#Data Loading
	emnist = spio.loadmat("EMNIST_Dataset/emnist-letters.mat")
	print('Data Loaded...')

	#Training Set
	x_train = emnist["dataset"][0][0][0][0][0][0]
	x_train = x_train.astype(np.float32)
	y_train = emnist["dataset"][0][0][0][0][0][1]

	y_label = []
	for index in range(0,y_train.shape[0]):
		y_label.append(y_train[index][0])

	y_train = y_label
	y_train = np.array(y_train)
	y_train = y_train.reshape(y_train.shape[0],1)

	y_train = label_process(y_train)
	print('Hot Encoding Done for Training Set...')

	#Testing Set
	x_test = emnist["dataset"][0][0][1][0][0][0]
	x_test = x_test.astype(np.float32)
	y_test = emnist["dataset"][0][0][1][0][0][1]

	y_label = []
	for index in range(0,y_test.shape[0]):
		y_label.append(y_test[index][0])

	y_test = y_label
	y_test = np.array(y_test)
	y_test = y_test.reshape(y_test.shape[0],1)

	y_test = label_process(y_test)
	print('Hot Encoding Done for Testing Set...')

	x_train /= 255
	x_test /= 255

	x_train = x_train.reshape(x_train.shape[0], img_width, img_height,1)
	x_test = x_test.reshape(x_test.shape[0], img_width, img_height, 1)

	y_train = np.array(y_train)
	y_train = y_train.reshape(y_train.shape[0],nb_classes)

	y_test = np.array(y_test)
	y_test = y_test.reshape(y_test.shape[0],nb_classes)


	x_train,y_train = tflearn.data_utils.shuffle(x_train,y_train)
	x_test,y_test = tflearn.data_utils.shuffle(x_test,y_test)
	print('Ready to Train...')

	return x_train,y_train,x_test,y_test

def create_own_model_two_conv2():
    network = input_data(shape=[None, img_width, img_height,1])

    network = conv_2d(network, 32, 3, strides=2, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = conv_2d(network, 64, 3, strides=2, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = fully_connected(network, 128, activation='relu')
    network = fully_connected(network, nb_classes, activation='softmax')
    model = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)

    return model

def create_own_model():

	# Real-time data preprocessing
	img_prep = ImagePreprocessing()
	img_prep.add_featurewise_zero_center()
	img_prep.add_featurewise_stdnorm()

	# Real-time data augmentation
	img_aug = ImageAugmentation()
	img_aug.add_random_flip_leftright()
	img_aug.add_random_rotation(max_angle=25.)

	convnet = input_data(shape=[None, img_width, img_height,1],
					data_preprocessing=img_prep,
					data_augmentation=img_aug)

	convnet = conv_2d(convnet, 28, 3, activation = 'relu')
	convnet = max_pool_2d(convnet,3)

	convnet = conv_2d(convnet, 28, 3, activation = 'relu')
	convnet = max_pool_2d(convnet,3)

	convnet = fully_connected(convnet, 512, activation='relu')
	oonvnet = dropout(convnet, 0.2)

	convnet = fully_connected(convnet, nb_classes, activation='softmax')

	model = regression(convnet, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='targets')

	return model

def train_model(model,x_train,y_train,x_val,y_val):
	model = tflearn.DNN(model,tensorboard_verbose=3)

	#comment this out after fully TRAINED
	model.fit(x_train,y_train, n_epoch=20, validation_set=(x_val,y_val),
		batch_size=500,snapshot_step=50, show_metric=True, run_id='emnist_run07')

	print('Training Done..')
	#saves weights of the model
	model.save('models/EMNIST_tflearncnn.model')
	# Comment out
	print('Training Model Saved...	')

def main():
	x_train,y_train,x_val,y_val = loadData()
	model = create_own_model()
	train_model(model,x_train,y_train,x_val,y_val)

	'''
	#just load the last trained model and ready for test
	model.load('models/EMNIST_tflearncnn.model')
	#test on individual
	print( model.predict( [test_x[2]] ) )
	print('Original Label = ',test_y[2])

	#Individual Test Image Show
	IMG = test_x[2]
	IMG = np.array(IMG, dtype='float')
	pixels = IMG.reshape((28, 28))
	plt.imshow(pixels, cmap='gray')
	plt.show()
	'''

if __name__== "__main__":
  main()
