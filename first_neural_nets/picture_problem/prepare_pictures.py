#!/usr/bin/env python3


# https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/

import sys
from matplotlib import pyplot



# example of loading the cifar10 dataset
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD


# load train and test dataset
def load_dataset():
    # load dat X=%s, y=%s' % (trainX.shape, trainY.shape))
    #    print('Test: X=%s, y=%s' % (testX.shape, testY.shape))
    # plot first aset
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    
    # convert into form that can be used to ML algorithm
    # one hot encode target values
    # to_categorical returns a binary matrix
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY




def print_dataset(trainX, trainY, testX, testY):
    # summarize loaded dataset
    print('Train:few images')
    for i in range(1):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        
        # plot raw pixel data
        print("\npicture shape:", trainX[i].shape)
        print("picture 32x32 pixels - row:", len(trainX[i]))
        print("picture 32x32 pixels - column  :", len(trainX[i][1]))
        print("picture one pixel RGB  :", len(trainX[i][1][1]))
        print("picture data:\n", trainX[i])
        
        pyplot.imshow(trainX[i])
    
    
    # show the figure
    pyplot.show()
    
    
# The integers are in the range of 0 to 255
# Here we normalize the picture values.
# Rescale them to the range [0,1]
def prepare_data(data):
    # convert from integers to floats
    data_norm = data.astype('float32')
    
    # normalize to range 0-1
    data_norm = data_norm / 255.0
    
    # return normalized images
    return data_norm
    

# Here we build the layers of the CNN (Convolutional Neural Networks) model
# CCN was designed to work with two dimensional image data.
def define_model():
	model = Sequential()
    
   
    # input shape
    # The pictures (data) is measuring 32 by 32 pixels squares
    # Each pixel has a value for its Red, Green and Blue 
    ## So there are basically three dimensions.
    ## 32 elements in the first and second (x,y) dimension
    ## 3  elements (data points) in the third dimension (z)  for R,G,B
    # https://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/
   
    
    # For the activation funtion we use the rectified linear 
    # activation function or ReLU,
    # since it overcomes the vanishing gradient problem which
    # appears in networks with many layers
    # https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
    # https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
    # https://machinelearningmastery.com/how-to-fix-vanishing-gradients-using-the-rectified-linear-activation-function/
    
    
    # For the initialization of the parameters of a tensors, 
    # the kernel initializer he_uniform is used which draws samples
    # from a uniform distribution
    # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeUniform


    ### Conv2D is a conventual layer
    # The purpose of a convential layer is to train the parameters of the filter.
    # Please note that the filter is The Filter which is supposed to find specific features.
    # So bascailly, the features of the pictures are unknown and 
    # the Conv2D layer is used to train the filter to find the features.

    # A conventual layer learns (outputs) more than one filter (or feature).
    # This means that each filter in the output represents a learned feature map.
    #
    # Color images have multiple channels, one channel for each of the colors R,G,B
    # This means that each filter has depth. Smaller images use a 3x3 filter.
    # This means we actually train a 3x3 filter, while moving the 3x3 filter over the image.
    #
    # The filter will move over the image (x,y) defined by strides and 
    # the value for strides defaults (1,1)
    # This means that the filter will move one pixel at a time.
    #
    # Padding can be used to chose whether the convulation output be the 
    # same size or smaller than the orgiginal image dimensions.
    # When 'same' is choosen, it means that 
    
    # https://missinglink.ai/guides/keras/keras-conv2d-working-cnn-2d-convolutions-keras/
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
    # https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/
    
    ### MaxPooling2D
    # This function reduces the data array
    # A pool size of (2, 2), means that 4 datapoints (2x2) will be reduced to
    # one data point. The new value will be the max value of a 2x2 area.
    # Since it is a two dimensional pooling function, the 3rd dimension will be not be touched.
    
    ### Flatten
    # This will flatten the data in the array into a one dimesional array
    # https://keras.io/api/layers/reshaping_layers/flatten/
    
    ### Dense
    
    
    # https://keras.io/api/layers/core_layers/dense/
    
    
    #######################################################################
    ### The model is created below
    #######################################################################
    
	model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    # The output are 32 filters/features, each identifying a feature. 
    # The shape for each filter will be the shape of the input (32x32), since padding is set to 'same'
    # These feature matrixes are input into the next layer
    
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    # The output are 32 filters/features , each representing a feature. 
    # The shape for each filter will be the shape of the input (32x32), since padding is set to 'same'

	model.add(MaxPooling2D(pool_size=(2, 2)))
    # The output wil be a reduced  matrix.
    # There will still be 32 fiters, however the matrix for each filter was reduced to 16x16

    
	model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    # The output will be 64 filters/features.
    # Basically now we are looking for smaller features within the picture
    
    
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    # The output wil be 64 filters/Refined Features
    
    
	model.add(MaxPooling2D((2, 2)))
    # The output will be a reduced matrix
    # There will still be 64 filters, but the matrix was reduced to 8x8
    
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    # The output will be 128 filters/features.
    # Basically now we are looking for smaller features within the picture
    
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    # The output will be 128 Refined Features
    
	model.add(MaxPooling2D((2, 2)))
    # The output wil be a reduced  matrix.
    # There will still be 128 filters, however the matrix for each filter was reduced to 8x8

    # Flatten the array
	model.add(Flatten())
    # The output is a one dimensional array with 128x8x8 - 2048 values
    
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    # the output will be 128 neurons, in a one dimensional matrix
    # The input were 128 features (2048 lines data points)  which were deteced in the picture

	model.add(Dense(10, activation='softmax'))
    # the output will be 10 neurons, using softmax as activation
    
    
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model



'''
The parameters are self explaining

'''
def main(argv):
    # load and prepare the dataset
    trainX, trainY, testX, testY = load_dataset()
    print_dataset(trainX, trainY, testX, testY)
    
    ## TRAIN / FIT the model
    # prepare pixel data
    trainX = prepare_data(trainX)
    
    # define and use a model
    model = define_model()
    print("model weights:\n", model.get_weights())

    print("model summary before training/fitting:\n", model.summary())
    #print("model:\n", model.get_weights())
    
        
    print("print model config:")
    print(model.get_config())
    
    # fit the model
    model.fit(trainX, trainY,\
        epochs=1,\
        #100,\
        batch_size=64,\
        validation_data=(testX, testY),\
        verbose=0)
    
    print("model summary after training/fitting:\n", model.summary())
    print("model:\n", model.get_weights())    
    
    
    ## EVALUATE the previously trained model
    testX = prepare_data(testX)
    #_, acc = model.evaluate(testX, testY, verbose=0)
  
if __name__ == '__main__':
    main(sys.argv[1:])
    
    
    





