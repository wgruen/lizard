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
    # load dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    
    # convert into form that can be used to ML algorithm
    # one hot encode target values
    # to_categorical returns a binary matrix
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY




def print_dataset(trainX, trainY, testX, testY):
    # summarize loaded dataset
    print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
    print('Test: X=%s, y=%s' % (testX.shape, testY.shape))
    # plot first few images
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
def define_model():
	model = Sequential()
    
    # https://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/
   
    # input shape
    ## how many elements and array or tensor has in each dimension
    ## three dimensions
    ## 32 elements in the first dimension
    ## 32 elements in the second dimension
    ## 3  elements in the third dimension
    ### The pictures (data) has 32 times 32 pixels
    ### Each pixel has a value for its Red, Green and Blue 
    
    # Conv2D
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
    # Total number of inputs is 32x32 = 1024 and three values (RGB) = 3072
    # activation relu = y = max(0, x)
    # VG based model
    # https://medium.com/datadriveninvestor/cnn-architecture-series-vgg-16-with-implementation-part-i-bca79e7db415
    
	model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

	model.add(MaxPooling2D(pool_size=(2, 2)))
    
    
	model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
    
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
    
    # Flatten the input
	model.add(Flatten())
    
    # First hidden network layer with 128 Neurons
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    
    # The output layer with 10 Neurons - output layer using softmax
	model.add(Dense(10, activation='softmax'))
    
    
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
    print("model summary:\n", model.summary())
    #print("model:\n", model.get_weights())
    
    # fit the model
    model.fit(trainX, trainY,\
        epochs=1,\
        #100,\
        batch_size=64,\
        validation_data=(testX, testY),\
        verbose=1)
    
    print(model.get_config())
    
    
    ## EVALUATE the previously trained model
    testX = prepare_data(testX)
    #_, acc = model.evaluate(testX, testY, verbose=0)
  
if __name__ == '__main__':
    main(sys.argv[1:])
    
    
    





