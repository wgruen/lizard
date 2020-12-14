#!/usr/bin/env python3


# https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/

import sys
from matplotlib import pyplot
import argparse
import yaml
import json

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from reportlab.pdfgen import canvas
from reportlab.platypus import *
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
styles = getSampleStyleSheet()


# example of loading the cifar10 dataset
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from tensorflow import keras



class myCIFAR10():
    def __init__(self, configuration):
        ### get the parameters
        self.machine_dump_file_name_base = configuration["machine_dump_file_name_base"]
        self.number_of_epochs = configuration["number_of_epochs"]
        self.model_file_name = self.machine_dump_file_name_base + "-epochs-" \
            + str(self.number_of_epochs) + ".bin"
        self.verbose = configuration["verbose"]
            
    '''
    load the dataset
    '''
    def load_dataset(self):
        # load dat X=%s, y=%s' % (self.trainX.shape, self.trainY.shape))
        #    print('Test: X=%s, y=%s' % (self.testX.shape, self.testY.shape))
        # plot first aset
        (self.trainX, self.trainY), (self.testX, self.testY) = cifar10.load_data()
        
        # convert into form that can be used to ML algorithm
        # one hot encode target values
        # to_categorical returns a binary matrix
        self.trainY = to_categorical(self.trainY)
        self.testY = to_categorical(self.testY)




    def print_dataset(self):
        # summarize loaded dataset
        print('Train:few images')
        for i in range(3):
            # define subplot
            pyplot.subplot(330 + 1 + i)
            
            # plot raw pixel data
            print("\npicture shape:", self.trainX[i].shape)
            print("picture 32x32 pixels - row:", len(self.trainX[i]))
            print("picture 32x32 pixels - column  :", len(self.trainX[i][1]))
            print("picture one pixel RGB  :", len(self.trainX[i][1][1]))
            print("picture data:\n", self.trainX[i])
            
            pyplot.imshow(self.trainX[i])
        
        
        # show the figure
        pyplot.show()
    
    
    '''
    The integers are in the range of 0 to 255
    Here we normalize the picture values.
    Rescale them to the range [0,1]
    
    This prepares the pixel data
    '''
    def prepare_data(self, data):
        # convert from integers to floats
        data_norm = data.astype('float32')
        
        # normalize to range 0-1
        data_norm = data_norm / 255.0
        
        # return normalized images
        return data_norm
 
       
    '''
    Define the keras model.
    Please note that the model defintion is highly dependenet on the 
    image size
    
    Here we build the layers of the CNN (Convolutional Neural Networks) model
    CCN was designed to work with two dimensional image data.
    '''
    def define_model(self):
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
        
        
        ### SGD Optimizer
        # influnces how much in in which way a sungeparameters will be changed
        # Influences the way the error will be changed
        # lr - learning rate - at which rate to change the parameters
        # a slower learning rate may keep values from oszillation
        # momentum - helps to find the minimum faster
        # For example, when the error is on a long downslope,
        # then the error value can move faster to the minimum
        # https://mlfromscratch.com/optimizers-explained/#/
        
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
        
        
        # pick an optimizer
        opt = SGD(lr=0.001, momentum=0.9)
    
        
        # compile the model
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        
        self.model = model
    
    
    '''
    The model will be trained (fit) and
    the trained model will be stored to hd.
    
    '''
    def fit_model(self):
        ### load and prepare the dataset
        self.load_dataset()
        self.print_dataset()
        
        # normalize the pixel data for X only
        self.trainX = self.prepare_data(self.trainX)
        self.testX = self.prepare_data(self.testX)
        
        ### Define and use a model
        self.define_model()
        print("model weights:\n", self.model.get_weights())
    
        print("model summary before training/fitting:\n", self.model.summary())
        #print("model:\n", model.get_weights())
            
        print("print model config:")
        print(self.model.get_config())
        
        ### FIT / TRAIN the model
        ## Sample - is a set of data, also called Rows Of Data
        # In the case of CIFAR-10, as sample is the same as the data of one picture
        # , I guess
        #
        ## batch size - is a number of samples (Rows of Data)
        # which are processed before the model is updated.
        # The size of a batch must be more than or equal to one and 
        # less than or equal to the number of samples in the training dataset.
        #
        ## epoch - number of complete passes through the training set
        #
        ## vaidation data
        #  Data on which to evaluate the loss of the model 
        # at the end of each epoch. The model will not be trained on this data.
        # https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/
        
        self.fit_history = self.model.fit(self.trainX, self.trainY,\
            epochs=self.number_of_epochs,\
            batch_size=64,\
            validation_data=(self.testX, self.testY),\
            verbose=self.verbose)
        
        print("model summary after training/fitting:\n", self.model.summary())
        print("model:\n", self.model.get_weights())    
        
        # SAVE the trained model
        self.model.save(self.model_file_name)
        self.create_output_pdf()
        
        
            
    def create_output_pdf(self):    
        pdf_plots  = PdfPages("output/" +  self.model_file_name + "_plots.pdf")
#        doc_outliers = SimpleDocTemplate("output/" + filebase + "_outliers.pdf", pagesize=letter)
#        doc_zeros    = SimpleDocTemplate("output/" + filebase + "_zeros.pdf", pagesize=letter)        
        
        # SAVE documentation of the trained model
        # plot loss
        #print("fit_history", self.fit_history.history)
        pyplot.subplot(3, 1, 1)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(self.fit_history.history['loss'], color='blue', label='train')
        pyplot.plot(self.fit_history.history['val_loss'], color='orange', label='test')
        # plot accuracy
        pyplot.subplot(3, 1, 3)
        pyplot.title('Classification Accuracy')
        pyplot.plot(self.fit_history.history['acc'], color='blue', label='train')
        pyplot.plot(self.fit_history.history['val_acc'], color='orange', label='test')
        
        # save plot to file
        pyplot.savefig(pdf_plots, format='pdf') #, bbox_inches='tight')
        pyplot.plot()
        pyplot.close()
        pdf_plots.close()
        
        
        
        doc_summary  = SimpleDocTemplate("output/" + self.model_file_name + "_summary.pdf", pagesize=letter)
    
        element = []
        header = Paragraph("\nSummary of Training Runs", styles["Heading1"])
        element.append(header)
        
        
        # sort by standard deviation  
        header = Paragraph("\nThe model", styles["Heading2"])
        element.append(header)
        print("model summary before training/fitting:\n", self.model.summary())
        text = Paragraph(str(self.model.summary()),  styles["Normal"])
        element.append(text)
  
        print(self.model.get_config())
        #json_object = json.loads(self.model.get_config())
        json_formatted_str = json.dumps(self.model.get_config(), indent=2)
        text2 = Paragraph(str(json_formatted_str),  styles["Normal"])
        element.append(text2)
    
        
        doc_summary.build(element)

       
            
    
    '''
    The model is evaluated with the test data, which is part of
    the training data
    '''
    def evaluate_data(self):
        ### load and prepare the dataset
        self.load_dataset()
        #print_dataset(self.trainX, self.trainY, self.testX, self.testY)
        
        
        
        ## load a previously stored model
        self.model = keras.models.load_model('keras_cifar-10.trained.bin')
    
    
        ## EVALUATE the previously trained model
        self.testX = self.prepare_data(self.testX)
        _, acc = self.model.evaluate(self.testX, self.testY, verbose=self.verbose)
        

'''
main
'''
def main(argv):
    parser = argparse.ArgumentParser(description='Training and running a small neural network.')
    parser.add_argument('-i', dest='input_config_file', 
                        action='store', 
                        required=True,
                        help='The yaml file containing configuration')
    
    parser.add_argument("--train",  action='store_true', 
                    #    required=False,
                        help='train the machine')
    
    parser.add_argument("--validate", action='store_true', 
                     #   required=False,
                        help='vaiidate the machine')
    args = parser.parse_args()
    #print(args)
       
       
    configuration = None
    with open(args.input_config_file, 'r') as stream:
        configuration = yaml.safe_load(stream)
        
    print(yaml.safe_dump(configuration, default_flow_style=False, default_style=None))
    
    mymodel = myCIFAR10(configuration)

    if args.train:
        mymodel.fit_model()
        
    if args.validate:     
        mymodel.evaluate_data()

    


if __name__ == '__main__':
    main(sys.argv[1:])
    
    
    





