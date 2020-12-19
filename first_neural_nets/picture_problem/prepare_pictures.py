#!/usr/bin/env python3


# https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
# https://towardsdatascience.com/a-beginners-guide-to-convolutional-neural-networks-cnns-14649dbddce8

import sys
from matplotlib import pyplot
import argparse
import yaml
import json
import io
from contextlib import redirect_stdout

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
        for i in range(1):
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
        self.model = Sequential()
        
       
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
        # https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/
        
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
        
        ### other standard optiizers
        # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
        
        #######################################################################
        ### The model is created below
        #######################################################################
        
        self.model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        # We are training 32 convolution filters. Eeach filter shall identify 
        # a feature in the picture.
        # The output are 32 features. 
        # The shape for each feature will be in the original size of the picture.
        #
        # How many neurons are in that layer?
        # There are 32 filters
        # Padding is set to valid (same)
        # The inputs depth is 3 , one for each  RGB
        # 32x3 = 96 Neurons - the number of filters times input depth
        #
        # How many parameters to trained?
        # There are 96 neurons
        # Each filter is 3x3
        # One bias parameter is Added for each filter (32) 
        # 96x3x3(864)  + 32(bias) = 896 prameters 
        
        self.model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        # We are training 32 convolution filters.
        # The output are 32 (refined?) features. 
        #
        # How many neurons are in that layer?
        # The previous filter has 96 neurons
        # This layer has 96 neurons as well
        #
        # How many parameters to trained?
        # One bias parameter is Added for each filter (32) 
        # 96x96 + 32(bias) = 9248 parameters

   
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # The input will be 32 filters
        # The input size is 96 neurons
        # The input is a feature of size 32x32
        # The output will be 32 filters
        # The output are 96 neurons
        # The output is a feature of size 16x16
        # The output is a reduced feature map
              
    
        
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        # The output will be 64 filters/features.
        # Basically now we are looking for smaller features within the picture
        #
        # How many neurons are in that layer?
        # There are 64 filters
        # Input depth is 3 (RGB)
        # 64x3 = 192 neurons
        #
        # How many parameters to trained?
        # This layer is 192 neurons
        # The previous layer has 96 neurons
        # One bias parameter is Added for each filter (64) 
        # 192x96 + 64(bias) = 18496 parameters
        
        
        self.model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        # The output wil be 64 filters/Refined Features
        #
        # How many neurons are in that layer?
        # This layer has 192 neurons
        # The previous layer has 192 neurons
        #
        # How many parameters to trained?
        # 192x192 + 64(bias) = 36928 parameters
        

        self.model.add(MaxPooling2D((2, 2)))
        # The input will be 64 filters
        # The input size is 192 neurons
        # The input is a feature of size 16x16
        # The output will be 64 filters
        # The output are 192 neurons
        # The output is a feature of size 8x8
        # The output is a reduced feature map
        
        
        self.model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        # The output will be 128 filters/features.
        # Basically now we are looking for smaller features within the picture
        #
        # How many neurons are in that layer?
        # There are 128 filters
        # Input depth is 3 (RGB)
        # 128x3 = 384 neurons
        #
        # How many parameters to trained?
        # This layer is 384 neurons
        # The previous layer has 192 neurons
        # One bias parameter is Added for each filter (128) 
        # 384x192 + 128(bias) = 73856 parameters
        
        
        self.model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        # The output wil be 128 filters/Refined Features
        #
        # How many neurons are in that layer?
        # This layer has 384 neurons
        # The previous layer has 384 neurons
        #
        # How many parameters to trained?
        # 384x384 + 128(bias) = 147584 parameters
        
        ### Now we are done with feature extraction
        # We will flatten before we look at all the features
        # and try to identify 10 different 
        # kind of pictures
        
        self.model.add(MaxPooling2D((2, 2)))
        # The input will be 128 filters
        # The input size is 384 neurons
        # The input is a feature of size 8x8
        # The output will be 128 filters
        # The output are 384 neurons
        # The output is a feature of size 4x4
        # The output is a reduced feature map
    
        # Flatten the array, go get ready for the Dense layers
        self.model.add(Flatten())
        # The input are 128 filters
        # The input has a depth of 3 (RGB)
        # The input size for each filter is 4x4
        # The output is a flattened one dimensional array
        # The output has 128x4x4 - 2048 neurons
        
        self.model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        # The input are 2048 neurons
        # The output are 128 neurons
        #
        # How many parameters are trained?
        # 2048x128 + 128(bias) = 262272 parameters
    
        self.model.add(Dense(10, activation='softmax'))
        # The input are 128 neurons
        # The output are 10 neurons, each output representing a large feature 
        #
        # How many parameters?
        # 128x10 + 10(bias) - 1290 parmeters
    
        
        # pick an optimizer for gradient descent with momentum optimizer
        opt = SGD(lr=0.001, momentum=0.9)
        
        # compile the model
        self.compile_output = self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
     
    
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
 

        # todo print to pdf before fitting
        
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
    

        # The history dictionary will have these keys: 
        # ['val_loss', 'val_acc', 'loss', 'acc']
            
        # todo print model after fitting
        

        # SAVE the trained model
        self.model.save(self.model_file_name)
        self.create_output_pdf()
        
        
            
    def create_output_pdf(self):
        ### print plots
        pdf_plots  = PdfPages("output/" +  self.model_file_name + "_plots.pdf")
        
        # SAVE documentation of the trained model
        # The history.history dictionary will have these keys: 
        # ['val_loss', 'val_acc', 'loss', 'acc']
        # plot loss
        #print("fit_history", self.fit_history.history)
        pyplot.subplot(3, 1, 1)
        pyplot.title('Cross Entropy Loss')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        pyplot.plot(self.fit_history.history['loss'], color='blue', label='train')
        pyplot.plot(self.fit_history.history['val_loss'], color='orange', label='test')
        # plot accuracy
        pyplot.subplot(3, 1, 3)
        pyplot.title('Classification Accuracy')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        pyplot.plot(self.fit_history.history['acc'], color='blue', label='train')
        pyplot.plot(self.fit_history.history['val_acc'], color='orange', label='test')
        
        # save plot to file
        pyplot.savefig(pdf_plots, format='pdf') #, bbox_inches='tight')
        pyplot.plot()
        pyplot.close()
        pdf_plots.close()
        
        
        ### create a text document with data
        doc_summary  = SimpleDocTemplate("output/" + self.model_file_name + "_summary.pdf", pagesize=letter)
    
        element = []
        header = Paragraph("\nSummary of Training Runs", styles["Heading1"])
        element.append(header)
                

        header = Paragraph("\nThe model summary", styles["Heading2"])
        element.append(header)
        f = io.StringIO() 
        with redirect_stdout(f):
            self.model.summary() 
        s = f.getvalue()
        print("model summary:\n", s)
        para = XPreformatted(s, styles["Code"], dedent=0)
        element.append(para)         

        header = Paragraph("\nThe model layers and weights", styles["Heading2"])
        element.append(header)
        para = XPreformatted(str(self.model.get_weights()),  styles["Code"], dedent=0)
        element.append(para)
            
        header = Paragraph("\nThe model configuration", styles["Heading2"])
        element.append(header)
        json_formatted_str = json.dumps(self.model.get_config(), indent=2, sort_keys=True)
        print(json_formatted_str)
        para = XPreformatted(json_formatted_str,  styles["Code"], dedent=0)
        element.append(para)
    
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
    
    
    





