##################################################################################
# Author:   Adam Tipton
# Title:    Project Bogie Trainer
# Version:  1
#
# Company:  Brigham Young University - Idaho
# Course:   CSE 499 Senior Project
# Semester: Fall 2020
# 
# Description:
#   Project Bogie is a Convolutional Neural Network written in the Python language. 
#   This CNN will train a model from a dataset the ability to identify military 
#   aircraft/jets. 
#   
#   Once trained, the program will create save the model for use in an application
#   that allows a user to input an image to test if it contains a military aircraft.
#
#   This version will use a 3 block CNN that draws from the VGG16 model. This gives
#   our model a headstart in training as many of the weights are preset, giving 
#   us the advantage of a shortened training time. 
#
#   Project Bogie uses TensorFlow as a backend and Keras as a driving force for training.
#
#   The TRAINER will be the primary workshorse for this project, which is divided into
#   two parts. Part I is this program, the trainer. Part II is the Predictor program.
#   The trainer is responsible for taking a data set of Modern Military Aircraft Images
#   and training a CNN to output a model. Therefore, the deliverable of this program
#   is to create h5 and hdf5 models. 
#   
#
# Sources:
#   Much inspiration and reusable code base is taken from the following tutorial -
#   URL: https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/
#   create_model(), diagnostics(), and train_and_evaluate_model() rely heavily on this 
#   tutorial and code.
###################################################################################   
###################################################################################

# System
import sys

# This important import gives us a headstart on training our model.
from keras.applications.vgg16 import VGG16

# Other important keras imports
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

# This import is for playing a sound
import winsound

# These imports are important for file handling
import os
from os import listdir

# This allows us to easily adjust epochs
epoch_rate = 200

# The learning rate is an important factor in training neural networks
learning_rate = 0.0016
# Import for pyplot
from matplotlib import pyplot


# The momentum variable can be adjusted to tweak learning
momentum_rate = 0.9

# The batch variable allows us to adjust batch sizes
batch = 42

# The model names
default_model_name = "Project_Bogie_Model.h5"
checkpoint_model_name = "weights.best"

# The location of our dataset for training
dataSet_dir = 'C:/Users/Adam/source/repos/military_aircraft/aircraft_folders_unsorted'

# The location of an alert sound we'd like to play once it finishes - replace with your own
sound_url = 'C:/Users/Adam/Music/Sounds/Soothing_Alert'

# The location of the model saves
model_save_path = ''


##########################################################
################         FUNCTIONS        ################
##########################################################

# This function welcomes the user with a informative message
def welcome():
    print(" ############################################################################################")
    print(" ############################################################################################")
    print(" ##################################                         #################################")
    print(" ##################################      PROJECT BOGIE      #################################")
    print(" ##################################         TRAINER         #################################")
    print(" ##################################                         #################################")
    print(" ##################################  Copyright BYU-I 2020   #################################")
    print(" ##################################  Author: Adam Tipton    #################################")
    print(" ##################################                         #################################")
    print(" ##################################  CS499 Final Project    #################################")
    print(" ##################################                         #################################")
    print(" ############################################################################################")
    print(" ############################################################################################")
    print(" #                                                                                          #")
    print(" #                                 Welcome to Project Bogie.                                #")
    print(" #                                                                                          #")
    print(" #                                        *  *  *  *                                        #")
    print(" #                                                                                          #")
    print(" #  Project Bogie Trainer uses a Convolutional Neural Network trained against image data    #")
    print(" #  for the purpose of creating a predictive model that can determine the likelihood of an  #")
    print(" #  input image containing a modern military aircraft or something else. For the purposes   #")
    print(" #  of clarification, a modern military aircraft will generally be post-Vietnam War era     #")
    print(" #  vintage.                                                                                #")
    print(" #                                                                                          #")
    print(" #  Generally, two classes of data are provided. The first class, which would produce near  #")
    print(" #  0 values, would contain modern military aircraft. The second class, which would produce #")
    print(" #  near 1 values, would contain other non-military aircraft and other unclassified random  #")
    print(" #  images. This is done in order to train the model to detect the difference between what  #")
    print(" #  we want to identify, modern military aircraft, and what we don't, everything else.      #")
    print(" #                                                                                          #")
    print(" #  The trainer will produce a hierarchical model with the .h5 and/or the .hdf5 extension.  #")
    print(" #  As the training progresses, checkpoints will periodically save the current best model   #")
    print(" #  which will be saved in a directory in the .hdf5 extension. Several 'best' models may be #")
    print(" #  produced. Additionally, a final model witht he .hdf5 extension will be produced at the  #")
    print(" #  conclusion of the epoch cycle. This model may differ significantly from the 'best'      #")
    print(" #  checkpointed model in its accuracy, which is why both model types are provided. Testing #")
    print(" #  has shown that the checkpoing 'best' model will generally contain a higher accuracy     #")
    print(" #  than the end of epoch model, which has the possibility to degrade slightly. if you are  #")
    print(" #  unsure which model to chose, chose the checkpointed 'best' model, as it will guaruntee  #")
    print(" #  you the highest accuracy.                                                               #")
    print(" #                                                                                          #")
    print(" ############################################################################################")
    print(" ############################################################################################")
    print("\n")
    print("\n")

# This function allows the user to initialize important variable if they choose and enter a save filename
def setup():

    # ask the user if they want to set up the variables
    response = input("\n Press 'Y' if you would like to setup the training variables or any other key to skip: ")
    if response == 'Y' or response == 'y':
        print("\n Sure . . .")
    else:
        print("\n Skipping setup . . . \n")
        return

    #epoch_rate
    global epoch_rate
    epoch_rate = input("\n Please enter the epoch rate (e.g. 50): ")
    #handle unexpected string input
    if not epoch_rate.isnumeric():
        epoch_rate = int(1);
    else:
        int(float(epoch_rate))
    if int(float(epoch_rate)) < int(1):
        print(" Epoch rate too low. Adjusting epoch rate to 1 . . .")
        epoch_rate = int(1)
    else:
        print(" Epoch rate is set to: ", str(epoch_rate))
    
    #learning_rate
    global learning_rate
    learning_rate = input("\n Please enter the learning rate (e.g. 0.001): ")
    #handle unexpected string input
    if not learning_rate.isnumeric():
        learning_rate = float(0.1)
    else:
        float(learning_rate)
    if float(learning_rate) >= float(1.0):
        print(" Learning rate too high. Adjusting learning rate to 0.1")
        learning_rate = float(0.1)
    else:
        print(" Learning rate is set to: ", str(learning_rate))
               
    #momentum_rate
    global momentum_rate
    momentum_rate = input("\n Please enter the momentum rate (e.g. 0.9): ")
    #handle unexpected string input
    if not momentum_rate.isnumeric():
        momentum_rate = float(0.1)
    else:
        float(momentum_rate)
    if float(momentum_rate) >= float(1):
        print(" Momentum rate too high. Adjusting momentum rate to 0.1")
        momentum_rate = float(0.1)
    else:
        print(" Momentum rate is set to: ", str(momentum_rate))

    #batch
    global batch
    batch = input("\n Please enter the batch count (e.g. 25): ")
    #handle unexpected string input
    if not batch.isnumeric():
        batch = int(10)
    else:
        int(batch)
    if int(batch) <= int(0):
        print(" Batch ammount is set too low. Adjusting batch to 10 . . .\n")
        batch = int(10)
    else:
        print(" Batch is set to: ", str(batch))
        print("\n")    

# This function allows the user to set the model name
def set_model_name():
    #access global variables
    global default_model_name
    global checkpoint_model_name

    #save default model filename
    response = input("\n Press 'Y' if you would like to set the filenames for the models," + 
                     " or press any other key to skip: ")
    if response == 'Y' or response == 'y':
        print("\n Sure . . .")
        #Default Model Name
        name = input("\n Please enter the new default model name: ")
        model_name = name + ".h5"
        print(" The Default model name is now set to: " , model_name)
        default_model_name = model_name
        
        #Checkpoint Model Name
        print("\n The current filename for the checkpoint model name is 'weights.best'" 
              + "\n appended by the accuracy rate and the .hdf5 extension.")
        print(" Changing the filename will not remove the accuracy rate or the .hdf5 extension.")
        name2 = input("\n Please enter the new checkpoint model name (default: weights.best): ")
        print("\n The checkpoint model name is now set to: " , name2 + "-{the accuracy value}.hdf5")
        checkpoint_model_name = name2
    else:
        print("\n Skipping model name change.")
        # Ensuring default model name stays default
        default_model_name = "Project_Bogie_Model.h5"
        print("\n Default model name is: ", default_model_name)
        print("\n Checkpoint model name is: weights.best-{the accuracy value}.hdf5")    

# This function allows the user to set the save path for the model
def set_model_save_path():
    # Ask the user if they want to set the save filepath for the model
    response = input("\n Enter 'Y' if you would like to set the save filepath for the default and checkpoint models: ")
    if response == 'y' or response == 'Y':
        print("\n Of course . . .\n")
    else:
        print(" Default save path selected.\n ")
        return

    # filepath
    filepath_found = False
    save_dir = ''

    # Prompt and check for valid path
    while not filepath_found:
        save_dir = input(" Please enter the filepath to the new save directory: ")
        
        # check if directory exists
        if not os.path.exists(save_dir):
            print("\n Filepath error: " + save_dir + " not found.\n")
        else:
            filepath_found = True
            print("\n Dataset directory found.\n")
            # Set the dataset directory to the new path
            global model_save_path
            model_save_path = save_dir

# This function allows the user to select a dataset directory path
def set_dataset_path():
    # Ask the user if they want to provide a different dataset dir
    print(" Project Bogie Trainer has a dataset directory that is hard coded.")
    response = input(" Enter 'Y' if you would like to set another dataset directory: ")
    if response == 'y' or response == 'Y':
        print("\n Of course . . .\n")
    else:
        print(" Default path selected.\n ")
        return

    # dataset
    datasetFound = False
    datasetDir = ''

    # Explain what is expected
    print(" For success, please ensure the directory path to the dataset contains classes (folders with images).")
    print(" You will need at least two classes (folders with images) to successfully train a model.\n")
    
    print(" The neural net will sort the classes inside the directory alphabetically, assigning them the value of 0-n.")
    print(" For example, class A (folder named A), will be assigned 0, class B will be assigned 1, etc,.\n")

    # Prompt and check for valid path
    while not datasetFound:
        datasetDir = input(" Please insert the directory location of the dataset: ")
        
        # check if directory exists
        if not os.path.exists(datasetDir):
            print("\n Filepath error: " + datasetDir + " not found.\n")
        else:
            datasetFound = True
            print("\n Dataset directory found.\n")
            # Set the dataset directory to the new path
            global dataSet_dir 
            dataSet_dir = datasetDir

    

# This function will create our model. Here we call VGG16. 
def create_model():
    # define a model variable and load the VGG16 information inside it.
    # it defines the expected shape of our files.
    model = VGG16(include_top=False, input_shape=(244, 244, 3))

    # Loop through and set the trainable flags to false for now.
    # We don't need to train them as VGG16 comes pretrained.
    for pre_existing_layer in model.layers:
        pre_existing_layer.trainable = False

    # Here we add new classifier layers to the model, see how they stack?
    # We'll use relu in class1 Dense(), and sigmoid in the output Dense()
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)

    # New model definition created
    # The inputs are taken from the VGG16 model, the outputs are define above.
    model = Model(inputs=model.inputs, outputs=output)

    # Compile and then return the model
    #opt = Adadelta(lr = float(learning_rate), rho = 0.95, epsilon = 1e-06)
    opt = Adam(lr=float(learning_rate), beta_1=0.9, beta_2=0.999, epsilon=0.1)
    #opt = SGD(lr=float(learning_rate), momentum=float(momentum_rate))
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Lets plot diagnostic data to see what is going on
def diagnostics(history):
    # plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='red', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='red', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

# This function will test and evaluate the model
# It takes a directory string for the sound we'd like to play
def train_and_evaluate_model(sound_url):
    # create the model
    model = create_model()

    # Here we create the data generator
    data_gen = ImageDataGenerator(featurewise_center=True, 
                                  width_shift_range=0.25, 
                                  height_shift_range=0.25, 
                                  horizontal_flip=True, 
                                  vertical_flip=False, 
                                  rotation_range=45, 
                                  brightness_range=[0.8, 1.2], 
                                  zoom_range=[0.4, 1.4])

    # Here we specify the image mean values for centering purposes
    data_gen.mean = [123.68, 116.779, 103.939]

    # Create the iterator for file handling
    training_iterator = data_gen.flow_from_directory(dataSet_dir,
        class_mode='binary', batch_size=int(batch), target_size=(244, 244))

    #callback
    global model_save_path
    #filepath = "C:/Users/Adam/source/repos/ProjectBogieTrainer/" + str(checkpoint_model_name) + "-{val_accuracy:.2f}.hdf5"
    filepath = str(checkpoint_model_name) + "-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath=model_save_path + filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]


    # Fit the model
    history = model.fit_generator(training_iterator, steps_per_epoch=len(training_iterator), 
        validation_data = training_iterator, epochs=int(epoch_rate), callbacks=callbacks_list, verbose=0)

    # Here we evalute the model
    _, acc = model.evaluate_generator(training_iterator, steps=len(training_iterator), verbose=0)
    print('Accuracy:')
    print('>%.3f' % (acc * 100.0))
    print("Learning Rate:")
    print(float(learning_rate))
    print('\n')
    

    # This will produce the learning curve graphs
    print('Summarizing daignostic curves...\n')
    diagnostics(history)
        
    # Now we save the model
    print("Saving Model...")
    model.save(model_save_path + default_model_name)

    # Play an alert when the model is done training. Replace the file with your own.
    winsound.PlaySound(sound_url, winsound.SND_FILENAME)
    print("Model now complete...\n")

def test():
    print("Epoch: ", epoch_rate)
    print("Learning rate: ", learning_rate)
    print("Momentum: ", momentum_rate)
    print("Batch: ", batch)
    print("Default Model Name: ", default_model_name)
    print("Checkpoint Model Name: ", checkpoint_model_name)
    print("Model save path: ", model_save_path)


# Welcome the user
welcome()

# Set the dataset dir
set_dataset_path()

# Setup important variables and model name
setup()
set_model_name()
set_model_save_path()
test()
# This is the entry point to our program. From here, all of the training begins.
train_and_evaluate_model(sound_url)