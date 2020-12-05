##################################################################################
# Author:   Adam Tipton
# Title:    Project Bogie Predictor
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
#   Project Bogie uses TensorFlow as a backend and Keras as a driving force for training
#   and predicting.
#
#   The Predictor program will take the model created by the Project Bogie Trainer.
#   The model will then be loaded into the predictor program. The user well then point
#   to an image, or set of images, for the model to predict an outcome. Possible outcomes
#   will be does the image contain a modern military aircraft or something else. 
#
# Sources:
#   Much inspiration and reusable code base is taken from the following tutorial -
#   URL: https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/
###################################################################################   
###################################################################################

# These imports are import for image loading and processing
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# These imports are important for file handling
import os
from os import listdir


# Welcome the user
def welcome():
	#prompts
	print("\n\n")
	print(" ############################################################################################")
	print(" ############################################################################################")
	print(" ##################################                         #################################")
	print(" ##################################      PROJECT BOGIE      #################################")
	print(" ##################################        PREDICTOR        #################################")
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
	print(" #  Project Bogie Predictor uses a convolutional nueral network model for image             #")
	print(" #  classification. The model will predict whether or not it is likely that a given input   #")
	print(" #  image would contain a modern military aircraft (a BOGIE) or something else.             #")
	print(" #                                                                                          #")
	print(" #  The predictor program will need a model with extension .h5 or hdf5, trained against a   #")
	print(" #  data set of two classes. One class containing only modern military aircraft, and        #")
	print(" #  another class containing non-military and other images. The program will also need an   #")
	print(" #  image file with either of the following extensions (.jpeg, .jpg, .png).                 #")
	print(" #                                                                                          #")
	print(" #  What you need to provide:                                                               #")
	print(" #                                                                                          #")
	print(" #                     1. The filepath to the predictive model. Extension .h5 or .hdf5      #")
	print(" #                     2. The filepath to the image file. Extension .jpg/.jpeg/.png         #")
	print(" #                                                                                          #")
	print(" ############################################################################################")
	print(" ############################################################################################")
	print("\n")
	print("\n")
	print(" You will now need to provide the filepath location to the predictive model and test image. \n")

# Prompt the user for model
def getModel():
	
	
	#model
	modelFound = False
	modelDir = ''
	
	# Valid model path and file check 
	while not modelFound:
		modelDir = input(" Please insert the directory location of the predictive model: ")
		modelExt = os.path.splitext(modelDir)
		
		# check if model[0] is valid dir
		if not os.path.exists(modelDir):
			print("\n Filepath error: " + modelDir + " not found.\n")
		# check if model[1] is a valid extension
		elif (modelExt[1] == '.h5') or (modelExt[1] == '.hdf5'):
			modelFound = True
			print(" Model found.\n")
		else:			
			print("\n Error: .h5 or .hdf5 model not found.")
			print(" Please select a compatible model (<model name>.h5, <model name>.hdf5) \n")
	
			

	return modelDir;

# Prompt the user for an image 
def getImage():
		
	#image
	imageFound = False
	imageDir = ''
	
				
	# Valid image path and file check	
	while not imageFound:
		imageDir = input(" Please insert the directory location of the image file you want to test: ")
		imageExt = os.path.splitext(imageDir)
		
		# check if imageDir[0] is valid dir
		if not os.path.exists(imageDir):
			print("\n Filepath error: " + imageDir + " not found.\n")
		# check if imageExt[1] is a valid extension
		elif (imageExt[1] == '.jpg') or (imageExt[1] == '.jpeg') or (imageExt[1] == '.png'):
			imageFound = True
			print(" Image found.\n")
		else:			
			print("\n Error: .jpg/.jpeg or .png image not found.")
			print(" Please select a compatible image type (<image name>.jpg/.jpeg, <image name>.png) \n")
		

	return imageDir;

# Prompt the user for an image director of up to 5 images
def getImageDirectory():

	#image 
	imageDirFound = False
	imageDir = ''
	imageExtensionFound = False

	# Valid image path and file check
	while not imageDirFound and not imageExtensionFound:
		imageDir = input(" Please insert the directory location of the set of images you want to test: ")

		# check if imageDir is a valid dir
		if not os.path.exists(imageDir):
			print("\n Filepath error: " + imageDir + " not found.\n")
		else:
			print(" Image directory found. \n")
			print(" Testing for valid images. \n")

		# Check that the directory contains valid images	
		counter = 0
		for filename in os.listdir(imageDir):
			if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
				counter += 1								
				imageDirFound = True
		
		print(" Directory contains " + str(counter) +  " compatible images. \n")		

	return imageDir

# Choose between loading a single image versus an image dir
def promptImageSelection():
	# the imageDir variable
	imageDir = ''
	response = input(" Press 1 if you want to load a single image, 2 if you want to load an image directory (<= 5 images): ")
	if not response.isnumeric():
		imageDir = getImage()
	elif int(response) == 1:
		# call the function for a single image
		imageDir = getImage()
	elif int(response) == 2:
		# call the function for a directory of images
		imageDir = getImageDirectory()

	return imageDir

# load and prepare the image for processing by the predictor
def loadImage(imageDir):
	# load the image
	img = load_img(imageDir, target_size = (244, 244))

	# convert the image to an array
	img = img_to_array(img)

	# reshape the img to have 3 channels
	img = img.reshape(1, 244, 244, 3)

	# declare the center for the pixel data
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]

	return img


# load the model and make the prediction against the image
def makePrediction(model, imageDir, singleImage):
	
	if singleImage == True:
		# take image and load it
		print("\n Loading image...\n")
		img = loadImage(imageDir)
		print(" Image loaded\n")
		
		# load the model
		print(" Loading model...\n")
		model = load_model(model)
		print("\n")
		print("**************")
		print("**************")
		print(" Model loaded")
		print("**************")
		print("**************\n")

		# make sure the user is ready to continue
		userReady = False
		while not userReady:
			response = input(" Enter 'p' if you are ready to make the (p)rediction: ")
			if response == 'p' or response == 'P':
				userReady = True

		# make the prediction
		print("\n")
		print(" ##################################")
		print(" ##################################\n")
		print(" Making a prediction...\n")
		prediction = model.predict(img, verbose = 0)
	
		print("\n")
		print(" Numerical prediction value: ")
		print(prediction[0])
		print("\n")
		
		# Display the results
		if prediction[0] <= 0.05:
			print(" ******************")
			print(" *                *")
			print(" * Bogie DETECTED *")
			print(" *                *")
			print(" ******************")
			print(" This file possibly contains a military aircraft.\n")
		elif prediction[0] > 0.05:
			print(" *****************")
			print(" *               *")
			print(" *  CLEAN image  *")
			print(" *               *")
			print(" *****************")
			print(" Miliatry aircraft NOT found.\n")
		
	elif singleImage == False:
		modelLoaded = False
		
		# take the image directory, process and load it
		#Military counter
		isMilitary = 0

		#something else counter
		somethingElse = 0

		#image counter
		imageCounter = 0

		# first loop through the dir
		for filename in os.listdir(imageDir):
			
			if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
				#load the image
				img = loadImage(imageDir + '/' + filename)
				imageCounter += 1

				# take model and load it
				if not modelLoaded:				
					print(" Loading model...\n")
					model = load_model(model)
					modelLoaded = True
					print("\n")
					print("**************")
					print("**************")
					print(" Model loaded")
					print("**************")
					print("**************\n")

					# make sure the user is ready to continue
					userReady = False
					while not userReady:
						response = input(" Enter 'p' if you are ready to make the (p)rediction: ")
						if response == 'p' or response == 'P':
							userReady = True

				# make the prediction
				print("\n")
				print(" ##################################")
				print(" ##################################\n")
				print("\n Making a prediction...\n")
				prediction = model.predict(img, verbose = 0)
	
				print(" Numerical prediction value: ")
				print(prediction[0])
				print("\n")
		
				# Display the results
				if prediction[0] <= 0.05:
					isMilitary += 1
					print(" ******************")
					print(" *                *")
					print(" * Bogie DETECTED *")
					print(" *                *")
					print(" ******************")
					print(" Image " + filename + " possibly contains a military aircraft.\n")
				elif prediction[0] > 0.05:
					somethingElse += 1
					print(" *****************")
					print(" *               *")
					print(" *  CLEAN image  *")
					print(" *               *")
					print(" *****************")
					print(" Miliatry aircraft NOT found in image " + filename + "\n")

				# We only allow up to 5 images, break out of the loop
				if imageCounter == 5:
					break

		#ratio
		milRatio = isMilitary / imageCounter
		elseRatio = somethingElse / imageCounter

		print(' ' + str(isMilitary) + ' out of ' + str(imageCounter) + ' images were classified as containing Military Aircraft.')
		print('\n Breakdown: ')
		print(' ' + str(milRatio * 100) + '% images were thought to be military...')
		print(' ' + str(elseRatio * 100) + '% images were thought to be something else...\n\n')

# start() is the primary workhorse of the predictor
def start():
	# Print the welcome message
	welcome()
	
	# Get the model
	model = getModel()
		
	# Give the user a choice between single image or directory
	imageDir = ''
	singleImage = False
	response = input(" Press 1 if you want to load a single image, 2 if you want to load an image directory (up to 5 images): ")
	if not response.isnumeric():
		imageDir = getImage()
		singleImage = True
	elif int(response) == 1:
		# call the function for a single image
		imageDir = getImage()
		singleImage = True
	elif int(response) == 2:
		# call the function for a directory of images
		imageDir = getImageDirectory()
		singleImage = False	
		
	# Make the prediction
	print(" Getting ready to make the prediction.")
	makePrediction(model, imageDir, singleImage)

# Get things rolling
start()

