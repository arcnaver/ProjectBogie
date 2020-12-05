# Project Bogie
**Author:** Adam Tipton
**School:** Brigham Young University - Idaho
**Course:** CSE 499 Senior Project Fall 2020

##Description:
Project Bogie is a Convolutional Neural Network written in the Python language. I chose this idea for two reasons, first, I really wanted to know how CNN's worked, second, I've never coded in Python before. This project taught me a lot about both topics and proved to myself that I'm able to learn new concepts and new ideas and implement them into a significant project. 

**Trainer**
The trainer program uses the Keras framework and the TensorFlow backend to create a trained model. Models are trained against a robust set of classes containing images. While one class contains images of modern military aircraft, the other contains non-military aircraft and random images. This allows the trainer to create a model that identifies with 95% accuracy rate images that contain modern military aircraft **Bogie** and those that do not **Clean**. The trained model is produced in the .h5 and .hdf5 format.

**Predictor**
The predictor program relies on the model created with the trainer program. It takes a .h5 or .hdf5 model and predicts if a given image or set of images contains a modern military aircraft or not. The predictor allows the user the option of testing either a singular image or a directory containing up to 5 images at once. This limitation can be increased by tinkering with the code, however, depending upon the system, memory management will be an issue. If multiple images are selected, the predictor will provide a percentage readout of how many images were **bogies**. 

##Project Score:
The final project score is **99%**

##Sources:
Much inspiration and reusable code base is taken from the following tutorial -
[URL:] (https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/)

##License:
As part of a BYU-I course, this code is owned by BYU-I. **NO LICENSE** is intended. 


