
**Behavioral Cloning Project**

This is one of the exicting project so far in term-1. Car is able to drive autonomously with no manual intervention just using deep learning with no lidar or GPS information. Implementation is based on Nvidia's paper(https://arxiv.org/abs/1604.07316)

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* ReadMe.md summarizing the results


Model architecture is inspired by Nviida's End to End Learning for Self-Driving Cars paper with few modifications. Below is detailed description of model:

Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 70, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 70, 320, 3)        12        
_________________________________________________________________
activation_1 (Activation)    (None, 70, 320, 3)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 33, 158, 24)       1824      
_________________________________________________________________
activation_2 (Activation)    (None, 33, 158, 24)       0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 15, 77, 36)        21636     
_________________________________________________________________
activation_3 (Activation)    (None, 15, 77, 36)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 37, 48)         43248     
_________________________________________________________________
activation_4 (Activation)    (None, 6, 37, 48)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 35, 64)         27712     
_________________________________________________________________
batch_normalization_1 (Batch (None, 4, 35, 64)         256       
_________________________________________________________________
activation_5 (Activation)    (None, 4, 35, 64)         0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 4, 35, 64)         0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 2, 33, 64)         36928     
_________________________________________________________________
activation_6 (Activation)    (None, 2, 33, 64)         0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 2, 33, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 4224)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 1164)              4917900   
_________________________________________________________________
batch_normalization_2 (Batch (None, 1164)              4656      
_________________________________________________________________
dropout_3 (Dropout)          (None, 1164)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 100)               116500    
_________________________________________________________________
batch_normalization_3 (Batch (None, 100)               400       
_________________________________________________________________
dense_3 (Dense)              (None, 50)                5050      
_________________________________________________________________
dropout_4 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 11        
=================================================================
Total params: 5,176,643
Trainable params: 5,173,987
Non-trainable params: 2,656

Data from inputs images is normalized in model itself so that no preprocessing is required for test data. Top 70 pixels and bottom 20 pixels are cropped to remove background data not suitable for training. Dropout and batch normalization is used to avoid over fitting.

The model used an adam optimizer, so the learning rate was not tuned manually.

###Training data
Training data was collected by driving car in Udacity's simulator. Training data include: 1 lap of center driving, 1 lap of center driving in opposite direction, 1 lap of driving along the edge.
To recover from edge to center of road, additional training data was captured where car drove from edge to center, this can be seen in captured video where car is able to recover from the edge. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 15 as validation accuracy started decreasing thereafter. I used an adam optimizer so that manually training the learning rate wasn't necessary.

To avoid turning at high speed, throttle was decreased for sharp left turns.

Generators for training and validation data was used in order to avoid loading whole dataset in memory. Model was trained on Nvidia's GTx1080 card.

With just training on track-1 car is able to complete Jungle track lap with just 3 manual interventions. Since track2 has more elevations and sharp turns more data needs to be captured.

Future improvements:

1. Capture more data on track2 which has high elevations and sharp turns.
2. Add data augumentation.
3. Try Removing 3,1,1, convolutional layer and convert input image from RGB-YUV. 