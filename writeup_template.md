#**Behavioural Cloning** 

##Write-up

**Behavioural Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behaviour
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points

###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---

###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of the CNN architecture. 

- line 76 - Deals with lambda to plug in the images of format
- line 77 - Cropping images to fit only the road
- line 78 to 82 going through the layers with 5x5 kernels in the first three layers, 3x3 kernels for the last two layers.
- Flatten layers for the model.
- The next 10 layers goes through different neurons and dropout functions. It's best to dropout in the first two layers then after those layers to use a dropout at .2

####2. Attempts to reduce overfitting in the model

the model does use dropout layers mentioned above, an appropriate amount has been used to help with overfitting. Trained with enough data to help the car not to go over the track or hit the track.

####3. Model parameter tuning

Using the model compile to go through a loss of mean squared error, an adam optimizer and a manual input of a learning rate of 0.0001.

####4. Appropriate training data

The data collected was 5 laps of me driving as perfect as I could in the center of the lane. There is some instances where my car went over a bit on the curb, which helped in some instances, if my car went the wrong way.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving model architecture was to follow with the recommended model of NVIDIA approved of.

My first step was to use a convolution neural network model similar to the CNN architecture or NVIDIA, I thought this model might be appropriate because recommendations, how many layers it uses and the appropriate amount to get the perfect amount of data.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I also flipped the images to help the car make sure that it was on the right lane.

To combat the overfitting, I modified the model so that it had dropouts at the the appropriate stages.

Then I used a learning rate of 0.0001, collected enough data and made sure each image was rent to the right 'camera angle' for the proper calculation.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle was going to hit the curb, but I improved this with flipping the images to the proper channels so the car stays on the right lanes.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 73-96) consisted of a convolution neural network with various  layers and layer sizes.

Here is the code of the architecture.

model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row, col, ch), output_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1164))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam', lr=0.0001)

####3. Creation of the Training Set & Training Process

To capture good driving behaviour, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text](https://github.com/TheAisBack/CarND-Behavioral-Cloning-P3/blob/master/center-image.jpg "Center Image")

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct itself. But mainly stayed on the center of the track. I completed 5 laps with the car staying center most of the track.

Here is an example of three images from left, center and right. Majority of my driving was focused to stay centered and with the model.py, the car made sure to stay perfectly center even with tough turns.

![alt text](https://github.com/TheAisBack/CarND-Behavioral-Cloning-P3/blob/master/left1.jpg "Left Image")![alt text](https://github.com/TheAisBack/CarND-Behavioral-Cloning-P3/blob/master/center1.jpg "Center Image")![alt text](https://github.com/TheAisBack/CarND-Behavioral-Cloning-P3/blob/master/right1.jpg "Right Image")

To augment the data sat, I also flipped images and angles thinking that this would help if the track came up to random objects that could affect the car to react differently.

After the collection process, I had 7197 images of each point of the camera. I then pre-processed this data by looping through the images and appended them to the right location.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 25, it was helpful to filter through this amount especially with a GPU to get the correct data. I used an adam optimizer because of it being the best tool for CNN architecture and a fast convergence, but manually entered a learning rate at 0.0001, instead of the 0.001, this helped to perfect the model.
