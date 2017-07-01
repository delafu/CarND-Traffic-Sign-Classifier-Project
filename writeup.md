**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/explorartory.jpg "Visualization"
[image2]: ./examples/sample_sign.jpg "Grayscaling"
[image3]: ./examples/augmented.jpg "Augmented"
[image4]: ./examples/Lenet.jpg "Lenet"
[image5]: ./examples/test_signs.jpg "Traffic Sign 1"
[image6]: ./examples/feature.jpg "Feature Map"



## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This is the writeup and here is a link to my [project code](https://github.com/delafu/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic and the pandas library to load the traffic sign codes and names
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed

![alt text][image1]

We can see that there are many more images of some classes than the others in the training set and in the validation set. We will generate some images for the classes that have less images than the average.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step I created some augmented images. I used zoom and rotation transformations in the images of the traffic signs that have less images than the average. I tried other transformations like translation but I didn´t obtain better results. Also I applied the transformations to all the training set but the training phase was very slow and the results were similar or worse.

I created a basic pipeline to preprocess the images:

* First I decided to apply a sharpener filter to get an image with better detail
* Second I equalized the image to get images with better contrast because there are images in the dataset with a very low and high contrast. I realized that equalization normalized the data
* Third I converted the images from RGB to gray scale

Here is an example of a traffic sign image in its original state, with the preprocess transformations applied and converted to gray scale  

![alt text][image2]


Here is an example of an original image and an augmented images:

![alt text][image3]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is a LeNet network

![alt text][image4]

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
|	Flatten					|	Output 400											|
|	Fully connected					|	Input 400 Output 120											|
| RELU					|												|
|	Fully connected					|	Input 120 Output 84											|
| RELU					|												|
|	Fully connected					|	Input 84 Output 43											|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the AdamOptimizer we used in the Lenet lab, a batch size of 32 because after a lot of tests I got the better results with it (The drawback is that is slower tan 64) and a learning rate of 0.01.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy 0.999
* validation set accuracy of 0.956 
* test set accuracy of 0.925

I did not take an iterative approcah choosing the architecture because I thought that I can get good results using the archictecture in the lessons and preprocessing the images.

If a well known architecture was chosen:
* What architecture was chosen? I chose Lenet architecture
* Why did you believe it would be relevant to the traffic sign application? Because It does a good job with images
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? I think It does a good work because test accuracy is above 0.92
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image5]


I used 7 different images to the test and one aditional image that is a cropped and centered version of the 6.jpg.
* 1.jpg: I think that this image is easy to recognize for the model. It´s centered, and is well focused and with good contrast. One of the difficulties is that the image has not the same aspect ratio of the input of the model and when I resize it, It can lost its shape and this is one of the most important parameters of the classifier model. 
* 10.jpg: This image has the difficulty that it has a watermark :-).
* 2.jpg: This image, like in the first one, has the difficulty of the aspect ratio
* 5.jpg: I think that this image is simple for the model
* 6.jpg: This is the most difficult image for the model. It´s not a perfect square and the stop sign is not centered. 
* 7.jpg: The most important difficulty of this image is that It´s a little dark but the training set has a lot of dark images.
* 8.jpg: The aspect ratio is the only difficult part of this image.
* 9.jpg: This is the cropped version of the 6.jpg image and I think that the model will work better with It.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right of the way at the next intersection      		| Right of the way at the next intersection  									| 
| No passing     			| No passing 										|
| No entry					| No entry											|
| 100 km/h	      		| 100 Km/h					 				|
| Yield			| Yield      							|
| Stop			| Go straight or right      							|
| Priority road			| Priority road     							|
| Stop			| Stop      							|



The model was able to correctly guess 7 of the 8 traffic signs, which gives an accuracy of 87.5%. 

Test Accuracy = 0.875
Class Speed limit (100km/h): 1.0/1   100.0%
Class No passing: 1.0/1   100.0%
Class Right-of-way at the next intersection: 1.0/1   100.0%
Class Priority road: 1.0/1   100.0%
Class Yield: 1.0/1   100.0%
Class Stop: 1.0/2   50.0%
Class No entry: 1.0/1   100.0%

This is a little lower comparing with the test dataset but in theese images we are using images with different aspect ratio and in one of them the traffic sign is not centered and I think that this image is very difficult for the model. To solve it I have to improve the preprocessing algorithm detecting the sign and cropping It in a perfect square.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 18th cell of the Ipython notebook and includes the visualization in a bar chart. I made a mistake and I only provide the top 3 softmax probabilities. 

In all of them the model is sure of Its predictions with porcentages above 99% in all the samples. 

#### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][image6]

Watching the features map generated by the first convolutional layer, I think that the most important features for the model are the shapes detected in the images.


