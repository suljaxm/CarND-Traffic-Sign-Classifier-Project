# **Traffic Sign Recognition** 
---
A traffic sign classifier is build using the LeNet deep neural network topology. Given a dataset containing images from 43 distinct classes from the German Traffic Sign Dataset, a classifier is built that obtains a 95% classification accuracy on a validation set. Class imbalances are addressed and corrected for, and an exploratory analysis of the dataset is performed.



**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/visualize_the_train_images.png "Visualization"
[image1_1]: ./writeup/train_data_bar.png "Bar"

[image4]: ./writeup/2.jpg "Traffic Sign 1"
[image5]: ./writeup/5.jpg "Traffic Sign 2"
[image6]: ./writeup/4.jpg "Traffic Sign 3"
[image7]: ./writeup/1.jpg "Traffic Sign 4"
[image8]: ./writeup/3.jpg "Traffic Sign 5"

[image2]: ./writeup/70.png  "70km/h"
[image3]: ./writeup/outputFeatureMap.png  "routputFeatureMap"

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](./Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is  3X32X32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribution.
![alt text][image1_1]
and some of the images are shown below,
![alt text][image1]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

I only normalize the pixel value of the image to [-1,1], and the code is ```(X - 128.0)/128.0```

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        		| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   		| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6	|
| RELU					|				|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16     |
| RELU					|				|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16	|
| Fully connected		| inputs 400, outputs 120    |
| Fully connected		| inputs 120, outputs 84    |
| Fully connected		| inputs 84, outputs 43    |
| Softmax						|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used hyperparameters as follows,
* the optimizer  is ```Adam```
* the rate of learning  is ```0.001```
* the size of batch is ```128```
* the number of epochs  is ```100```


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.953
* test set accuracy of 0.933

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

These images might be easy to classify because the images are clear and less distracting. In other words, we need to make sure the logo takes up all of the image. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|  
| Keep right   			| Keep right									|
| Pedestrians			| Speed limit (70km/h)     		|
| Stop	      		| Stop					 				|
| Turn right ahead      		| Turn right ahead  			|
| Speed limit (70km/h)					| Speed limit (70km/h)		|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the ```Step 3``` cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 1.0), and the image does contain a ```Keep right``` sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00        			| Keep right     									| 
| 3.91e-16     				| Yield									|
| 2.59e-37					|Speed limit (80km/h)											|
|.00	      			|Speed limit (20km/h)					 				|
| .00				    |Speed limit (30km/h)      							|


For the second image,

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00        			| Speed limit (70km/h)     									| 
| 3.91e-16     				| Keep left									|
| 2.59e-37					|Roundabout mandatory											|
|.00	      			|Speed limit (30km/h)					 				|
| .00				    |Go straight or left    							|

For the third image,

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.999e-01        			| Stop    									| 
| 7.98e-05     				| Speed limit (60km/h)									|
| 5.62e-05					|Speed limit (80km/h)											|
|3.92e-06	      			|Yield					 				|
| 2.21e-11				    |No vehicles     							|

For the forth image,

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|9.97e-01        			| Turn right ahead    									| 
| 3.32e-03     				| Ahead only									|
| 2.89e-17					|Roundabout mandatory											|
|8.97e-32	      			|Keep left					 				|
| 1.43e-32				    |No passing      							|

For the fifth image,

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00        			| Speed limit (70km/h)     									| 
| 1.53e-08     				| Speed limit (20km/h)										|
| 1.31e-23					|  Speed limit (30km/h)											|
|2.91e-35	      			| Dangerous curve to the left					 				|
| 1.86e-35				    |Keep left      							|

### Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

We can see the characteristic graph of 70km/h under the first convolution layer, which clearly shows the outline of 70. It shows that the network is not so ''black box".

input:
![alt text][image2]
outputs:
![alt text][image3]
