# **Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

[//]: # (Image References)

[image1]: ./readme/label_count.png "Visualization"
[image2]: ./readme/grayscale.jpg "Grayscaling"
[image3]: ./readme/accuracy_x_epoch.png "Graph"
[image4]: ./readme/speed-limit-60.png "Speed Limit 60"
[image5]: ./readme/end-of-no-overtaking.png "End of No Overtaking"
[image6]: ./readme/bicycle-crossing.png "Traffic Sign 3"

<!-- [image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5" -->

[image9]: ./readme/test_images.png "Web Test Images"
[image10]: ./readme/test_images_gray.png "Web Test Images Grayscale"
[image11]: ./readme/bar_softmax_1.png "Softmax Results for Image 1"
[image12]: ./readme/bar_softmax_2.png "Softmax Results for Image 2"  
[image13]: ./readme/bar_softmax_3.png "Softmax Results for Image 3"  
[image14]: ./readme/bar_softmax_4.png "Softmax Results for Image 4"  
[image15]: ./readme/bar_softmax_5.png "Softmax Results for Image 5"  
[image16]: ./readme/bar_softmax_6.png "Softmax Results for Image 6"  

[image20]: ./readme/conv1_feature_map.png "Feature map for the first Convolution Layer"  
[image21]: ./readme/conv2_feature_map.png "Feature map for the second Convolution Layer"  
[image22]: ./readme/conv3_feature_map.png "Feature map for the third Convolution Layer"  
[image23]: ./readme/conv1_filter.png "Feature map for the first Convolution Layer Method 2"  
[image24]: ./readme/conv2_filter.png "Feature map for the second Convolution Layer Method 2"  
[image25]: ./readme/conv3_filter.png "Feature map for the third Convolution Layer Method 2"  

---

### Data Set Summary & Exploration

#### 1. Data Set Summary
I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the data from the validation set. The chart shows the count of validation images for each label.

###### Count of each label/class
![alt text][image1]

### Design and Test a Model Architecture

#### 1. Pre-processing

The pre-processing of the data sets include grayscaling and normalization. With grayscaling a matrix multiplier was used so that all the images could be processed in fewer lines of code. This allows the pipeline to gather necessary data from the images without losing the defining characteristics of the images. The data was then normalized to confine the matrix data between 0 and 1. Normalizing the data does not alter the shape of the image.

###### Grayscaling
![alt text][image2]

#### 2. Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input | 32x32x1 Grayscale image |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU	||
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 24x24x32 	|
| RELU	||
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 20x20x64 	|												
| RELU	||
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Flattening   |  output 6400									|
| Fully connected		| input  6400, ouput 1600 |
| RELU  ||
| Fully connected		| input  1600, ouput 400 |
| Dropout ||
| Fully connected		| input  400, output 43 |
| Dropout |||


#### 3. Hyperparameters

To train the model, I initially used a

* batch size of 256
* epoch length of 15
* Adam optimizer with a learn rate of .001

After Each convolutional layer, I added a RELU and flattened the layer before sending it to the fully connected ones.

Using the model provided by the Lenet lab. The accuracy would not increase above 90%. Using an interative approach. While varying all the hyperparameters, the most notable difference came from increasing the learning rate from .0001 to .001.

###### Accuracy Line Graph
![alt text][image3]

#### 4. The Scientific Approach

I started off with the architecture model borrowed from Lenet Lab solution. From there, I increased the number of convolutional layers. What this did was increase the initial testing and validation accuracy. To increase the accuracy further, I tried adding max pooling and dropout after every single layer. This obviously proved to be inefficient and inaccurate, but I began removing max pooling and dropout 1 by 1. When max pooling is set after the convolutional layers, it alters the input, drastically reducing it for the fully connected layer. Drop out layers after the last fully connected layers were more beneficial to the accuracy than not having them there. After concluding with this architecture, the testing accuracy would only stay around .90. Even altering the epoch count to 30 and increasing the batch size did not drastically change any results. After raising the learning rate to .001, the model consistently achieved a .95 or above testing accuracy rate.

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of .96
* test set accuracy of .954

### Test a Model on New Images

#### 1. German Traffic Signs From the Web

###### Images from the web:

![alt text][image9]

###### Grayscale
![alt text][image10]

#### 2. Model Prediciton

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Keep Right      		| Keep Right  									|
| Speed Limit 30     			| Speed Limit 30								|
| Speed Limit 30					  | Speed Limit 30											|
| Speed Limit 60	      		| Speed Limit 60				 				|
| Priority Road		| Priority Road      							|
| Roadwork	| Roadwork   							|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. Although in this certain instance, there is a 100% accuracy, the model will sometimes get the roadwork sign incorrect resulting in 83%.

#### 3. Softmax Analysis

For the first image, the model is completely sure of its prediction with a near 100% accuracy rating. All the other probabilities are negligible with values less than 1.0e-9.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| Keep Right  									|
| 0.00     				| Wild Animal Crossing 										|
| 0.00					| Speed limit (120km/h)											|
| 0.00	      			| No Passing					 				|
| 0.00				    | Right-of-way At The Next Intersection     							|

![alt text][image11]

<br>

The model had a more difficult time with predicting the second image. Although the highest accuracy was 52.2% no other image probability was half of that amount. It makes sense that images with any probability at all are image with that have details encapsulated within a circular boundary. Pre-proccessing images to grayscale is one possibility as to why the priority road sign is within the top 5 results as its distinct trait is a yellow border around the center detail.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.5         			| Speed limit (30km/h)  									|
| 0.2     				| Stop										|
| 0.1					| End of speed limit (80km/h)	|
| 0.06	      			| Speed limit (60km/h)					 				|
| 0.04				    | Priority road     							|

![alt text][image12]

<br>

Using a different image with the same label as image 2, image 3 had a much better accuracy rating. This is believed to be from the data gained from the analysis of image 2. The top 5 results are also very similar to image 2 in that it they all exhibit details being encapsulated.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| Speed limit (30km/h)  									|
| .00     				| Speed limit (20km/h) 										|
| .00					| Stop									|
| .00	      			| Priority road					 				|
| .00				    | Speed limit (80km/h)     							|

![alt text][image13]

<br>

Much like image 1, image 4 had a prediction accuracy of near 100%. The top 2 and 3 predictions have results above .1% and is understandable as they are also speed limit traffic signs. Even with very low probabilities, results 4 and 5 are very interesting. Bicycle crossing is encapsulated by a triangle where as Speed limit (60km/h) and End of no passing's inner detail are enclosed in circles. Between these 3 signs, they all exhibit inner details that have 2 loops. This shows how the model is prioritizing its pattern recognition in a very logical way.


| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.97         			| Speed limit (60km/h)  									|
| 0.03     				| Speed limit (20km/h)										|
| .00					| Speed limit (120km/h)											|
| .00	      			| End of no passing			 				|
| .00				    | Bicycles Crossing     							|

![alt text][image14]

###### Image Comparison [Speed Limit 60, End of No Overtaking, Bicycle Crossing]
![alt text][image4] ![alt text][image5] ![alt text][image6]

<br>

Image 5 also has a near 100% prediction score. It shows that the model is very certain of its classification. With the majority of the labels count skewed toward the first few images, it is understandable that they are recognized more.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| Priority road 									|
| .00     				| No vehicles									|
| .00					| Roundabout mandatory										|
| .00	      			| Stop				 				|
| .00				    | Speed limit (80km/h)     							|

![alt text][image15]

<br>

Of all the test images from the internet, image 6 has the most intricate details. It is expected that this image has a more difficult time being classified. Although not as high as others. Roadwork scored a probability of 83%. The uncertainty trails beyond the top 5 results show that the image could use a little more training of image. Perhaps with a more evenly distribution of the labels would help in raising the certainty.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .83         			| Roadwork  									|
| .12     				| Speed limit (20km/h)									|
| .01					| Keep left											|
| .01	      			| Right-of-way at the next intersection				 				|
| .01				    | Bumpy Road     							|

![alt text][image16]

<br>

### (Optional) Visualizing the Neural Network
#### 1. Feature Maps

The first convolutional layer went from a 32x32 to a 28x28 image and the defining characteristics of the image are still present. A human can still accurately recognize what the sign is despite the grayscaling. The depth of the layer is represented as a single feature map. It is worth noting that the "coloring" of each feature map is altered in some way. Feature map 0 and 8 seem to be inversions of each other. The neural network blurs the image and attempts to discern shapes that are relatively close in color.

###### Convolutional Layer 1
![alt text][image20]

In the second convolutional layer, the image is blurred even more and the depth increased by a factor of 2. There is more variance of the grayscaling and the defining features of the traffic sign are still visible.

###### Convolutional Layer 2
![alt text][image21]

In the final convolutional layer, the images are blurred even more because of the filtering, and some of the more defining characteristics are spread out into different feature maps.

###### Convolutional Layer 3
![alt text][image22]

The following are feature maps of another image using a different, but essentially the same method.

###### Convolutional Layer 1
![alt text][image23]

###### Convolutional Layer 2
![alt text][image24]

###### Convolutional Layer 3
![alt text][image25]
