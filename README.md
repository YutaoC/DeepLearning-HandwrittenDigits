# hand-written-digit-recognition
hand-written digit recognition and a way to implement transfer learning

### -------------------------SVM--------------------------  
## main.m   
This file implement the SVM both training and testing using 'one vs. one' and 'one vs. all' algorithm     

## loadMNISTImages.m    
This function read the MNIST image data set and reshape it into desired format 
images - the matrix contains all the images and every column is a image   
filename - MNIST file name   

## loadMNISTLabels.m
This function read the MNIST label data set and reshape it into desired format  
labels - a row vector and every element is the label of corresponding image  
filename - MNIST file name   

### ---------------------DeepLearning-----------------------
## main.m   
This file implement the convolutional neural network   

## GetMNISTData.m   
This function load the training and testing data and reshape them into a desired format   
imdb - a structure contains all the images and labels along with some other information about the structure   

## loadMNISTImages.m   
This function read the MNIST image data set and reshape it into desired format  
images - the matrix contains all the images and every column is a image  
filename - MNIST file name   

## loadMNISTLabels.m  
This function read the MNIST label data set and reshape it into desired format   
labels - a row vector and every element is the label of corresponding image   
filename - MNIST file name   

## PrepareData_MNIST.m   
This function rearrange and prepare the data for training and testing   
opts - the structure contains the training and testing data   

## NetInit_MNIST.m   
This function initialize the layers of the network   
opts - the structure contains the training and testing data obtained in the function 	"PrepareData_MNIST"   
net - the designed CNN   

### ---------------------TransferLearning-----------------------
## simpleCNN.m   
This file build a simple CNN with 4 convolution layers and trained and and tested it on the 10-monkey-species dataset  

## TransferLearning.m  
This file implement the transfer learning. The fine tuned network is also implemented in this file   

## loadTrainingImages.m
This function read training images and their corresponding labels from a folder and resize them into the form that is suitable for alexnet   
training_images - a WxHxCxD 4-D matrix where W and H represent the width and height of the 	image,C represents the number of channels and D represents the number of training 	images.The images are resized into the size that is suitable for alexnet   
training_labels - a vector contains the labels of training images    

## loadTestingImages.m
This function read testing images and their corresponding labels from a folder and resize them into the form that is suitable for alexnet     
training_images - a WxHxCxD 4-D matrix where W and H represent the width and height of the 	image, C represents the number of channels and D represents the number of testing 	images. The images are resized into the size that is suitable for alexnet   
training_labels - a vector contains the labels of testing images    
