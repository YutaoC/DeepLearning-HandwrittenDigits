%% load the dataset
[images,labels] = loadTrainingImages(); %read the training images
[test_images,test_labels] = loadTestingImages(); %read the testing images

%% build and train the network
%bulid the network
layers = [
    imageInputLayer([150 150 3]) %input
    
    convolution2dLayer(10,6,'Padding','same') %conv
    batchNormalizationLayer %normalizes each input channel across a mini-batch
    reluLayer %ReLu
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(10,16,'Padding','same') %conv
    batchNormalizationLayer %normalizes each input channel across a mini-batch
    reluLayer %ReLu
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(10,32,'Padding','same') %conv
    batchNormalizationLayer %normalizes each input channel across a mini-batch
    reluLayer %ReLu
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(10,32,'Padding','same') %conv
    batchNormalizationLayer %normalizes each input channel across a mini-batch
    reluLayer %ReLu
    
    fullyConnectedLayer(10) %fully connected layer
    softmaxLayer %softmax
    classificationLayer]; %classifiction

%traning options
options = trainingOptions('sgdm', ... %training method
    'InitialLearnRate',0.01, ... %learning rate
    'MaxEpochs',4, ... %number of epoch
    'Shuffle','every-epoch', ... %data shuffling for every epoch
    'Plots','training-progress'); %show progress

%train the net
net = trainNetwork(images,labels,layers,options);

%% test the trained network
[predicted_labels,score] = classify(net,test_images); %test the neteork on the testing images
accuracy = mean(predicted_labels == test_labels'); %calculate the accuracy
display(accuracy)
