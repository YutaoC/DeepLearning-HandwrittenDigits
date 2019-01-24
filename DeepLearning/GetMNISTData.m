function imdb = GetMNISTData()
%
% This function load the training and testing data and reshape them into a
% desired format
%
% imdb = GetMNISTData()
%
% imdb - a structure contains all the images and labels along with some
%        other information about the structure
%
% Yutao Chen
% 12/10/2018
%
   images = loadMNISTImages('train-data'); %read the training images
   images = permute(reshape(images,28,28,60e3),[2 1 3]); %reshape the images
   labels = loadMNISTLabels('train-labels')+1; %read the labels of training images
   test_images = loadMNISTImages('test-data'); %read the testing images
   test_images = permute(reshape(test_images,28,28,10e3),[2 1 3]); %reshape the images
   test_labels = loadMNISTLabels('test-labels')+1; %read the labels of testing images
   
   set = [ones(1,numel(labels)) 3*ones(1,numel(test_labels))]; %the set used to identify the testing images and training images
   images_all = single(reshape(cat(3,images,test_images),28,28,1,[])); %reshape the training anf testing images into one 4-D matrix
   images_mean = mean(images_all(:,:,:,set == 1),4); %calculate the mean value of all the images
   images_all = bsxfun(@minus,images_all,images_mean); %subtract the mean
   
   imdb.images.data = images_all; %all the images
   imdb.images.data_mean = images_mean; %the mean value of the images
   imdb.images.labels = [labels,test_labels]; %all the labels
   imdb.images.set = set; %the set
   imdb.meta.sets = {'train','val','test'}; %name of the sets
   imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false); %10 classes   
end