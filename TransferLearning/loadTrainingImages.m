function [training_images,training_labels] = loadTrainingImages()
%
% This function read training images and their corresponding labels from 
% a folder and resize them into the form that is suitable for alexnet
%
% [training_images,training_labels] = loadTrainingImages()
%
% training_images - a WxHxCxD 4-D matrix where W and H represent the width 
%                   and height of the image, C represents the number of 
%                   channels and D represents the number of training images.
%                   The images are resized into the size that is suitable 
%                   for alexnet.
% training_labels - a vector contains the labels of training images
%
% Yutao Chen
% 12/13/2018
%
    training_images = [];
    training_labels = [];
    for i = 1:10 %for every folder
        D = sprintf('/Users/cheny/Downloads/10-monkey-species/training/n%d',i-1); %the loaction of the folders
        S = dir(fullfile(D,'*.jpg')); %load all the items end with .jpg
        for k = 1:numel(S) %for every image
            F = fullfile(D,S(k).name); %the name of the image
            I = imread(F); %read the image
            training_images(:,:,:,end+1) = imresize(I,[227 227]); %resize the image
            training_labels(:,end+1) = i; %store the labels
        end
    end
    training_images = training_images(:,:,:,(2:end)); %the fist one is a zero matrix
    training_labels = categorical(training_labels); %make it a categoty
end