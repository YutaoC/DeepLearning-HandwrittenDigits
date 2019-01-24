function [testing_images,testing_labels] = loadTestingImages()
%
% This function read testing images and their corresponding labels from 
% a folder and resize them into the form that is suitable for alexnet
%
% [testing_images,testing_labels] = loadTestingImages()
%
% training_images - a WxHxCxD 4-D matrix where W and H represent the width 
%                   and height of the image, C represents the number of 
%                   channels and D represents the number of testing images.
%                   The images are resized into the size that is suitable 
%                   for alexnet.
% training_labels - a vector contains the labels of testing images
%
% Yutao Chen
% 12/13/2018
%
    testing_images = [];
    testing_labels = [];
    for i = 1:10 %for every folder
        D = sprintf('/Users/cheny/Downloads/10-monkey-species/validation/n%d',i-1); %loaction of the folder
        S = dir(fullfile(D,'*.jpg')); %load all the items end with .jpg
        for k = 1:numel(S) %for every image
            F = fullfile(D,S(k).name); %name of the iamge
            I = imread(F); %read the image
            testing_images(:,:,:,end+1) = imresize(I,[227 227]); %resize the images
            testing_labels(:,end+1) = i; % store the labels
        end
    end
    testing_images = testing_images(:,:,:,(2:end)); %the fist one is a zero matrix
    testing_labels = categorical(testing_labels); %make it a category
end