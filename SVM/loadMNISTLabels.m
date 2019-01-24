function labels = loadMNISTLabels(filename)
%
% This function read the MNIST label data set and reshape it into desired
% format
%
% labels = loadMNISTLabels(filename)
%
% labels - a row vector and every element is the label of corresponding
%          image
% filename - MNIST file name.
%
% Yutao Chen
% 12/5/2018
%
    fid = fopen(filename,'r'); %open the file as fid
    
    magic = fread(fid,1,'int32',0,'b'); %read the magic number
    assert(magic == 2049, ('Wrong label set!')); %check if the data is the right one
    
    N_items = fread(fid,1,'int32',0,'b'); %number of images(labels)
    
    labels = fread(fid,inf,'unsigned char'); %read the labels
    labels = reshape(labels,1,N_items); %reshape the data into a row vector
    
    fclose(fid); %close the file
end