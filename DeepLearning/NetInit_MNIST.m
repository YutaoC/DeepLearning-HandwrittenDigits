function net = NetInit_MNIST(opts)
%
% This function initialize the layers of the network
%
% net = NetInit_MNIST(opts)
%
% opts - the structure contains the training and testing data obtained in
%        the function "PrepareData_MNIST"
% net - the designed CNN
%
% Yutao Chen
% 12/10/2018
%
    %control the random number generator
    rng('default'); 
    rng(0);
    
    f = 1/100; %scaler
    
    net.layers = {}; %initial the net
    
    net.layers{end+1} = struct('type','conv',... %conv layer
                               'weights',{{f*randn(5,5,1,6,'single'),zeros(1,6,'single')}},... %size of the filter and the dimension of input and output
                               'stride',1,... %stride size
                               'pad',2); %zero pad size
    net.layers{end+1} = struct('type','relu'); %activation function (ReLu)
    net.layers{end+1} = struct('type','pool','pool',2,'stride', 2); %pooling
    
    net.layers{end+1} = struct('type','conv',... %conv layer
                               'weights',{{f*randn(5,5,6,16,'single'),zeros(1,16,'single')}},... %size of the filter and the dimension of input and output
                               'stride',1); %stride size
    net.layers{end+1} = struct('type','relu'); %activation function (ReLu)
    net.layers{end+1} = struct('type','pool','pool',2,'stride', 2);%pooling
    
    net.layers{end+1} = struct('type','conv',... %conv layer
                               'weights',{{f*randn(5,5,16,120,'single'),zeros(1,120,'single')}},... %size of the filter and the dimension of input and output
                               'stride',1); %stride size
    net.layers{end+1} = struct('type','relu'); %activation function (ReLu)
    
    net.layers{end+1} = struct('type','conv',... % fully connected layer
                               'weights',{{f*randn(1,1,120,10,'single'),zeros(1,10,'single')}},... %the dimension of input and output
                               'stride',1); %stride size
    
    net.layers{end+1} = struct('type','softmaxloss'); %softmax
end