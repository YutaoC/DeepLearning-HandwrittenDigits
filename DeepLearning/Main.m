%% prepare the data and the network
addpath(genpath('../CoreModules')); %add the path contains the CoreModules
n_epoch = 30; %number of epoch
dataset_name = 'MNIST'; %the name of the dataset
network_name = 'LeNet'; %the name of the network
use_gpu = (gpuDeviceCount > 0); %check if GPU is available

if use_gpu % if GPU is available
   opts.use_nntoolbox = license('test','neural_network_toolbox'); %use the toolbox 
end

PrepareDataFunc = @PrepareData_MNIST; %(function handle) prepare the data 
NetInit = @NetInit_MNIST; %(function handle) initial the network
use_selective_sgd = 1; %1-select the best learning rate for sgd using built-in function
                       %2-use the learning rate specified by variable "sgd_lr"
ssgd_search_freq = 10; 
learning_method = @sgd; %the learning method
% sgd_lr = 1e-3; %learning rate if use_selective_sgd = 0

%% train and test the network
Main_Template();