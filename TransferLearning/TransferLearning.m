%% load the pretrained network
net = alexnet;

%% load the images
[training_images,training_labels] = loadTrainingImages(); %read training data
[testing_images,testing_labels] = loadTestingImages(); %read testing data

%% transfer and form a new network
N_class = 10;

%replace the layers
layers = net.Layers;
layers(23) = fullyConnectedLayer(N_class,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20);
layers(25) = classificationLayer;

%freeze the weight 
for i = 1:numel(layers)-3 %for all the previous layers
    if isprop(layers(i),'WeightLearnRateFactor')
        layers(i).WeightLearnRateFactor = 0;
    end
    if isprop(layers(i),'WeightL2Factor')
        layers(i).WeightL2Factor = 0;
    end
    if isprop(layers(i),'BiasLearnRateFactor')
        layers(i).BiasLearnRateFactor = 0;
    end
    if isprop(layers(i),'BiasL2Factor')
        layers(i).BiasL2Factor = 0;
    end
end

%training options
opts = trainingOptions('sgdm',... %training method
    'MiniBatchSize',64,... %batch size
    'MaxEpoch',2,... %number of epoch
    'InitialLearnRate',0.0001,... %learning rate
    'Plots','training-progress');

%% fine tune
%initial the network
layers_ft = net.Layers;
layers_ft(23) = trainnet.Layers(23);
layers_ft(25) = trainnet.Layers(25);

%training options
opts_ft = trainingOptions('sgdm',... %training method
    'MiniBatchSize',64,... %batch size
    'MaxEpoch',2,... %number of epoch
    'InitialLearnRate',0.0001,... %learning rate
    'Plots','training-progress');

%% train the network
% trainnet = trainNetwork(training_images,training_labels,layers,opts); %train the network
trainnet_ft = trainNetwork(training_images,training_labels,layers_ft,opts_ft); %train the network(fine-tune)

%% test the netwoek
% [predicted_labels,score] = classify(trainnet,testing_images); %test the neteork on the testing images
[predicted_labels,score] = classify(trainnet_ft,testing_images); %test the neteork on the testing images(fine-tune)
accuracy = mean(predicted_labels == testing_labels'); %calculate the accuracy
display(accuracy)
