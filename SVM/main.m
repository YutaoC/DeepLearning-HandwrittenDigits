%% load the training and testing data and correspongding labels
images = loadMNISTImages('train-data'); %training images
labels = loadMNISTLabels('train-labels'); %labels of training images
test_images = loadMNISTImages('test-data'); %testing images
test_labels = loadMNISTLabels('test-labels'); %labels of testing images

%% LDA on the train data
Mdl = fitcdiscr(images',labels','DiscrimType','pseudolinear'); %LDA
sigma_b = Mdl.BetweenSigma; %between-class matrix
sigma_w = Mdl.Sigma + 1*eye(size(images,1)); %within-class matrix
[eigvec,~] = eigs(sigma_b,sigma_w,9); %the first 9((# of classes) -1) eigenvectors

Train_Data = zeros(9,size(images,2));
Test_Data = zeros(9,size(test_images,2));
for i = 1:size(images,2)
    Train_Data(:,i) = eigvec.' * images(:,i); %project the training images
end
for i = 1:size(test_images,2)
    Test_Data(:,i) = eigvec.' * test_images(:,i); %project the testing images
end

%% training SVM using "one vs all" algorithm
Mdl_svm = cell(1,10); % 10 different SVMs
for i = 1:10 %for every class
   l_train = labels; 
   for l = 1:60000 %for every images in training and testing data
       if l_train(l) ~= i-1 %if the image is not in the current class
           l_train(l) = -1; %set the label to -1
       else
           l_train(l) = 1; %else set the label to +1
       end
   end
   %train the SVM
%    Mdl_svm{i} = fitcsvm(Train_Data',l_train','KernelFunction','linear'); %linear SVM
%    Mdl_svm{i} = fitcsvm(Train_Data',l_train','KernelFunction','polynomial','PolynomialOrder',0.05); %kernel SVM(polynomial)
   Mdl_svm{i} = fitcsvm(Train_Data',l_train','KernelFunction','rbf'); %kernel SVM(RBF)
end

%% training SVM using "one vs one" algorithm
Mdl_svm = cell(10,10); % allocate space for SVM models
for i = 1:9 
    for j = i+1:10 % total of 45 SVM models
        count = 0; %count the number of model we are training
        l_train = [];
        im_train = [];
        for l = 1:60000 %for every images in training data
            if labels(l)+1 == i %if the image is in in class i
                count = count + 1;
                l_train(count) = 1; %set the label to 1
                im_train(:,count) = Train_Data(:,l); %extract the image the training data
            elseif labels(l)+1 == j %if the image is in class j
                count = count + 1;
                l_train(count) = -1; %else set the label to -1
                im_train(:,count) = Train_Data(:,l);%extract the corresponding image
            else
                continue
            end
        end
   %train the SVM
%         Mdl_svm{i,j} = fitcsvm(im_train',l_train','KernelFunction','linear'); %linear SVM
%         Mdl_svm{i,j} = fitcsvm(im_train',l_train','KernelFunction','polynomial','PolynomialOrder',0.05); %kernel SVM(polynomial)
        Mdl_svm{i,j} = fitcsvm(im_train',l_train','KernelFunction','rbf'); %kernel SVM(RBF)
    end
end

%% test the trained SVM (one vs all algorithm)
scores = zeros(10000,10);
for i = 1:10 %for every SVMs
    [~,score] = predict(Mdl_svm{i},Test_Data'); %the score for each SVM
    scores(:,i) = score(:,2); %keep the posterior probability that observation is in class 1
end

[~,predict_labels] = max(scores,[],2); %choose the one with the highest score as the predicted label
predict_labels = predict_labels - 1; %set the labels from 0~9

%% test the trained SVM (one vs one algorithm)
vote = zeros(10000,45); %allocate the space for vote
cnt = 0; %count the number of models we are testing
for i = 1:9
    for j = i+1:10 %for every SVMs
        pre_labels = predict(Mdl_svm{i,j},Test_Data'); %the labels for each SVM
        for k = 1:10000 %convert the labels form 1 and -1 to the original labels
           if pre_labels(k) == 1
               pre_labels(k) = i;
           else
               pre_labels(k) = j;
           end
        end
        cnt = cnt + 1;
        vote(:,cnt) = pre_labels; %save the result for every SVMs
    end
end
predict_labels = mode(vote,2)-1; %find the one with majority vote and set the labels from 0~9

%% accuracy
correct = 0;
for i = 1:10000
    if predict_labels(i) == test_labels(i) %if the predicted label and true label are the same
        correct = correct + 1;
    end
end
accuracy = correct / 10000; %calculate accuracy(0~1)
display(accuracy)
