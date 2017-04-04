%% Assignment 1 
% Author: Lucas Rod?s Guirao
%
% This file contains the code implemented for the assignment 1. In this
% regard, there are two parts. 
% 
% * PART 1 deals with basic and elementary
% aspects of the code (such as implementation of the required functions).
% * PART 2 then focuses on using all these implemented functions to 
% construct the Mini-Batch SGD algorithm and evaluates its performance on 
% the test data.
%
% Please run each of the sections individually to observe the building
% process I have undertaken.
%

clear
clc
addpath Datasets/cifar-10-batches-mat/;
addpath Functions;

%% PART I
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% In this part we implement the Forward, the Backward pass and the loss
% function. In this regard, we first need to load the datasets for this
% assignment. Both PART 1 and PART 2 are independent, can be run separately
%

%% Visualize some images from the dataset
A = load('data_batch_1.mat');
I = reshape(A.data', 32, 32, 3, 10000);
I = permute(I, [2, 1, 3, 4]);
montage(I(:, :, :, 1:500), 'Size', [5,5]);

%% 1. Load the training set

[ X_train, Y_train, y_train ] = LoadBatch( 'data_batch_1.mat' );

%% 2. Initialize weight matrix

% Obtain d: #features and K: #classes
[d, ~] = size(X_train);
[K, ~] = size(Y_train);

% Randomly initialize W
std_dev = 0.01;
W = std_dev*randn(K, d);
b = std_dev*randn(K, 1);

%% 3. Classify data
% We implement the forward-pass

P = EvaluateClassifier( X_train(:,1:100), W, b );

% Running the classifier with random weights does not give us much 
% information. In fact it works similarly to a random guess, since scores 
% for each class are pretty much the same. In other words, the variance 
% along the columns is very small

figure(1);
hist(var(P,1),20, 'k');
h = findobj(gca, 'Type','patch');
set(h, 'FaceColor','k', 'EdgeColor','w')
title('Histogram of the score vector variance for each evaluated data point'...
    , 'Interpreter', 'latex', 'fontsize', 16);
xlabel('Variance values', 'Interpreter', 'latex', 'fontsize', 14);
grid on;

%% 4. Compute cost
% We implement our cost/loss measure and test it on our train dataset.

lambda = 1;
J = ComputeCost( X_train, Y_train, W, b, lambda );

%% 5. Compute Accuracy
% We compute an accuracy evaluator

acc = ComputeAccuracy( X_train, y_train, W, b );

% We note that the accuracy is roughly 10%, which corresponds to random
% guess (10 classes) and is consistent with previous results.
sprintf('Accuracy: %.3f', acc)

%% 6. Gradient check
% We implement, analytically, the gradient computation. To verify that it
% is well implemented we run this "gradient check" test, which compares our
% results with the results obtained using numerical methods.

X_train = X_train(1:100,:);
W = W(:,1:100);

% Runfor each batch
for i=1:100
    disp(i)
    P = EvaluateClassifier( X_train(:,1+100*(i-1):100*i), W, b );
    [ grad_W, grad_b ] = ComputeGradients( X_train(:,1+100*(i-1):100*i),...
        Y_train(:,1+100*(i-1):100*i), P, W, lambda );
    [ngrad_b, ngrad_W] = ComputeGradsNumSlow(X_train(:,1+100*(i-1)...
        :100*i), Y_train(:, 1+100*(i-1):100*i), W, b, lambda, 1e-6);

    err_W = norm(reshape(grad_W,numel(grad_W),1)-reshape(ngrad_W,numel...
        (ngrad_W),1))/...
        (norm(reshape(grad_W,numel(grad_W),1))+norm(reshape(ngrad_W, ...
        numel(ngrad_W),1)));
    err_b = norm(grad_b-ngrad_b)/(norm(grad_b)+norm(ngrad_b));
    
    % Display warning if the difference is above a threshold
    if (err_W>1e-6)
        disp('Weight gradient error!');
    end
    if (err_b>1e-6)
        disp('Bias gradient error!');
    end
end


%% PART 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% The subsequent code executes the Mini-Batch SGD algorithm with some
% optimization techniques.
% Maximum Accuracy obtained was %.
%

%% Prepare training set

% Load data sets
[ X_train, Y_train, y_train ] = LoadBatch( 'data_batch_1.mat' );
[ X_val, Y_val, y_val ] = LoadBatch( 'data_batch_2.mat' );
[ X_test, Y_test, y_test ] = LoadBatch( 'test_batch.mat' );

% % We take 9000 of the samples from the validation set
X_train = [X_train, X_val(:, 1:9000)];
Y_train = [Y_train, Y_val(:, 1:9000)];
y_train = [y_train; y_val(1:9000)];

X_val = X_val(:, 9001:end);
Y_val = Y_val(:, 9001:end);
y_val = y_val(9001:end);

% Center data using the mean of the training set
mu = mean(X_train, 2);
X_train = bsxfun(@minus, X_train, mu);
X_val = bsxfun(@minus, X_val, mu);
X_test = bsxfun(@minus, X_test, mu);

%% Prepare learning parameters

% Initialize learning parameters
GDparams.n_batch = 100;%50;
GDparams.eta = 0.01;
GDparams.n_epochs = 40;
lambda = 0;

% Obtain d: #features and K: #classes
[d, ~] = size(X_train);
[K, ~] = size(Y_train);

% Randomly initialize weight matrix and bias vector
rng(400);
std_dev = 0.01;
W = std_dev*randn(K, d);
b = std_dev*randn(K, 1);

% Noise added in training data
std_noise = 1e-3;%1e-2;

%% Run Mini-batch SGD algorithm
[ Wstar, bstar, loss_train , loss_val] = MiniBatchGD( X_train, Y_train, ...
   X_val, Y_val, GDparams, W, b, lambda, std_noise );

%% Obtain accuracy on test data

acc = ComputeAccuracy( X_test, y_test, Wstar, bstar );
fprintf('Accuracy = %.2f %%\n', acc*100);

%% Run Ensembles approach

%% 1
% Obtain d: #features and K: #classes
[d, ~] = size(X_train);
[K, ~] = size(Y_train);

n_ensembles = 12;
W = cell(1,n_ensembles);
b = cell(1,n_ensembles);
std_dev = 0.01;

GDparams = cell(1,12);
n_batch = [100,100,100,100,100,50,50,50,50,50,75,75];
eta = [0.01, 0.001, 0.005, 0.02, 0.0075, 0.01, 0.001, 0.005, 0.02, 0.0075,0.01,0.03];
std_noise = {0,0,0,0.001,0.01,0,0,0,0.0001,0.01,0,0.2};

lambda = {0,0,0,0.01,0,0,0,0,0,0,0.1,.001};
for i=1:n_ensembles
    GDparams{i}.n_batch = n_batch(i);
    GDparams{i}.eta = eta(i);
    GDparams{i}.n_epochs = 40;
    W{i} = std_dev*randn(K, d);
    b{i} = std_dev*randn(K, 1);
end

%% 2
[ Wstar, bstar] = MiniBatchGDEnsemble( X_train, ...
    Y_train, X_val, Y_val, GDparams, W, b, lambda, std_noise );

save('ensemblePEPI2.mat','Wstar','bstar','GDparams', 'lambda');
%% 3
acc = ComputeAccuracyEnsemble( X_test, y_test, X_train, Y_train, Wstar, bstar, lambda );
fprintf('Accuracy = %.2f %%\n', acc*100);

%% Visualize Loss

figure;
plot(loss_train);
hold on;
plot(loss_val);
h_legend = legend('Training loss', 'Validation loss');
set(h_legend, 'Fontsize', 16, 'Interpreter','latex');
set(gca,'fontsize',14)
ylabel('Loss','Interpreter','latex', 'fontsize', 18);
xlabel('Epoch','Interpreter','latex', 'fontsize', 18);
grid on

%% Visualize "class templates"
for i=1:10
    im = reshape(Wstar(i,:), 32, 32, 3);
    s_im{i} = (im - min(im(:)))/(max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end

% Plot templates for each of the 10 classes
f = figure;
set(f, 'Position', [100, 100, 10000, 200]);
label = {'airplane','auto', 'bird', 'cat', 'deer', 'dog', 'frog', ...
    'horse', 'ship', 'truck'};
for i = 1:10
    
    im = s_im{i};
    subplot(1,10,i);
    imshow(im);
    title(label{i}, 'fontsize', 16, 'Interpreter', 'latex')
end