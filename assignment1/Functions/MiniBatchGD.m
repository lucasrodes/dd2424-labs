function [ Wstar, bstar, loss_train , loss_val] = MiniBatchGD( X_train, ... 
    Y_train, X_val, Y_val, GDparams, W, b, lambda, std_noise )
% MINIBATCHGD  Implementation of the mini-batch gradient descent algorithm
%
% [ Wstar, bstar, loss_train , loss_val] = MiniBatchGD(X_train, ... 
%    Y_train, X_val, Y_val, GDparams, W, b, lambda, std_noise) performs the
%    mini-batch SGD updating the model parameters W and b based on the cost
%    computed on the training set X_train.
%
% Inputs:
%   X_train: Each column of X corresponds to an image, it has size (dxN).
%               Samples belong to train set.
%   Y_train: One-hot ground truth label for the corresponding image vector 
%           in X, it has size (KxN). Samples belong to train set.
%   X_val: Each column of X corresponds to an image, it has size (dxN)
%               Samples belong to validation set.
%   Y_val: One-hot ground truth label for the corresponding image vector 
%           in X, it has size (KxN). Samples belong to validation set.
%   GDparams: Parameters of the training
%   W: Weight matrix, it has size (Kxd)
%   b: Bias vector, it has size (Kx1)
%   lambda: Weight on the regularization term
%   std_noise: Standard deviation of the noise added to the training images
%
% Outputs:
%   Wstar: Optimal solution found for W, size (Kxd)
%   bstar: Optimal solution found for b, size (Kx1)
%   loss_train: Loss obtained on the training set
%   loss_val: Loss obtained on the validation set


% Obtain training parameters
[~, N] = size(X_train);
n_batch = GDparams.n_batch;
eta = GDparams.eta;
n_epochs = GDparams.n_epochs;

% Initialize loss on the training set
loss_train = zeros(n_epochs+1,1);
loss_train(1) = ComputeCost( X_train, Y_train, W, b, lambda );
fprintf('Cost = %d\n', loss_train(1));

% Initialize loss on the validation set
loss_val = zeros(n_epochs+1,1);
loss_val(1) = ComputeCost( X_val, Y_val, W, b, lambda );
    
for epoch=1:n_epochs
    rand_perm = randperm(size(X_train,2));
    X_train = X_train(:, rand_perm);
    Y_train = Y_train(:, rand_perm);
    for j=1:N/n_batch
        % Obtain batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = X_train(:, inds);
        Ybatch = Y_train(:, inds);

        % Add some jitter to pictures
        Xbatch = Xbatch + std_noise*randn(size(Xbatch));
        
        % Forward pass
        P = EvaluateClassifier( Xbatch, W, b );

        % Backward pass
        [dW, db] = ComputeGradients( Xbatch, Ybatch, P, W, lambda );

        % Update network parameters
        W = W - eta*dW;
        b = b - eta*db;
    end
    % Obtain loss for training and validation sets
    loss_train(epoch+1)=ComputeCost( X_train, Y_train, W, b, lambda );
    fprintf('%.d) Cost = %d\n', epoch, loss_train(epoch+1));
    loss_val(epoch+1) = ComputeCost( X_val, Y_val, W, b, lambda );
    
    % Decrease learning rate
    %eta = 0.99*eta;
    % eta = eta*exp(-0.001*epoch);
end

% Output results
Wstar = W;
bstar = b;

end
