function [ Wstar, bstar] = MiniBatchGDEnsemble( ...
    X_train, Y_train, X_val, Y_val, GDparams, W, b, lambda, std_noise )
% MINIBATCHGD  Implementation of the mini-batch gradient descent algorithm
%
% [ Wstar, bstar] = MiniBatchGDEnsemble( ...
%    X_train, Y_train, X_val, Y_val, GDparams, W, b, lambda, std_noise )
%    performs the minibatch for a set of models. 
%   
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
%   W: Cell array containing weight matrices of different Networks, each
%       of them has size (Kxd)
%   b: Cell array containing the bias vectors, each has size (Kx1)
%   lambda: Weight on the regularization term
%   std_noise: Standard deviation of the noise added to the training images
%
% Outputs:
%   Wstar: Optimal solution found for W, size (Kxd)
%   bstar: Optimal solution found for b, size (Kx1)
%   loss_train: Loss obtained on the training set
%   loss_val: Loss obtained on the validation set

Wstar = cell(1, numel(W));
bstar = cell(1, numel(W));

for i=1:numel(W) 
    [ Wstar{i}, bstar{i}, ~ , ~] = MiniBatchGD( X_train,...
        Y_train, X_val, Y_val, GDparams{i}, W{i}, b{i}, lambda{i}, std_noise{i} );
end

end
