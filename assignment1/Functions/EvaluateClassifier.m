function [ P ] = EvaluateClassifier( X, W, b )
% EVALUATECLASSIFIER   evaluates the scores of a batch of images
%   P = EvaluateClassifier( X, W, b ) performs the forward pass computing
%   the scores of each class for all data samples in X using the model
%   parameters W and b.
%
% Inputs:
%   X: Each column of X corresponds to an image, it has size (dxn)
%   W: Weight matrix, it has size (Kxd)
%   b: bias vector, it has size (Kx1)
%
% Outputs:
%   P: contains the probability for each label for the image 
%       in the corresponding column of X. It has size (Kxn)

Y = bsxfun(@plus,W*X,b);

P = bsxfun(@rdivide, exp(Y), sum(exp(Y), 1));

end

