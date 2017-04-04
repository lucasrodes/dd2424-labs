function [ grad_W, grad_b ] = ComputeGradients( X, Y, P, W, lambda )
%COMPUTEGRADIENTS Computes the gradients of the model parameters (W, b)
%
% Inputs:
%   X: Each column of X corresponds to an image, it has size (dxn)
%   Y: One-hot ground truth label for the corresponding image vector in X,
%       it has size (Kxn)
%   P: contains the probability for each label for the image 
%       in the corresponding column of X. It has size (Kxn)
%   W: Weight matrix, it has size (Kxd)
%   lambda: Weight on the regularization term
%
% Outputs:
%   grad_W: Gradient of the Weight matrix, size (Kxd)
%   grad_b: Gradient of the bias vector, size (Kx1)

% Size of the batch
B = size(X,2);

% Initialize gradients
grad_W = zeros(size(W));
grad_b = zeros(size(W,1),1);

% Iterate for each input sample
for i=1:size(X, 2)
    x = X(:, i);
    y = Y(:,i);
    p = P(:, i);
    
    g = - y'/(y'*p) * (diag(p) - p*p');
    grad_b = grad_b + g';
    grad_W = grad_W + g'*x';
end

% Normalize and add regularization term
grad_W = grad_W/B + 2*lambda*W;
grad_b = grad_b/B;

end

