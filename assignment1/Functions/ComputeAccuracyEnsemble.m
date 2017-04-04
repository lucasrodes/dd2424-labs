function [ acc ] = ComputeAccuracyEnsemble( X_test, y_test, X_train, Y_train, W, b, lambda )
% COMPUTEACCURACYENSEMBLE  Computes the accuracy of the model on some 
%   dataset.
%   acc = COMPUTEACCURACY(X, y, W, b) computes accuracy on the set X with 
%   labels yof the model described by a set of W and b.
%
% Inputs:
%   X: Each column of X corresponds to an image, it has size (dxn)
%   y: Ground truth labels for the corresponding image vectors in X,
%       it has size (nx1)
%   W: Cell array containing weight matrices of different Networks, each
%       of them has size (Kxd)
%   b: Cell array containing the bias vectors, each has size (Kx1)
%
% Outputs:
%   acc: Accuracy obtained with the current model

% Obtain number of samples
n = size(X_test,2);

IDX = zeros(numel(W), n);
votes = zeros(size(W{1},1), n);
summ = 0;
J = zeros(1, numel(W));

% Obtain costs for each model and store they weight on the voting
for i=1:numel(W)
    J(i) = exp(1+ComputeCost( X_train, Y_train, W{i}, b{i}, lambda{i} ));
    summ = summ + J(i);
end
J = (summ - J)/summ;

for i=1:numel(W)
    % Obtain scores
    P = EvaluateClassifier( X_test, W{i}, b{i} );
    % Obtain classification for each point
    [~, IDX(i,:)] = max(P);
    %J(i)=1;
    votes = votes + J(i)*full(ind2vec(IDX(i,:)));    
end


[~, idx] = max(votes);
%idx = mode(IDX, 1);
% Obtain accuracy
acc = sum(y_test == idx')/n;

end

