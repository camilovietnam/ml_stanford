function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

% there should be 1 theta for each xi, plus the theta_0
% remember that n is the number of features , so the number of Xi values for each example
% each Xi goes with one theta, but also we have the theta_0, so number of thetas is +1 the number of Xi
theta_zeros = zeros(n+1, 1);

% set up options for the function call
% this one I copied from week 2's pdf
options = optimset('GradObj', 'on', 'MaxIter', 400);

% We loop through each label, getting the thetas for each
% one of the classification labels (1,2,3...10)
% then we put that vector of thetas as a row of the all_theta
for i = 1:num_labels
[theta,cost] = fminunc(@(t)(lrCostFunction(t, X, (y==i), lambda)), theta_zeros, options);
all_theta(i, :) = theta';
endfor

% the result is a matrix where each row represents the prediction for one example, 
% and each column will represent the prediction for one category. 

% =========================================================================


end
