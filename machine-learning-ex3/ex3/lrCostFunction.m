function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


%%%%%%% OUR SOLUTION %%%%%%% 

% first we create the regularized theta, with the first position set to zero:
% [0, theta_1, theta_2, theta_3, ... ]
% because when we add this to the grad, we want to add nothing for the first position
% that is, we don't regularize the value of theta_0 (because it is not multiplied by
% any feature, so should not be modified. Remember:
% h = theta_0 + theta_1 * x1 + theta_2*x2 + .... 
% theta_0 not affecting any feature Xi, we don't do anything to it
reg_theta = [0; theta(2:end)];

% we create our hypothesis vector by computing the sigmoid of the matrix product X*theta
h = sigmoid(X * theta);

% our regularized J has two parts, here I separate them because it is easier for me
% to understand short lines of code

% part 1 is the same cost function J we know from before
% J = sum (1/m * (-y * log(h)) - (1-y)*log(1-h) ) 
J_1 = (1/m) * sum(-y .* log(h) - (1-y).* log (1 - h));

% part 2 is the extra we add to regularize
J_2 = (lambda/(2*m)) * sum(reg_theta.^2);

% finally we combine the two parts into one J 
J = J_1 + J_2;  % this value should be a scalar

% gradient is similar to J, to regularize we need to add the two parts
% part 1 is the regular gradient vector
grad_1 = (1/m) * (X' * (h-y));

% part 2 
grad_2 = (lambda/m) * reg_theta;

% finally we combine into one vector
grad = grad_1 + grad_2; % this value should be a vector

% =============================================================

grad = grad(:);

end
