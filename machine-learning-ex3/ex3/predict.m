function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% In the first layer, activation a1 is basically the same as X, but with 
% an extra column for the bias unit (a column of 1s)
a1 = [ones(m, 1) X];

% for the second layer, we first multiply our inputs (a1) by our weight (theta_1)
% and this would be our 2nd layer inputs. These we need to pass to the sigmoid
% function, and it will return our activation a2.
z2 = a1 * Theta1';
m2 = size(z2, 1);
% remember that you don't need to sigmoid the column of ones! I made that mistake
% before
a2 = [ones(m2, 1) sigmoid(z2)];


% in the third layer, we repeat the process. The input z3 will be the output of
% layer 2 (which is a2) multiplied by the weights Theta2. This result we will
% pass to sigmoid
z3 = a2 * Theta2';
% Layer 3 is the output layer so we don't have any bias units here. We can
% pass the whole vector to the sigmoid function
a3 = sigmoid(z3);

% here we have a matrix where every row represents the predictions for one example,
% and each column will be the prediction for one category. Of these categories,
% the one with the highest value will be the best prediction, so we return the max
[max_p, p] = max(a3, [], 2);

% =========================================================================


end
