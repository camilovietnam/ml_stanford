function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Add the bias unit to the X
m = size(X, 1);
X = [ones(m, 1), X];

% We find the input of the hidden layer
Z2 = X * Theta1';

% Then we apply sigmoid to calculate the activation of the hidden layer
A2 = sigmoid(Z2);

% Now we add the bias unit before moving into the output layer
A2 = [ones(m, 1), A2];

% Now we calculate the inputs to the hidden layer
Z3 = A2 * Theta2';

% And the activation of layer 3 is the sigmoid of Z3
A3 = sigmoid(Z3);

% Let's convert the categories into the format with 1 in category
identi = eye(num_labels);
y_adj = (identi(y, :));

% Now we can calculate the cost 
J = sum(sum( (-y_adj .* log(A3) ) - ( (1-y_adj) .* log(1 - A3)) )) / m;


% And we also need to add the regularization component
Theta1_reg = Theta1(:, 2:end);
Theta2_reg = Theta2(:, 2:end);
Reg = (lambda/(2*m))*( sum(sum(Theta1_reg .^ 2)) + sum(sum(Theta2_reg .^ 2)) );

% And finally, we add the component to the error
J = J + Reg;

% 
%	Forward Propagation
%
DELTA_1 = zeros(size(Theta1));
DELTA_2 = zeros(size(Theta2));


for i = 1:m
	% Let's start with forward propagation
	x_actual = X(i, :);
	y_actual = y_adj(i, :);
	
	% First let's calculate the z2 from X and Theta1 
	Z2_actual = x_actual * Theta1';
	
	% Now the sigmoid will give us the activation for Layer 2
	A2_actual = sigmoid(Z2_actual);
	
	% Now let's add the bias unit to A2
	A2_actual = [1, A2_actual];
	
	% for layer 3, let's calculate z3
	Z3_actual = A2_actual * Theta2';
	
	% and we calculate the sigmoid of Z3
	A3_actual = sigmoid(Z3_actual);
	
	% Now let's calculate delta3, the error on the last layer
	Delta_3 = A3_actual - y_actual;
	
	% Now the error of the hidden layer
	Delta_2 = Theta2' * Delta_3';
	
	% Now let's calculate the error for the hidden layer
	% remember to delete the first row of Delta_2, the weights 
	% related to the bias unit
	Delta_2 = Delta_2(2:end, :)' .* sigmoidGradient(Z2_actual);
	
	% Now the error accumulators, the big Deltas
	DELTA_1 = DELTA_1 + (Delta_2' * x_actual);
	DELTA_2 = DELTA_2 + (Delta_3' * A2_actual);
end

Theta1_size = size(Theta1, 1);
Theta2_size = size(Theta2, 1);

Theta1_grad = (1/m)*DELTA_1 + (lambda/m)*[zeros(Theta1_size, 1), Theta1(:, 2:end)];
Theta2_grad = (1/m)*DELTA_2 + (lambda/m)*[zeros(Theta2_size, 1), Theta2(:, 2:end)];

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
