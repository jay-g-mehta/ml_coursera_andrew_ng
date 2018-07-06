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

%yvectorlabel=[1,2,3,4,5,6,7,8,9,0]
%recode_y=[] % 5000 X 10
%for i=1:m
%  recode_y= [recode_y; (yvectorlabel==y(i))']
%endfor

yd = eye(num_labels);
recode_y = yd(y,:);

X = [ones(m, 1) X]; % Add bias unit

%fprintf('Size of X: %d %d \n', size(X,1), size(X, 2)); %5000 X 401
%fprintf('Size of Theta1: %d %d \n', size(Theta1,1), size(Theta1, 2)); % 25 X 401
%fprintf('Size of Theta2: %d %d \n', size(Theta2,1), size(Theta2, 2)); % 10 X 26
%fprintf('Size of y: %d %d \n', size(y,1), size(y, 2)); % 5000  X 1

z2=X*Theta1'; % 5000 X 401  *  401 X 25
A2 = sigmoid(z2); % 5000 X 401  *  401 X 25
A2 = [ones(m, 1) A2]; % Add bias unit
A3 = sigmoid(A2 * Theta2'); % 5000 X 26 * 26 X 10

J = sum(-(sum(recode_y .* log(A3), 2)) - (sum((1-recode_y) .* log(1-A3), 2)));
J = (1./m)* J;

% Given, only 3 layers
Theta1_cols = size(Theta1, 2);
Theta2_cols = size(Theta2, 2)
reg_parameter = (lambda/(2*m)) *( sum(sum(Theta1(:, 2:Theta1_cols).^2 )) + sum(sum(Theta2(:, 2:Theta2_cols).^2 )));
J = J + reg_parameter;

% -------------------------------------------------------------


delta3 = A3 - recode_y; % 5000 X 10

z2=[ones(m, 1) z2]; % 5000 X 26
delta2 = (delta3 * Theta2) .* (sigmoidGradient(z2)); %  (5000 X 10) *(10 X 26) .* ( 5000 X 26)
delta2_cols = size(delta2, 2);

A1 = X;
Tdelta1 = 0;
Tdelta2 = 0;

Tdelta1 = Tdelta1 + (delta2(:, 2:delta2_cols))' * A1; % (5000X(26-1))' * 5000 X 401
Tdelta2 = Tdelta2 + (delta3)' * A2; % (5000 X 10)' * 5000 X 25

Ddelta1 = (1.0/m) * Tdelta1; %
Ddelta2 = (1.0/m) * Tdelta2;

Theta1_grad = Ddelta1;
Theta2_grad = Ddelta2;
% =========================================================================
theta1_rows = size(Theta1, 1);
Theta1_reg = (lambda/m)*[zeros(theta1_rows,1) Theta1(:, 2:Theta1_cols)];

theta2_rows = size(Theta2, 1);
Theta2_reg = (lambda/m)*[zeros(theta2_rows,1) Theta2(:, 2:Theta2_cols)];

Reg_theta1_grad = Theta1_grad + Theta1_reg;
Reg_theta2_grad = Theta2_grad + Theta2_reg;

% Unroll gradients
grad = [Reg_theta1_grad(:) ; Reg_theta2_grad(:)];

end
