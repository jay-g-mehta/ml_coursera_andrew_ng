function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


% ( y_exp(i) - y(i)) ^ 2
% ( ( X(i,:) * theta) - y(i) ) ^ 2
%  Dimensions:
%  X= (m X j ), theta = (j X 1), y = m X 1

J = (sum( ((X * theta) - y ) .^ 2 ) ) / (2*m);

% =========================================================================

end
