function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);


% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the
%                     partial derivatives w.r.t. to each element of Theta
%
% X(i) is a vector for i-th movie that represents values for features 1 to n
% Example: X(1) = [1 0.9] represents first movie features values 1 for action(1=full of action) and 0.9 for romance (almost no romance)
% Theta(j) is a vector for j-th user that indicates like-ness of user j for movie features action and romance
% X= [movie1_feature1 movie1_feature2 ... ; movie2feature1 movie2feature2 ...]
% Theta = [user1feature1 user1feature2... ; user2feature1 user2feature2 ....]
% Y =[movie1user1rating movie1user2rating ...; movie2user1rating movie2user2rating ...]

J = (1./2) * sum(sum( (((X*Theta') - Y).^2) .* R) );

% Gradient calculation:
% X_grad = [ movie1_derivative_feature1 movie1_derivative_feature2, ... ; movie2_derivative_feature1 movie2_derivative_feature2, ... ;]
% X_grad(i,:) = [movie1_derivative_feature1 movie1_derivative_feature2, ... ]
% X_grad(i, :) = ((X(i, :) * Theta') - Y) .* R * Theta
X_grad = ( ((X*Theta') - Y) .* R) * Theta;


% Theta_grad = [user1_derivative_feature1 user1_derivative_feature2, ...; user2_derivative_feature1 user2_derivative_feature2, ...; ]
% Theta_grad(J, :) = ( ((X*Theta(j, :)') - Y) .* R)' * X;
Theta_grad = ( ((X*Theta') - Y) .* R)' * X;

% With regularization:

J = J + (lambda/2)* sum(sum(X.^2)) + (lambda/2)* sum(sum(Theta.^2)) ;


X_grad = X_grad + lambda.*X;
Theta_grad = Theta_grad + lambda.*Theta;



% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
