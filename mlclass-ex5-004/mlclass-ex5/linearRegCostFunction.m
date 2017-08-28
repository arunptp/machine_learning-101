function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

temp=((theta'*X')'-y);
temp1=temp.^2;
J1=(1/(2*m))*sum(temp1);

temp2=theta(2:end).^2;
J2=(sum(temp2)*lambda)/(2*m);

J=J1+J2;

grad1=(X'*temp)./m;
grad2=[theta(2:end).*(lambda/m)]+grad1(2:end);
grad=[grad1(1,1);grad2];










% =========================================================================

grad = grad(:);

end
