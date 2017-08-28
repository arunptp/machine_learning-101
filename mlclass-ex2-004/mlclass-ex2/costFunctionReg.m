function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

z=X*theta;
g=sigmoid(z);

temp1=y'*log(g);
temp2=(1-y')*log(1-g);
J1=sum(-temp1-temp2)/m;
theta2=theta(2:end).^2;
J2=(sum(theta2)*lambda)/(2*m);

J=J1+J2;


temp3=g-y;
temp4=X'*temp3;
grad1=temp4./m;

grad2=[theta(2:end).*(lambda/m)]+grad1(2:end);

grad=[grad1(1,1);grad2];








% =============================================================

end
