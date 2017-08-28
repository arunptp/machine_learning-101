function [final] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history=ones(num_iters,1);


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    temp=((theta'*X')'-y)'*X;
    temp1=(alpha/m).*temp;
    theta=theta-(temp1)';
    





J_history(iter)= computeCost(X, y, theta);

    % ============================================================

    % Save the cost J in every iteration    
    

end

final=[theta;J_history];

end
