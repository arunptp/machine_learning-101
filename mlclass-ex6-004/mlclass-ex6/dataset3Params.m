function [c, sigma, error] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
c = 0.01;
sigma = 0.01;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
error=0;
i=1;j=1;uff=1;temp=1;
while c(i)<1
   sigmatemp = 0.01; uff=1;sigma(j)=sigmatemp(uff);
    while sigmatemp(uff)<1
        model= svmTrain(X, y, c(i), @(x1, x2) gaussianKernel(x1, x2, sigmatemp(uff)));
        predictions = svmPredict(model, Xval);
        error(temp)=mean(double(predictions ~= yval));
        temp=temp+1;
        
        uff=uff+1;
        sigmatemp(uff)=sigmatemp(uff-1)*3;
        j=j+1;
        sigma(j)=sigmatemp(uff);
    end
   i=i+1;
    c(i)=c(i-1)*3;
  
end


[value, pos]=min(error);

mf=(size(sigma,2)-1)/(size(c,2)-1);

sigma=sigma(pos);

index=uint8(pos/mf);
c=c(index);





% =========================================================================

end
