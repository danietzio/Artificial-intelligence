function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30
values = [ 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
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

error = ones(8 * 8,1);
predections = zeros( size(Xval,1) , 1); 
_C = 0;
_sigma = 0;

for i = 1:8,
  _C = values(i);
  for j = 1: 8,
      _sigma = values(j);
 
      model = svmTrain(X, y, _C, @(x1, x2) gaussianKernel(x1, x2, _sigma)); 
      predections = svmPredict(model, Xval);   
      error( 8 * ( i - 1 ) + j, 1 )  = mean(double(predections ~= yval));
      
  end;
end;

indexOfMin = find(error == min(error(:)));
indexOfMin = indexOfMin(1,1);

iIndexValues = ceil( indexOfMin / 8 ) ;

jIndexValues = indexOfMin - ( iIndexValues - 1 ) * 8;


C = values(iIndexValues);
sigma = values(jIndexValues);


% =========================================================================

end
