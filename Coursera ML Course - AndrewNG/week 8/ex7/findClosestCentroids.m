function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%


X1 = sum( ( ( X - centroids(1 , :) ) .^ 2 )' )' ;
X2 = sum( ( ( X - centroids(2 , :) ) .^ 2 )' )' ;
X3 = sum( ( ( X - centroids(3 , :) ) .^ 2 )' )' ;

X0 = [ X1 X2 X3 ];

[ i j ] = find( X0 == min(X0')');

g = ones(size(X,1), 1);
g(i) = j;

idx = g;

pause;
% =============================================================

end

