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


% Cost Function 
% Cost Function 
% Cost Function 


X = [ ones(size(X,1),1) X];
h1 = sigmoid( X * Theta1' );

h1 = [ ones(size(h1,1),1) h1];
h2 = sigmoid( h1 * Theta2' );  

new_y = ([1:size(Theta2,1)] ==  y)';

log_h2 = log(h2);
log_1_h2 = log(1 .- h2);

theta1_two = Theta1(:,2:size(Theta1,2)) .^ 2;
theta2_two = Theta2(:,2:size(Theta2,2)) .^ 2;

reg_part = ( lambda / ( 2 * m ) ) * ( sum(sum(theta1_two')') + sum(sum(theta2_two')') );


J = (1 / m ) * sum(sum( ( (-1 .* new_y) .* log_h2' ) .- (( 1 .- new_y) .* log_1_h2' ) )) + reg_part;



% BackPropagation
% BackPropagation
% BackPropagation

delta_1 = zeros(size(Theta1));
delta_2 = zeros(size(Theta2));

for i=1 : m
   %get i_th training set
   data_set = X(i,:);
   
   %compute activation for hidden layer
   z_2 = (data_set * Theta1')';
   hidden_a = sigmoid( z_2 );
   
   %add bias 
   hidden_a = [ 1 ; hidden_a];
   
   %compute activation of output layer
   z_3 = ( hidden_a' * Theta2')' ;
   output_a = sigmoid( z_3 );
  
   %compute error for output layer
   output_err = output_a .- ( new_y(:,i) );

   %compute error for hidden layer
   hidden_err = ( ( Theta2 )' * output_err ) .* sigmoidGradient( [ 1 ; z_2 ]);
   hidden_err = hidden_err( 2:end );
   
   %compute delta
   delta_1 = delta_1 + hidden_err * data_set;
   delta_2 = delta_2 + output_err * hidden_a';
   
end;

Theta1_grad(1,:) = ( 1 / m ) .* delta_1(1,:);
Theta1_grad(2 : hidden_layer_size , :) = ( 1 / m ) .* delta_1(2 : hidden_layer_size, : ) + ( lambda / m ) .* Theta1(2 : hidden_layer_size, : );

Theta2_grad(1,:) = ( 1 / m ) .* delta_2(1,:);
Theta2_grad(2 : size(Theta2,1) , :) = ( 1 / m ) .* delta_2(2 : size(Theta2,1), : ) + ( lambda / m ) .* Theta2(2 : size(Theta2,1), : ) ;


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
