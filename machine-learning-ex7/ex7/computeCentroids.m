function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% COMPUTECENTROIDS(X, idx, K)
sums = zeros(length(X), size(X, 2));
total = zeros(length(X), 1);

% loop all the examples and add each one to a centroid
% according to the indexes in idx.
for i = 1:length(X)
	example = X(i, :);
	centid = idx(i, :);	
	sums(centid, :) = sums(centid, :) + example;
	total(centid, :) = total(centid) + 1;
end

% calculate the new centroid dividing the sum by
% the total
for i=1:K
	centroids(i, :) = sums(i, :) ./ total(i);
end

%%
%% VARIANT: 
%% You could also use the find() method to find directly
% all the examples that belong to a given centroid: 
%% 

for i=1:K
	% find the index where you match the current centroid
	centroid_examples = find(idx == i);
	% now, find the matching examples in X, add them (sum) and divide by total
	centroids(i,:) = sum(X(centroid_examples, :)) / length(centroid_examples);
end

% =============================================================


end

