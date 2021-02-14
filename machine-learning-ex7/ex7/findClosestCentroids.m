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

  numexamples = length(X);
  
  % iterate through all the examples
  % for each example, find its closest centroid
  for exid = 1:numexamples
  
    example = X(exid, :);
    errcents = inf(size(centroids), 1);
    
    % iterate through all the centroids
    % for each centroid, find the distance (error) to the example
    for centid = 1:K
      centroid = centroids(centid, :);
      errcents(centid) = norm(example - centroid) ^ 2;
    end

    % return the index of the centroid with the closest distance
    % as the index for our current example (at exid)
    [_, idx(exid)] = min(errcents);
  end

% =============================================================

end

