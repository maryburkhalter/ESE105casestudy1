
clear all;
close all;

%% In this script, you need to implement three functions as part of the k-means algorithm.
% These steps will be repeated until the algorithm converges:

  % 1. initialize_centroids
  % This function sets the initial values of the centroids
  
  % 2. assign_vector_to_centroid
  % This goes through the collection of all vectors and assigns them to
  % centroid based on norm/distance
  
  % 3. update_centroids
  % This function updates the location of the centroids based on the collection
  % of vectors (handwritten digits) that have been assigned to that centroid.

% Reads in 28 by 28 images from mnist database as a 784 dimensional vector.
% Each vector has entries whose values are between 0 and 255, indicating
% levels of gray --> 0=completely white, 1=completely black
% A working algorithm produces k centroids where any given image is closest
% to one of these k centroids
%% Initialize Data Set
% These next lines of code read in two sets of MNIST digits that will be used for training and testing respectively.

% training set (1500 images)
train=csvread('mnist_train_1500.csv');
trainsetlabels = train(:,785);
train=train(:,1:784);
train(:,785)=zeros(1500,1);

% testing set (200 images with 11 outliers)
test=csvread('mnist_test_200_woutliers.csv');
% store the correct test labels
correctlabels = test(:,785);
test=test(:,1:784);

% now, zero out the labels in "test" so that you can use this to assign
% your own predictions and evaluate against "correctlabels"
% in the 'cs1_mnist_evaluate_test_set.m' script
test(:,785)=zeros(200,1);

%% After initializing, you will have the following variables in your workspace:
% 1. train (a 1500 x 785 array, containins the 1500 training images)
% 2. test (a 200 x 785 array, containing the 200 testing images)
% 3. correctlabels (a 200 x 1 array containing the correct labels (numerical
% meaning) of the 200 test images

%% To visualize an image, you need to reshape it from a 784 dimensional array into a 28 x 28 array.
% to do this, you need to use the reshape command, along with the transpose
% operation.  For example, the following lines plot the first test image

figure;
colormap('gray'); % this tells MATLAB to depict the image in grayscale
testimage = reshape(test(1,[1:784]), [28 28]);
% we are reshaping the first row of 'test', columns 1-784 (since the 785th
% column is going to be used for storing the centroid assignment.
imagesc(testimage'); % this command plots an array as an image.  Type 'help imagesc' to learn more.

%% After importing, the array 'train' consists of 1500 rows and 785 columns.
% Each row corresponds to a different handwritten digit (28 x 28 = 784)
% plus the last column, which is used to index that row (i.e., label which
% cluster it belongs to.  Initially, this last column is set to all zeros,
% since there are no clusters yet established.

%% This next section of code calls the three functions you are asked to specify

k= ; % set k
max_iter= ; % set the number of iterations of the algorithm

%% The next line initializes the centroids.  Look at the initialize_centroids()
% function, which is specified further down this file.

centroids=initialize_centroids(train,k);

%% Initialize an array that will store k-means cost at each iteration

cost_iteration = zeros(max_iter, 1);

%% This for-loop enacts the k-means algorithm

% form clusters around centroids, then keep moving the centroids until the
% moves are no longer significant

% keep the old clusters and current clusters so we can get the max
% difference betweeen centroid positions every time

% the difference change is over once the list of differences is all zeros
% and therefore the algorithm has converged

% we want to implement k-median clustering because the median is more
% resiliant to outliers than the mean is (median is the middle of the data
% wheras the mean is an average of the data)
 
for iter=1:max_iter
    
      % FILL THIS IN!
    
end

%% This section of code plots the k-means cost as a function of the number
% of iterations

figure;
% FILL THIS IN!


%% This next section of code will make a plot of all of the centroids
% Again, use help <functionname> to learn about the different functions
% that are being used here.

figure;
colormap('gray');

plotsize = ceil(sqrt(k));

for ind=1:k
    
    centr=centroids(ind,[1:784]);
    subplot(plotsize,plotsize,ind);
    
    imagesc(reshape(centr,[28 28])');
    title(strcat('Centroid ',num2str(ind)))

end

%% Function to initialize the centroids
% This function randomly chooses k vectors from our training set and uses them to be our initial centroids
% There are other ways you might initialize centroids.
% ***Feel free to experiment.***
% Note that this function takes two inputs and emits one output (y).

% assign a digit label to each centroid by picking the most common label
% within the cluster
function y=initialize_centroids(data,num_centroids)

random_index=randperm(size(data,1));

centroids=data(random_index(1:num_centroids),:);

y=centroids;

end

%% Function to pick the Closest Centroid using norm/distance
% This function takes two arguments, a vector and a set of centroids
% It returns the index of the assigned centroid and the distance between
% the vector and the assigned centroid.

function [index, vec_distance] = assign_vector_to_centroid(data,centroids)

%assign the data to a centroid by taking its norm and finding the centroid 
%that it's norm is the smallest distance away from

for i=1:size(centroids(:,1))
  %using the norm to calculate the distance from the current row
  %vector to a centroid
    dist_centroids = norm(data(1:784)-centroids(i,1:784)); 
  if (i == 1)
  %smallest distance is the smallest norm value which occurs in the first
  %iteration where i = 1 for centroids
    min_dist_centroids = dist_centroids;
    idx = 1;
  elseif (dist_centroids < min_dist_centroids)
  %when the centroid at an index is closer to the vector in question that
  %will become distance of that vector, and the index will be the one at
  %which this occurs
    vec_distance = dist_centroids;
    idx = i;
    end
end
end


%% Function to compute new centroids using the mean of the vectors currently assigned to the centroid.
% This function takes the set of training images and the value of k.
% It returns a new set of centroids based on the current assignment of the
% training images.

% centroid assignment is in column 785
function new_centroids=update_Centroids(data,K)
new_centroids = zeros(K, 785);
for i=1:K
    for j=1:size(data)
        if (data(j,785)==i)
            new_centroids(i,:) = 


end