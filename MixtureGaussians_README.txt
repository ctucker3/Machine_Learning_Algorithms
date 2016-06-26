To run: 

python2 MixtureGaussians.py ./example_data/points.dat

The code will run on the example data, which has two numeric features. The code needs to be checked to see if it runs on a higher 
number of features. 

All functionality should be put into a class. Take the reading functionality away from the class and give it to the user to have them
pass in a numpy matrix.

Function should be added to check to make sure the features are numeric and not categorical. 

************ Algorithm **************

The clusters are initiated with the Kmeans algorithm. Initializing your clusters with Kmeans allows the Mixture of 
Gaussians EM algorithm to converge faster. We can see this is the case for this data set because the log likelihood converges a few iterations into the algorithm. 

Because the Kmeans algorithm initialize the clusters, the number of clusters passed to Mixture of Gaussians could be less than the
the number of clusters specified in the beginning. Becuase Kmeans uses hard assignments, many clusters have zero points in them. So when the user may desire 10 clusters, the algorithm will only produce 4 because there were 6 clusters with no points assigned to them.  

The Kmeans algorithm uses random sampling in order to initiate its clusters. As the code stands now, 30 iterations of Kmeans are performed for the data. 

The Mixture of Gaussians code clusters on both pooled and separate covariance matrices. The pooled covariance matrices are summed over all k (clusters) for all data points in the data set and then divided by N-K. The separate covariance matrices are simply the covariances weighted by the resposibility of class k for each data point.

The clusters are adjusted and determined using an EM algorithm. 

************ References ************

Hastie, T.; Tibishirani, R.; Friedman, J. (2008) The Elements of Statistical Learning: Data Mining, Inference, and 
Prediction. Springer, 2, 50-72.





