import numpy as np
import Kmeans
import matplotlib.pyplot as plt
import math
import sys



def matrixbuild(filepath):
    # Create a list of all the data points
    data = open(filepath,"r").read()
    data = data.split('\n')
    finaldata = []
	
    for entry in data:

        if '   ' in entry:
            finaldata.append(entry.split('   '))

        else:
            finaldata.append(entry.split('  '))
    del finaldata[(len(finaldata)-1)]

    # Convert strings to floats

    for entry in range(0, len(finaldata)):
        for number in range(0, len(finaldata[entry])):
            finaldata[entry][number] = float(finaldata[entry][number])
    
    finaldata = np.asarray(finaldata)

    return finaldata

def update(data_matrix, resp_matrix, cluster_matrix, var_type):

	num_data = len(data_matrix)
	num_cluster = len(cluster_matrix)

	cluster_matrix = np.zeros((num_cluster, len(data_matrix[0])))		# Matrix of means for clusters. 
	pi = np.zeros(num_cluster) 		# Priors for k clusters. 

	for k in range(0, num_cluster):

		Nk = 0.0

		for index in range(0, num_data):

			cluster_matrix[k] = cluster_matrix[k] + (resp_matrix[index][k] * data_matrix[index])
		
			Nk += resp_matrix[index][k]

		cluster_matrix[k] = cluster_matrix[k]/Nk
		pi[k] = Nk/float(num_data)

	covariances = update_covariance(data_matrix, resp_matrix, cluster_matrix, want_class = var_type)

	return cluster_matrix, pi, covariances

		
# This function takes in one data point vector, the cluster matrix, and the covariance matrices of the clusters and returns the responsibility vector
# Which is a K length vector. 

def responsibility(data_vector, cluster_matrix, covariances, priors):

	num_attr = len(data_vector)
	num_clusters = len(cluster_matrix)
	resp_vector = np.zeros(num_clusters)

	for k in range(0, num_clusters):

		base = 1/(np.power(2.0*np.pi, num_attr/2)	* np.sqrt(np.linalg.det(covariances[k])))
		exponent = ((-1/2.0)* np.dot(np.dot(np.transpose(data_vector - cluster_matrix[k]), np.linalg.inv(covariances[k])), (data_vector - cluster_matrix[k])))

		prob = priors[k] * base * np.exp(exponent)
		'''print priors[k]
		print prob'''
		resp_vector[k] = prob

	# Normalize the probabilities
	total = np.sum(resp_vector)
	resp_vector = resp_vector/np.sum(resp_vector)
	
	return resp_vector, total
		
# Build on to find the resp of all the data points at once

def all_resp(data_matrix, cluster_matrix, covariances, priors):
	
	num_attr = len(data_matrix[0])
	num_points = len(data_matrix)
	num_clusters = len(cluster_matrix)
	resp_matrix = np.zeros((num_points, num_clusters))
	tot_vect = np.zeros(num_points)

	for index in range(0, num_points):
	
		resp_matrix[index], tot_vect[index] = responsibility(data_matrix[index], cluster_matrix, covariances, priors)

	return resp_matrix, tot_vect


# Calculates the covariance matrix for the data. If you want to use a pooled covariance matrix for all of the clusters set want_class = "pooled". By default it
# is not pooled. The function input is the data matrix with NxD dimensions, where D is the number of attributes. The responsibility matrix whose dimensions are
# NxK, where K is the number of clusters. The cluster matrix, whose dimensions are KxD. The output of the function is a vector of length K where the kth element	
# is the covariance matrix for the kth cluster. 

def update_covariance(data_matrix, resp_matrix, cluster_matrix, want_class = "no", phase = "iterative"):
   
	num_attr = len(data_matrix[0])
	num_points = len(data_matrix)
	num_clusters = len(cluster_matrix)
	covariances = []		# kth element will be the kth covariance matrix. Even if pooled used, still returns k length vector. 

	# Calculate a pool covariance matrix for all the clusters

	if want_class == "pooled":

		covariance = np.zeros((num_attr, num_attr))
		for k in range(0, num_clusters):
			for index in range(0, num_points):

				# Below is resposnibility of k * Xn - cluster_mean of k * tranpose of Xn - cluster_mean
				term_sum = resp_matrix[index][k] * np.outer((data_matrix[index] - cluster_matrix[k]), np.transpose(data_matrix[index] - cluster_matrix[k]))
				covariance = np.add(covariance, term_sum)

		covariance = covariance/(num_points - num_clusters)  # Covariance matrix divided by (N - K), number of data points minus the number of clusters. 

		for i in range(0, num_clusters):
			covariances.append(covariance)

		return covariances

	else:		# Means separate covariance matrices for each cluster. 
					
		for k in range(0, num_clusters):
			covariance = np.zeros((num_attr, num_attr))		# Fresh covariance since the covariance matrices are different in this option. 
			Nk = 0.0
			for index in range(0, num_points):

				# Below is resposnibility of k * Xn - cluster_mean of k * tranpose of Xn - cluster_mean
				term_sum = resp_matrix[index][k] * np.outer((data_matrix[index] - cluster_matrix[k]), np.transpose(data_matrix[index] - cluster_matrix[k]))
				covariance = np.add(covariance, term_sum)
				Nk += resp_matrix[index][k]

			covariance = covariance/Nk		
			covariances.append(covariance)

		return covariances

		

			

# This function initializes from the output of the K means algorithm and gets our priors and covariance matrices. 

def initialize(data_matrix, cluster_matrix, class_matrix, want_class = "separate"):
	
	num_attr = len(data_matrix[0])
	num_points = len(data_matrix)
	num_clusters = len(cluster_matrix)
	covariances = []		# kth element will be the kth covariance matrix. Even if pooled used, still returns k length vector.
	priors = np.zeros(num_clusters)

	if want_class == "separate":
		
		for k in range(0, num_clusters):
			covariance = np.zeros((num_attr, num_attr))		# Fresh covariance since the covariance matrices are different in this option. 		
			Nk = 0
			for index in range(0, num_points):

				if class_matrix[index][k] == 1:		# only calculate covariance for this point if it is in that class

					term_sum = np.outer((data_matrix[index] - cluster_matrix[k]), np.transpose(data_matrix[index] - cluster_matrix[k]))
					covariance = np.add(covariance, term_sum)
					priors[k] += 1
			
			covariance = covariance/(priors[k] - 1)		# Just the covariance matrix divided by n-1
			covariances.append(covariance)

		priors = priors/float(num_points)

		return covariances, priors

	else:		# Means we want pooled covariances. 

		covariance = np.zeros((num_attr, num_attr))
		for k in range(0, num_clusters):
			for index in range(0, num_points):

				term_sum = np.outer((data_matrix[index] - cluster_matrix[k]), np.transpose(data_matrix[index] - cluster_matrix[k]))
				covariance = np.add(covariance, term_sum)
				priors[k] += class_matrix[index][k]

		covariance = covariance/(num_points - num_clusters)  # Covariance matrix divided by (N - K), number of data points minus the number of clusters. 
	
		priors = priors/float(num_points)

		for i in range(0, num_clusters):
			covariances.append(covariance)

		return covariances, priors
		
# For each data point we take the argmax responsibility to calculate the likelihood
# The idea is that we want to get the maximum cluster assignment probability as close to 1 as possible 
# so that we get the maximum log likelihood with the least uncertainty.  

def likelihood(tot_vector):
	
	likelihood = 0
	
	for observ in tot_vector:

		likelihood += np.log(observ)

	return likelihood

# This function only works for a maximum of 6 clusters
def visualize(data_matrix, resp_matrix):
	
	num_cluster = len(resp_matrix[0])
	cluster_dict = {}
	for i in range(0, num_cluster):
		cluster_dict[i] = [[],[]]		# X and Y lists

	for index in range(0, len(data_matrix)):
		high_prob = np.argmax(resp_matrix[index])						# Find the cluster point is most likely to belong to. 
		cluster_dict[high_prob][0].append(data_matrix[index][0])		# Put the x value in
		cluster_dict[high_prob][1].append(data_matrix[index][1])		# Put the y value in

	colors = ['r', 'b', 'g', 'c', 'y', 'm', 'k']
	color_index = 0
	for k in range(0, num_cluster):
		plt.scatter(cluster_dict[k][0],cluster_dict[k][1], color = colors[color_index])
		color_index += 1
	plt.title('Clusters plotted')
	plt.xlabel('X1')
	plt.ylabel('Y1')
	return plt

def main(vartype, clusters):

	data_matrix = matrixbuild(sys.argv[1])
	tenth = math.floor(len(data_matrix)/float(10))
	if tenth < 1:
		print "Data input is <10 and is too small to cluster on"
		return
	dev_matrix = data_matrix[(len(data_matrix) - tenth):]
	data_matrix = data_matrix[:(len(data_matrix) - tenth)]
	
	t_cluster_matrix, t_class_matrix = Kmeans.main(data_matrix, clusters, 30)			# Initialize the clusters with K-means
	d_cluster_matrix, d_class_matrix = Kmeans.main(dev_matrix, clusters, 30)	
	
	t_like = []
	d_like = []
	
    # Initalize the covariance matrix

	t_covariances, t_priors = initialize(data_matrix, t_cluster_matrix, t_class_matrix, want_class = vartype)
	d_covariances, d_priors = initialize(dev_matrix, d_cluster_matrix, d_class_matrix, want_class = vartype)

    # Now find responsibilities.
	t_resp_matrix, t_tot_matrix = all_resp(data_matrix, t_cluster_matrix, t_covariances, t_priors)
	d_resp_matrix, d_tot_matrix = all_resp(dev_matrix, d_cluster_matrix, d_covariances, d_priors)

	# Calculate the likelihood on initial start point
	t_like.append(likelihood(t_tot_matrix))
	d_like.append(likelihood(d_tot_matrix))

	# Update the means and covariance

	t_cluster_matrix, t_priors, t_covariances = update(data_matrix, t_resp_matrix, t_cluster_matrix, vartype)
	d_cluster_matrix, d_priors, d_covariances = update(dev_matrix, d_resp_matrix, d_cluster_matrix, vartype)
	# Iterate through the repeating steps.
	runs = 20
	print ("Beginning clustering for %s covariance\n\n" % vartype)
	for i in range(0, runs):
		
		sys.stdout.write("\rIteration number %d of %d" % (i+1, runs))
		sys.stdout.flush()
		# Update responsibilities
		t_resp_matrix, t_tot_vect = all_resp(data_matrix, t_cluster_matrix, t_covariances, t_priors)
		d_resp_matrix, d_tot_vect = all_resp(dev_matrix, d_cluster_matrix, d_covariances, d_priors)

		# Update parameters
		t_cluster_matrix, t_priors, t_covariances = update(data_matrix, t_resp_matrix, t_cluster_matrix, vartype)
		d_cluster_matrix, d_priors, d_covariances = update(dev_matrix, d_resp_matrix, d_cluster_matrix, vartype)

		# Find Log Likelihood
		t_like.append(likelihood(t_tot_vect))
		d_like.append(likelihood(d_tot_vect))

	data = visualize(data_matrix, t_resp_matrix)

	return t_like, d_like, data
	
def test():

	print "\nRunning algorithm with 2 clusters"
	t_pool2, d_pool2,data_pool2 = main("pooled", 2)		# These may be different due to Kmeans being done for each of them. 
	t_sep2, d_sep2, data_sep2 = main("separate", 2)
	print "\nRunning algorithm with 3 clusters"
	t_pool3, d_pool3, data_pool3 = main("pooled", 3)		# These may be different due to Kmeans being done for each of them. 
	t_sep3, d_sep3, data_sep3 = main("separate", 3)
	print "\nRunning algorithm with 4 clusters"
	t_pool4, d_pool4, data_pool4 = main("pooled", 4)		# These may be different due to Kmeans being done for each of them. 
	t_sep4, d_sep4, data_sep4 = main("separate", 4)
	print "\nRunning algorithm with 5 clusters"
	t_pool5, d_pool5, data_pool5 = main("pooled", 5)		# These may be different due to Kmeans being done for each of them. 
	t_sep5, d_sep5, data_sep5 = main("separate", 5)

	data_sep5.show()

	plt.plot(t_pool2, color = 'r', label = 'train k = 2 pooled var')
	plt.plot(t_pool3, color = 'k', label = 'train k = 3 pooled var')
	plt.plot(t_pool4, color = 'g', label = 'train k = 4 pooled var') 
	plt.plot(t_pool5, color = 'c', label = 'train k = 5 pooled var') 
	plt.plot(d_pool2,color = 'k', label = 'dev k = 2 pooled var') 
	plt.plot(d_pool3, color = 'y', label = 'dev k = 3 pooled var')
	plt.plot(d_pool4, color = 'm', label = 'dev k = 4 pooled var')
	plt.plot(d_pool5, color = 'k', label = 'dev k = 5 pooled var')
	plt.title('Log Likelihood Pooled Variance')
	plt.xlabel('Runs')
	plt.ylabel('Log Likelihood')
	plt.legend()

	plt.show()

	plt.plot(t_sep2, color = 'r', label = 'train k = 2 separate var')
	plt.plot(t_sep3, color = 'k', label = 'train k = 3 separate var')
	plt.plot(t_sep4, color = 'g', label = 'train k = 4 separate var') 
	plt.plot(t_sep5, color = 'c', label = 'train k = 5 separate var') 
	plt.plot(d_sep2,color = 'k', label = 'dev k = 2 separate var') 
	plt.plot(d_sep3, color = 'y', label = 'dev k = 3 separate var')
	plt.plot(d_sep4, color = 'm', label = 'dev k = 4 separate var')
	plt.plot(d_sep5, color = 'k', label = 'dev k = 5 separate var')
	plt.title('Log Likelihood Pooled Variance')
	plt.xlabel('Runs')
	plt.ylabel('Log Likelihood')
	plt.legend()


	plt.show()
	
	

test()
