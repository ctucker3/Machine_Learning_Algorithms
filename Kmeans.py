# This function is a dependent of the main Mixture of Gaussians method.
# This script is used to find the initial starting clusters for the Mixture of Gaussians EM algorithm. 

import numpy as np
import math
import matplotlib.pyplot as plt
import sys

# THis function takes the data matrix with n data points as rows and d attributes as columns. Attributes must be continuous. 
# Will return a dictionary of clusters 1-K and the class vectors for the data points that corresponds to the same index as the data matrix. 

def sample(data_matrix, K):
	
	num_data = len(data_matrix)
	sample_size = int(math.floor(num_data/float(K)))		# Size of starting clusters. 

	rando = list(range(0,num_data))
	rando = np.random.permutation(rando)		# Get a random permutation of indices. 

	class_matrix = np.zeros((num_data, K))		# Initialize our classification vector

	num_attr = len(data_matrix[0])

	cluster_matrix = np.zeros((K, num_attr))		# Initialize the cluster means matrix

	for i in range(0, K):

		randoslice = rando[0:sample_size]		# Get the random sample indices
		rando = rando[(sample_size-1):(num_data-1)]		# Cut off the sample indices

		for index in randoslice:		

			class_matrix[index][i] = 1			# Mark the appropriate classification vector place

	# Find the means of the clusters. 
	cluster_matrix = clustermeans(data_matrix, class_matrix, cluster_matrix)

	return class_matrix, cluster_matrix

	
# Input is the matrix of data points, the class matrix that has NxK dimensions where K is the number of clusters and the cluster matrix
# which has KXD dimensions, where D is the number of attributes for each data point. Returns a cluster matrix with updated means. 

def clustermeans(data_matrix, class_matrix, cluster_matrix):

	for i in range(0, len(cluster_matrix)):

		mean = np.zeros(len(cluster_matrix[0]))		# Our means vector which will be an aggregate of our data point values
		cluster_size = 0							# Number of data points in the cluster

		for index in range(0, len(data_matrix)):

			mean = np.add(mean, (class_matrix[index][i] * data_matrix[index]))		# adds rnk * xn to the mean vector
			cluster_size += class_matrix[index][i]									# Increments cluster size if the point is in that cluster.

		if cluster_size != 0:
			mean = mean/cluster_size		# Mean vector complete. 

		cluster_matrix[i] = mean

	return cluster_matrix


# This function takes in a data vector, Xn, and the cluster matrix and returns a vector of length K with distances between point Xn and cluster k. 

def distances(data_vector, cluster_matrix):

	dist_vector = np.zeros(len(cluster_matrix))

	for k in range(0, len(cluster_matrix)):
		
		dist_vector[k] = np.linalg.norm(data_vector - cluster_matrix[k])

	return dist_vector



# This function takes in the same paramaters as the clustermeans function and returns the square distance value (the objective function). 

def objective(data_matrix, class_matrix, cluster_matrix):

	J = 0

	for k in range(0, len(cluster_matrix)):

		for index in range(0, len(data_matrix)):

			if class_matrix[index][k] == 1:

				J += np.linalg.norm(data_matrix[index] - cluster_matrix[k])

	return J

def assign(data_matrix, cluster_matrix):

	num_data = len(data_matrix)

	class_matrix = np.zeros((num_data, len(cluster_matrix)))

	for i in range(0, len(data_matrix)):

		dist_vector = distances(data_matrix[i], cluster_matrix)

		cluster = np.argmax(dist_vector)

		class_matrix[i][cluster] = 1

	return class_matrix

# This will eliminate any class that has no assignments in it. 

def emptycheck(class_matrix, cluster_matrix):
	class_counts = np.zeros(len(class_matrix[0]))

	for index in range(0, len(class_matrix)):
		for k in range(0, len(class_matrix[0])):
			class_counts[k] += class_matrix[index][k]

	ind = 0
	bad = []
	while ind < len(class_counts):
		if class_counts[ind] == 0:
			bad.append(ind)
		ind += 1
	print ("\n\n%d clusters had no objects in them" % len(bad))
	for entry in bad:
		class_matrix = np.delete(class_matrix, entry, 1) 		# Delete the column of the class that has no counts
		cluster_matrix = np.delete(cluster_matrix, entry, 0)
		for index in range(0,len(bad)):
			bad[index] = bad[index] - 1

	return class_matrix, cluster_matrix
	
def main(data_matrix, K, runs):

	Js = []

	class_matrix, cluster_matrix = sample(data_matrix, K)
	Js.append(objective(data_matrix, class_matrix, cluster_matrix))

	for i in range(0, runs):

		sys.stdout.write("\rKmeans iteration number %d of %d" % (i+1, runs))
		sys.stdout.flush()

		class_matrix = assign(data_matrix, cluster_matrix)

		cluster_matrix = clustermeans(data_matrix, class_matrix, cluster_matrix)
		
		Js.append(objective(data_matrix, class_matrix, cluster_matrix))

	# Has the ability to make a plot from Js if it is wanted. 
	class_matrix, cluster_matrix = emptycheck(class_matrix, cluster_matrix)		# This removes any empty clusters. 

	return cluster_matrix, class_matrix
		
		
		

		

		
		
		
