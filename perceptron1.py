import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import time
import math

# This is the sign function. May not need when using gradient descent though. 
def sign(value):
	if value > 0:
		return 1
	elif value == 0:
		return 0
	else:
		return -1

def accuracytest(predicted,actual):
    total = len(predicted)
    correct = 0.0
    # Checks if the actual matches the predicted.
    for i in range(0,len(predicted)):
        if predicted[i] == actual[i]:
            correct += 1
    return correct,total


############## Gradient descent method #########################################


# This performs the gradient descent version of the perceptron.
# Input is the nth tuple and the weight vector.
def predict(data_vector, weights):
	wtx = np.dot(np.transpose(weights), data_vector)
	tn = (math.exp(wtx) - math.exp(-wtx))/(math.exp(wtx)+math.exp(-wtx))
	return tn, wtx

def gprime(wtx):
	# The derivative of the activation function. (4e^(2a))/((e^(2a)+1)^2).
	a = math.exp(2*wtx)
	try:
		anns = (4*a)/((a + 1)**2)
		
	except ArithmeticError:

		exit()
		
	return anns 



def adjustweights(data_vector, tn, yn, wtx, weights,learn):
	# Updates the weight vector. w = w - (tn-yn)g'(wtx)x
	weights = np.subtract(weights, np.multiply(learn, np.dot((tn - yn)*gprime(wtx),data_vector)))
	#print weights
	#print (tn - yn)
	#print np.multiply((tn - yn)*gprime(wtx),data_vector)
	return weights

def update_progress(progress,runs):
	sys.stdout.write('\r[{0}] {1}%'.format('#'*(int(progress/(5.0/runs))), progress*100))
    	sys.stdout.flush()
    	

def gradienttrain(data_matrix, real_classes, runs):
	# Initialize the weights vector at all 0's. The length of the vector is determined by the number of attributes. Add one more feature
	# for b. 
	weights = np.asarray([0]*(data_matrix.shape[1]+1))
	predicted = []
	curr_run = 1
	best_accuracy = 0
	best_weights = None
	progress = 0
	update_progress(progress, runs)
	# Here we will loop through the training data matrix as many times as the runs specifies to keep refining the accuracy. 
	while curr_run <= runs:

		for obsindex in range(0, data_matrix.shape[0]):
			# insert a 1 into the vector
			data_vector = np.insert(data_matrix[obsindex], data_matrix[obsindex].shape[0], 1.0)
			tn, wtx = predict(data_vector, weights)
			predicted.append(sign(tn))
			weights = adjustweights(data_vector, tn, real_classes[obsindex], wtx, weights, 0.2)
		correct, total = accuracytest(predicted, real_classes)
		accuracy = correct/total
		if accuracy > best_accuracy:
			best_accuracy = accuracy
			best_weights = weights

		
		curr_run += 1
		predicted = []

		# Shuffle the data_matrix so that the nodes
		progress += (1/(runs*1.0))
		update_progress(progress,runs)
		#np.random.shuffle(data_matrix)

		
	
	#print ("\nAfter %d runs, the best accuracy was %s" % (curr_run, str(round(best_accuracy,2))))
	# Return the weights for classification
	
	return best_weights

def classify(test_matrix, test_class, weights):
	predicted = []
	for obsindex in range(0, test_matrix.shape[0]):
		test_vector = np.insert(test_matrix[obsindex], test_matrix[obsindex].shape[0], 1.0)
		tn, wtx = predict(test_vector, weights)
		predicted.append(sign(tn))
	correct, total = accuracytest(predicted, test_class)
	accuracy = correct/total
	#print ("For the SVM classifier the accuracy on the test set was %s" % (str(round(accuracy, 2))))
	return (round(accuracy,4)*100)
	


############################### Normal Perceptron Method ###############################################


def normpredict(data_vector, weights):
	wtx = np.dot(np.transpose(weights), data_vector)
	tn = sign(wtx)
	return tn, wtx

def normadjustweights(data_vector, tn, yn, wtx, weights):
	# Updates the weight vector. w = w - (tn-yn)g'(wtx)x
	if tn != yn:
		weights = np.add(weights,np.dot(yn, data_vector))
	return weights

def normtrain(data_matrix, real_classes, runs):
	# Initialize the weights vector at all 0's. The length of the vector is determined by the number of attributes.
	weights = np.asarray([0]*data_matrix.shape[1])
	predicted = []
	curr_run = 1
	# Here we will loop through the training data matrix as many times as the runs specifies to keep refining the accuracy. 
	print predict(data_matrix[1], weights)
	while curr_run <= runs:
		for obsindex in range(0, data_matrix.shape[0]):
			#print obsindex
			tn, wtx = normpredict(data_matrix[obsindex], weights)
			predicted.append(tn)
			weights = normadjustweights(data_matrix[obsindex], tn, real_classes[obsindex], wtx, weights)
		correct, total = accuracytest(predicted, real_classes)
		accuracy = correct/total
		print ("Run number %d had an accuracy of %s" % (curr_run, str(accuracy)))
		curr_run += 1
		predicted = []
		



# python /home/chris/Documents/Machine_Learning/Perception_SVM.py




