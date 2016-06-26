import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import time
import math

def convertvalue(tn):
	
	if tn > 1:
		return 1
	elif tn == 0:
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

def predict(data_vector, weights, b):
	wtx = np.dot(np.transpose(weights), data_vector) 
	tn = wtx + b
	return tn, wtx

def adjustweights(data_vector, tn, yn, wtx, b, weights, num_data, cost, learn):

	# Updates the weight vector if our classification was not correct
	if 1 - yn*tn > 0:

		# This is the change to the w function. Learn*(1/N * w - C * yn * xn)
		change = np.multiply(learn, np.subtract(np.multiply(1/num_data,weights), np.multiply(cost*yn, data_vector)))
		weights = np.subtract(weights, change)

		# Changing b
		b = b + learn * cost * yn

	# If it was correct we make the margin wider by shrinking w.
	else:
		weights = np.subtract(weights, np.multiply(1/float(num_data), weights))

	return weights, b

def update_progress(progress,runs):
	sys.stdout.write('\r[{0}] {1}%'.format('#'*(int(progress/((1/(runs*1.0))*5.0))), progress*100))
    	sys.stdout.flush()

def train(data_matrix, real_classes, runs, learn, cost, code = "No"):
	# Initialize the weights vector at all 0's. The length of the vector is determined by the number of attributes.
	num_data = data_matrix.shape[0]
	b = 0
	weights = np.asarray([0]*data_matrix.shape[1])
	predicted = []
	curr_run = 1
	best_accuracy = 0
	best_weights = None
	best_b = None
	progress = 0
	# Here we will loop through the training data matrix as many times as the runs specifies to keep refining the accuracy. 
	while curr_run <= runs:
		for obsindex in range(0, data_matrix.shape[0]):
			tn, wtx = predict(data_matrix[obsindex], weights, b)
			predicted.append(convertvalue(tn))
			weights, b = adjustweights(data_matrix[obsindex], tn, real_classes[obsindex], wtx, b,weights, num_data, cost, learn)
		correct, total = accuracytest(predicted, real_classes)
		accuracy = correct/total
		if accuracy > best_accuracy:
			best_accuracy = accuracy
			best_weights = weights
			best_b = b
		if code == "train":
			progress += (1/(runs*1.0))
			update_progress(progress,runs)
		#print ("SVM iteration number %d out of %d with C = %f" % (curr_run,runs, cost))
		
		# Randomly shuffle the order of the data tuples for the next round. 
		
		#np.random.shuffle(data_matrix)

		curr_run += 1
		predicted = []

	#print ("With C = %f, the best accuracy was %s" % (cost, str(best_accuracy)))

	return best_accuracy, best_weights, best_b

def find_c(dev_matrix,dev_class, runs_each, learn):
	best_accuracy = 0
	best_c = 0
	C = 0.0
	progress = 0
	predicted = []
	accuracy_list = []
	C_list = []
	while C <=10.0:
		accuracy, weights, b = train(dev_matrix, dev_class, runs_each, learn, C)
		accuracy_list.append(accuracy)
		C_list.append(C)
		if accuracy > best_accuracy:
			best_accuracy = accuracy
			best_c = C
		progress += (0.1)
		update_progress(progress,10/1.0)
		C += 1.0
	return best_c, accuracy_list, C_list

def classify(test_matrix, test_class, weights, b):
	predicted = []
	for obsindex in range(0, test_matrix.shape[0]):
		tn, wtx = predict(test_matrix[obsindex], weights, b)
		predicted.append(convertvalue(tn))
	correct, total = accuracytest(predicted, test_class)
	accuracy = correct/total
	#print ("For the SVM classifier the accuracy on the test set was %s" % (str(round(accuracy, 2))))

	return (round(accuracy,4)*100)

