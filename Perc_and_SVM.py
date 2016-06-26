import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import time
import math
import perceptron1
import svm
import argparse

def matrixbuild(filepath):
    # Create a list of all the data points
    data = open(filepath,"r").read()
    data = data.split()

    # Create a list of tuples

    datalist = []
    datatuple = ()
    classifications = [float(data[0])]
    # This number will index each tuple so that 0 value data entries can be filled in accordingly.
    currentnum = 0

    # This is the number of features in our data set. This will be used to fill entries with 0's later.
    maxx = 123

    for item in data[1:]:
        # If the item is a classification symbol we start another tuple for the next entry and append the filled tuple
        # to the datalist. The classification is also stored.
        if item == "+1" or item == "-1":
            classifications.append(float(item))

            # Fill in the remaining values with 0's so the tuple is completely full.
            while currentnum != maxx:
                datatuple = datatuple + (0.0,)
                currentnum += 1
            datalist.append(datatuple)
            datatuple = ()
            currentnum = 0.0

        else:
            # This value retrieves the number of the data entry from the entry.
            itemnum = float(item[:(len(item)-2)])

            # Here we fill in all the data entries between the previous point to the current one with 0's.
            while itemnum != (currentnum + 1):
                datatuple = datatuple + (0.0,)
                currentnum += 1
            datatuple = datatuple + (1.0,)
            currentnum = itemnum

    # Fix the last tuple that was missed. This will work fine, but can be patched up later for elegance.
    while len(datatuple)!= maxx:
        datatuple = datatuple +(0.0,)
    datalist.append(datatuple)

    # This will be the first column in the matrix and will be a column of all 1's for the intercept.
    for i in range(0,len(datalist)):
        datalist[i]=  datalist[i]

    # X is now the observation matrix
    X = np.asarray(datalist)
    # Y is our prediction variable matrix
    Y = np.asarray(classifications)
    # Returns both the observation matrix and the classification matrix.
    return X,Y

def main():
	parser = argparse.ArgumentParser(description='Run SVM and Perceptron algorithms of Adult Data Set.')
	parser.add_argument('Training_filename', help='Training file')
	parser.add_argument('Test_filepath', help = 'Test File')
	args = parser.parse_args()
	dev = args.Training_filename
	test = args.Test_filepath

	print "Loading the data files...\n"
	X,Y = matrixbuild(args.Training_filename)
	DX,DY = matrixbuild(dev)
	TX, TY = matrixbuild(test)

	print "Training for the perceptron.\n"
	perc_weights = perceptron1.gradienttrain(X,Y,100)

	print "\n\nChecking accuracy on the test set.\n"
	perc_accuracy = perceptron1.classify(TX, TY, perc_weights)
	
	print ("The accuracy of the perceptron on the test set was %s%%\n" % perc_accuracy)
	

	#-------------Run the SVM algorithm------------#

	# find_c(dev_matrix,dev_classes, runs_each, learn)
	print "Finding best c from the dev set...\n"
	c, c_accuracy, c_list = svm.find_c(DX, DY, 20, 0.5)

	# train(data_matrix, real_classes, runs, learn, cost)

	print ("\n\nTraining for the SVM with C = %f\n" % c)
	acc, svm_weights, b = svm.train(X, Y, 100, 0.5, c, "train")
	
	# classify(test_matrix, test_class, weights, b)

	print "\n\nChecking accuracy on the test set.\n"
	svm_accuracy = svm.classify(TX,TY,svm_weights, b)

	print ("The accuracy of the SVM on the test set was %s%%" % svm_accuracy)

	plt.plot(c_list, c_accuracy)
	plt.xlabel("Cost value")
	plt.ylabel("Accuracy")
	plt.title("C vs. Accuracy")
	plt.show()

main()
	
