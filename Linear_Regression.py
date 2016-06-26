import numpy as np
import numpy.linalg as linalg
import sys
import os

# Loads file into a data matrix with each row being another observation. First column is set to all one for the
# intercept. The function input is a filepath and the output is a matrix and a vector of classifications.

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
        datalist[i]=  datalist[i] + (1.0,)

    # X is now the observation matrix
    X = np.asarray(datalist)
    # Y is our prediction variable matrix
    Y = np.asarray(classifications)
    # Returns both the observation matrix and the classification matrix.
    return X,Y

# This is the function to find the weight parameters from the training set. The input to this function is the model
# matrix (observation matrix) and the classifications of those entries and an optional entry is lambda. Lambda is 0 if
# not specified.

def trainregression(X,Y,lam=0):
    # w is a vector of our weights
    # This is based off the equation W = ((X * X^T + Lambda*I)^(-1)) X^T * Y
    # When a lambda is not specified in the paramaters, the default is 0 and it has no effect on the equation.
    w = np.dot(np.dot(linalg.pinv(np.add(np.dot(np.transpose(X),X),np.dot(lam,np.identity(124)))),np.transpose(X)),Y)

    return w

# This function finds the classifications of the test data based on the weights taken from the training data.
# The Input is 'Test' which is the matrix of observations to classify, and the weights from the training data set.

def classify(Test, Weights):

    classifications = np.dot(Test,Weights)
    # Assigns the -1 classification to anything < 0 and the 1 classification to anything >= 0.
    for index in range(0, len(classifications)):
        if classifications[index] < 0.0:
            classifications[index] = -1.
        else:
            classifications[index] = 1.

    return classifications

# This function checks the accuracy of the classifications by comparing the actually classifications vs. the predicted
# classification. The input is the vector of predicted classifications and the vector of actual classifications.

def accuracytest(predicted,actual):
    total = len(predicted)
    correct = 0.0
    # Checks if the actual matches the predicted.
    for i in range(0,len(predicted)):
        if predicted[i] == actual[i]:
            correct += 1
    return correct,total

# This function finds the ideal lambda value for the equation. The input is the file path of the training data, file
# path of the development data, the start of the range, end of range, and increment.

def lambdafind(train_filename,dev_filename,start,end,increment):

    # Calls the run function to process the files.
    X,Y,XP,YP,maxvalue = main(dev_filename, "Dev")

    print maxvalue
    max = None

    # This tests a range of lambda's.
    increment = float(increment)
    end = float(end)
    start = float(start)
    lam = start
    while lam <= end:
        Weights = trainregression(X,Y,lam)
        Predicted = classify(XP,Weights)
        accuracy = accuracytest(Predicted,YP)
        if  accuracy > maxvalue:
            max = lam
            maxvalue = accuracy
        lam += increment
        print lam
    print max, maxvalue

# This function is the main function and ties all of the functions together. The inputs are the file paths to the
# training data, the testing data, and the optional is the trial, the two options being "test" or "dev".

def main(trial = "test"):
    # Build the training matrix and testing matrix from the flat files.
    # Retrieves the training data set from the same directory as the script, therefore be sure they are in the same
    # directory.
	
	print sys.argv[1]

	X, Y = matrixbuild(sys.argv[1])

	# The testing data set.
	XP, YP = matrixbuild(sys.argv[2])

	# Find the weights of the training data
	Weights = trainregression(X,Y,81.2)

	# Find the predicted classifications of the testing matrix.
	Predicted = classify(XP,Weights)

	# This function was also used for the lambdafind function so the purpose of the function must be specified in the
	# optional paramter 'trial'. If none is specified, the default is "test".

	if trial == "test":
		correct, total = accuracytest(Predicted,YP)
		print "\n"
		print ("%d correctly classified out of %d" % (correct,total))
		accuracy = round(correct/total,3)
		print ("Accuracy = %s%%" % (round(correct/total,3)*100))
	else:
		return X,Y,XP,YP,accuracytest(Predicted,YP)

if __name__ == '__main__':
    main()


