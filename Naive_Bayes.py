
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import time

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
        datalist[i]=  datalist[i]

    # X is now the observation matrix
    X = np.asarray(datalist)
    # Y is our prediction variable matrix
    Y = np.asarray(classifications)
    # Returns both the observation matrix and the classification matrix.
    return X,Y


# This function reads the classifications and splits the matrix into a positively classified matrix and a negatively
# classified matrix. 
def splitmatrix(matrix,classifications):
	# Loop through the classifications and make a list of all that are positive and all that are negative. 
	positives = []
	negatives = []
	for i in range(0,len(classifications)):
		if classifications[i] > 0:
			positives.append(i)
		else:
			negatives.append(i)
	
	# For every entry in the positive list take that row from the data matrix and put it in the positive matrix
	posmatrix = []
	for i in positives:
		posmatrix.append(matrix[i])
	
	# Likewise for the negative matrix
	negmatrix = []
	for i in negatives:
		negmatrix.append(matrix[i])

	
	# Return the matrices
	posmatrix = np.asarray(posmatrix)
	negmatrix = np.asarray(negmatrix)
	return posmatrix, negmatrix

# This function adds vectors of 1's onto the matrices for Dirichlet smoothing

def dirichlet(posmatrix,negmatrix, alpha):
	while alpha > 0.0:
		posmatrix = np.append(posmatrix, [[1] * 123], axis = 0)
		posmatrix = np.append(posmatrix, [[0] * 123], axis = 0)
		negmatrix = np.append(negmatrix, [[1] * 123], axis = 0)
		negmatrix = np.append(negmatrix, [[0] * 123], axis = 0)
		alpha = alpha - 1.0
	return posmatrix, negmatrix


# This function will take the positive classification matrix and the negative classification matrix and create the
# positive probability matrix and the negative classification matrix. 
def probmatrix(posmatrix, negmatrix):

	# Transpose the matrices so each row is a feature
	posmatrix = np.transpose(posmatrix)
	negmatrix = np.transpose(negmatrix)
	# For each feature calculate P(feature|class) throw it into a matrix as column 1 and For each feature calculate P(not feature|class) throw it into a matrix as column 2
	posprobmatrix = []
	posobserv = float(posmatrix.shape[1])
	for row in posmatrix:
		posoccur = np.sum(row)/(posobserv)
		posprobmatrix.append([posoccur, (1-posoccur)])

	# Do the same for the negative matrix
	negprobmatrix = []
	negobserv = float(negmatrix.shape[1])
	for row in negmatrix:
		negoccur = np.sum(row)/(negobserv)
		negprobmatrix.append([negoccur, (1-negoccur)])
	
	# Return the probability matrices. 
	posprobmatrix = np.asarray(posprobmatrix)
	negprobmatrix = np.asarray(negprobmatrix)
	return posprobmatrix,negprobmatrix


# This function will take the data matrix of entries to classify and will return a vector of classifications. 
def classify(posprobmatrix, negprobmatrix, posmatrix, negmatrix, classifymatrix):
	# For each row in classify matrix pull probability corresponding to the value and multiply to the total prob
	probpos = float(posmatrix.shape[0])/(posmatrix.shape[0] + negmatrix.shape[0])
	probneg = 1.0 - probpos
	classifications = []
	# Both positive and negative classification probability calculated in this loop. 
	for row in classifymatrix:
		totalposprob = 1.0
		totalnegprob = 1.0
		for featureindex in range(0,len(row)):
			if row[featureindex] > 0.5:
				totalposprob = totalposprob * (posprobmatrix[featureindex][0])
				totalnegprob = totalnegprob * (negprobmatrix[featureindex][0])
			else:
				totalposprob = totalposprob * (posprobmatrix[featureindex][1])
				totalnegprob = totalnegprob * (negprobmatrix[featureindex][1])


	# Multiply the probability of a positive or negative classification
		totalposprob = totalposprob * probpos
		totalnegprob = totalnegprob * probneg

	# Enter the higher classification of the two classifications into the classification vector
		if totalposprob > totalnegprob:
			classifications.append(1)
		else:
			classifications.append(-1)
	#print (classifications[0:100])
	# Return the vector of classifications
	classifications = np.asarray(classifications)
	return classifications

# Accuracy check to see how well the predicted classifications match the real classifications. 

def accuracytest(predicted,actual):
    total = len(predicted)
    correct = 0.0
    # Checks if the actual matches the predicted.
    for i in range(0,len(predicted)):
        if predicted[i] == actual[i]:
            correct += 1
    return correct,total

# Function to run a series of alpha values for Dirichlet smoothing on the dev set and determine the best alpha. 
def decidealpha(posmatrix, negmatrix, classifymatrix, realclassifications, alphastart, alphaend, increment = 1):
	alphastart = float(alphastart)
	alphaend = float(alphaend)
	increment = float(increment)
	topaccuracy = 0.0
	bestalpha = 1.0
	alphalist = []
	alphaaccuracy = []
	# Does what the main function does but loops through different alpha values to determine the most accurate one. 
	while alphastart <= alphaend:
		posmatrix, negmatrix = dirichlet(posmatrix,negmatrix,alphastart)
		PosProb, NegProb = probmatrix(posmatrix,negmatrix)
		classifications = classify(PosProb, NegProb, posmatrix, negmatrix, classifymatrix)
		correct, total = accuracytest(classifications, realclassifications)
		alphalist.append(alphastart)
		alphaaccuracy.append(float(correct)/float(total))
		#print ("Accuracy is %s for alpha = %d" % ((correct/(total*1.0)), alphastart))
		
		# Check if the current alpha is better than the best to date alpha
		if (correct/(total*1.0)) > topaccuracy:
			topaccuracy = (correct/(total*1.0))
			bestalpha = alphastart
		alphastart += 1.0
	
	return bestalpha, alphalist,alphaaccuracy


# Main function to wrap all functions together. 
def main():
	# Load the matrices 
	start = time.time()
	trainset = sys.argv[1] 
	devset = sys.argv[2]
	testset = sys.argv[3]
	X,Y = matrixbuild(trainset)	
	PX, PY = matrixbuild(devset)	
	TX, TY = matrixbuild(testset)
	# Decide which alpha is best	
	Pos, Neg = splitmatrix(X,Y)
	alpha, alphalist, alphaaccuracy = decidealpha(Pos,Neg,PX,PY,0,50)
	# Classify test data
	Pos, Neg = dirichlet(Pos,Neg, alpha)
	PosProb, NegProb = probmatrix(Pos,Neg)
	classifications = classify(PosProb, NegProb, Pos, Neg, TX)
	correct, total = accuracytest(classifications, TY)
	print ("\n Correctly classified %d out of %d" %(correct, total))
	print ("The optimal alpha is %d" % alpha)
	print ("Accuracy = %s %%" % (str(round(float(correct)/total,3)*100)))
	plt.plot(alphalist, alphaaccuracy)
	plt.ylabel('Accuracy')
	plt.xlabel('Alpha')
	plt.title('Accuracy for Given Alphas')
	end = time.time()
	#print ("This took %d seconds to run" % (end-start))
	plt.show()

if __name__ == "__main__":
	main()

# python3 /home/chris/Documents/Machine_Learning/Naive_Bayes.py
