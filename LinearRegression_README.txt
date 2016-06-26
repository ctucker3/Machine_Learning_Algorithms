
** Run the program **

python .\Adult_predict.py .\a7a(test).test

To test the script run the command python2 run the command:

python2 Linear_Regression.py ./Adult_Data/a7a.train ./Adult_Data/a7a.test

This script needs some work. First it needs to be put into a class for better functionality. In this class
there should be a function to find the best lambda given the dev data set (implement in code) and then the weight 
vector for the trained model so a "predict" function can be called later on the test data. 
sklearns linear regression model is a good example for what to go for. 

The reading operations on the data sets should also be convert to pandas. This will reduce the bulk a bit. 

************ Files *********
In the package you will find the "Chris_Tucker_LinRegress.py" file and the a7a(train).train file and the a7a(test).test file 
for your convenience. 

************ Algorithm *****
My algorithm runs on the equation W = ((X* X^T + Lam * I) ^(-1)) * (X^T) * Y

Where W is the vector of weights for the linear regression equation, X is the matrix
with each row being an observation. Lam is the lambda regularization factor, in this 
case applied for the ridge regression type of linear regression. 
Y is the vector of classifications for the training set of observations. To find the
classifications for the testing set the equation,

YP = XP * W

was used. YP stands for the vector of predicted classifications. XP stands for the 
matrix of observations to be classified. And W are the weights that were found from
the training data above.

Lambda was determined by trying every value from 0 to 1000 with an increment of 0.1.
The discussion of this can be seen in the results. 

************ Instructions ***
To run the scripts the training data set, submitted with the regression script must 
be in the same directory as the regression script. To run the program, through the command
line type python {your_directory_here}\Chris_Tucker_LinRegress.py {your_test_file}

************ Results *******
As was stated above, lambda was chosen by iterating through every value from 0 to 1000 on an increment 
of 0.1. The lambda value that produced the best accuracy was 81.2, yet only improved accuracy on the 
development data set by 0.003. The accuracy on the training set, the development set, and the test set
should all be around 84.6%. 

************ References ************
Hastie, T.; Tibishirani, R.; Friedman, J. (2008) The Elements of Statistical Learning: Data Mining, Inference, and 
Prediction. Springer, 2, 50-72.







