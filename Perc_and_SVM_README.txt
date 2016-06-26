To run:

python2 Perc_and_SVM.py ./Adult_Data/a7a.train ./Adult_Data/a7a.test

I wrote the perceptron and SVM algorithm at the same time so right now they are in the same script. 

Each should be put into its own class. Give data reading responsibility to the user, have them put in a numpy matrix. 

Remove the progress bars, or if keeping, fix the SVM progress bar. When finding the cost variable it does not work. 

Allow the user to specify the search space, and the increment for the cost variable. 

Need to fix the issue with perceptron overflow error. When too many iterations are run, a floating point overflow error occurs. See
algorithm section for details. 

************ Algorithm **************

The perceptron training is based on stochastic gradient descent. 

For the intercept, a 0 is added to the weights vector, and a 1 is added to the end of each vector. 

To train the perceptron, 100 iterations through the data set are used, with the most accurate iteration being chosen as the weights moving
forward into the testing. The dev set is not used for the perceptron. 

The algorithm will give an error if the iterations are set to more than 100. This is because the tanh(wtx) function will produce an overflow error as the number of sig figs exceeds the allowable number. The learning rate for the perceptron is set to 0.2 and this allows the algorithm to run for more iterations. 

The SVM algorithm uses standard SVM. This algorithm does not implement the dual form yet. For the cost variable, C, values from 0 to 1 with increments of 0.025 are tested for the best accuracy. 

The SVM algorithm uses 20 iterations through the data set to test each C. I have found that iterations over 20 did not improve the 
accuracy of the C's by much.

In the end all of the C values with their accuracies are plotted. 

A learning rate of 0.5 is used for SVM. 


************ References ************

Hastie, T.; Tibishirani, R.; Friedman, J. (2008) The Elements of Statistical Learning: Data Mining, Inference, and 
Prediction. Springer, 2, 50-72.





