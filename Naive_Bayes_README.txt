To run:

python2 Naive_Bayes.py ./Adult_Data/a7a.train ./Adult_Data/a7a.dev ./Adult_Data/a7a.test

Put all functionality into a class. Allow the user to specify the range of alphas to search over. 

Take the reading functionality away from the class and leave it to the user, then just have the 
user put in a numpy matrix for the test, dev and train. 

************ Algorithm *****
The script takes about 2 minutes to run because 50 alpha values are tested to show the leveling off of probabilities in a plot produced
afterwards.  

The data matrix is split up into two matrices. One matrix contains all the tuples with positive classification and the other one contains
all the tuples for the negative classifications. 

To capture the effect of Dirichlet smoothing, K*alpha tuples are added onto each matrix. For every value of alpha, one tuple with 
all 1 values and one tuple with all 0 values is added on. When the probabilities are calculated this has the effect of adding alpha to the numerator for all
features and adding K*alpha to the denominator for all features, where K is 2 when the two possible values are 0 and 1. 

Probabilties are calculated for each state of each feature (0 and 1) and added to a probability matrix. 

For the classification of new tuples in the dev set and test set, the state of the feature (0 or 1) is read and pulled from the appropriate
matrix position in the probability matrix and multiplied to the total probability as is performed in Naive Bayes. Of course, the probability
of the given classification is multiplied in the end. 

The classification of the tuple is then assigned based on which classification probability is the highest.

For the choice of alpha, 50 values from 0 to 50 are looped over and the alpha that provides the best accuracy in the dev set is selected. 
Even though alpha is usually small, 50 values were looped over to show the leveling off of the accuracy as alpha got very high.




************ References ************
Hastie, T.; Tibishirani, R.; Friedman, J. (2008) The Elements of Statistical Learning: Data Mining, Inference, and 
Prediction. Springer, 2, 50-72.








