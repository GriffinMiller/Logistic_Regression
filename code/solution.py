import numpy as np 
from math import exp

'''
Homework2: logistic regression classifier
'''


'''
def logistic_regression(data, label, max_iter, learning_rate):
    '''
    #The logistic regression classifier function.
    
    #Args:
	#data: train data with shape (1561, 3), which means 1561 samples and 
	#	  each sample has 3 features.(1, symmetry, average internsity)
	#label: train data's label with shape (1561,1). 
	#	   1 for digit number 1 and -1 for digit number 5.
	#max_iter: max iteration numbers
	#learning_rate: learning rate for weight update
    #
	#Returns:
	#	w: the seperater with shape (3, 1). You must initilize it with w = np.zeros((d,1))
'''
    N = len(data[0])
    w = np.zeros((len(data[0]),1))
    t = 0 # number of iterations
    dE = np.transpose(len(data[0])*0)
    while( t < max_iter ):
        for i in range(N):
            dE = dE + (-1/N)*(label[i] * data[i][:] / ( 1 + exp(np.dot(np.dot(label[i], np.transpose(w[:])), data[i][:]))))
        w = np.transpose(np.transpose(w) - learning_rate * dE)
        t = t + 1
    return w
'''

def logistic_regression(data, label, max_iter, learning_rate):

    d = len(data[0])
    w = np.zeros((1,d))
    Ein = np.zeros((1,d))
    Summation = np.zeros((1,d))
    numerator = np.zeros((1,d))
    denomintor = 0

    N = len(data)

    for i in range (max_iter):
        for n in range(N):
            numerator = np.dot(data[n],label[n]) #ynxn
            denomintor  = 1 + np.exp(np.dot(np.dot(label[n],w),data[n])) # 1 + e^ynWT*xn
            Summation = np.add(Summation,np.divide(numerator,denomintor)) # Summing our equation 
        Ein = np.divide(Summation,-1 * N) * -1 # Ein = -1/N * Summmation
        w = np.add(w,np.dot(Ein,learning_rate))# w = w + alpha * Ein

    return w

def accuracy(x, y, w):
    '''
    This function is used to compute accuracy of a logsitic regression model.
    
    Args:
    x: input data with shape (n, d), where n represents total data samples and d represents
        total feature numbers of a certain data sample.
    y: corresponding label of x with shape(n, 1), where n represents total data samples.
    w: the seperator learnt from logistic regression function with shape (d, 1),
        where d represents total feature numbers of a certain data sample.

    Return 
        accuracy: total percents of correctly classified samples. Set the threshold as 0.5,
        which means, if the predicted probability > 0.5, classify as 1; Otherwise, classify as -1.
    '''
    n = len(x)
    prob = .5
    correctly_classified = 0
    total_theta = 0
    
    case1_correct = 0;
    case2_correct = 0;
    case1_incorrect = 0;
    case2_incorrect = 0;
    
    for i in range(n):
        s = np.dot(y[i],np.dot(w,x[i]))
        theta = 1/(1 + np.exp(-s))
        if(theta >= prob):
            correctly_classified = correctly_classified + 1
            total_theta = total_theta +1
            if(y[i] ==1):
                case1_correct = case1_correct + 1
            elif( y[i] == -1):
                case2_correct = case2_correct + 1
        else:
            if(y[i] ==1):
                case1_incorrect = case1_incorrect + 1
            elif( y[i] == -1):
                case2_incorrect = case2_incorrect + 1
    
    print()
    print("Average Probability: " + str(total_theta/n))
    print("Drafted     | Correct: " + str(case1_correct) + " | Incorrect: " + str(case1_incorrect))
    print("Not Drafted | Correct: " + str(case2_correct) + " | Incorrect: " + str(case2_incorrect))
    
    return correctly_classified/n

