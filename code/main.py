from solution import *
from dataHelper import *
import numpy as np

#Use for testing the training and testing processes of a model
def train_test_a_model(modelname, train_data, train_label, test_data, test_label, max_iter, learning_rate):
    print(modelname+" testing...")
    # max iteration test cases 
    for i, m_iter in enumerate(max_iter):
        w = logistic_regression(train_data, train_label, m_iter, learning_rate[1])
        Ain, Aout = accuracy(train_data, train_label, w), accuracy(test_data, test_label, w)
        print("max iteration testcase%d: Train accuracy: %f, Test accuracy: %f"%(i, Ain, Aout))
        print(w)
        # learning rate test cases
    for i, l_rate in enumerate(learning_rate):
        w = logistic_regression(train_data, train_label, max_iter[3], l_rate)
        Ain, Aout = accuracy(train_data, train_label, w), accuracy(test_data, test_label, w)
        print("learning rate testcase%d: Train accuracy: %f, Test accuracy: %f"%(i, Ain, Aout))
    print(modelname+" test done.")	


def test_logistic_regression():
    max_iter = [100, 200, 300, 500]
    learning_rate = [0.1, 0.2, 0.5]
    dataloc = "../data/test.txt"
    train,test = get_train_and_test_data(get_manipulated_data(dataloc))
    np.savetxt("../data/train_data.csv", train, delimiter=",")
    np.savetxt("../data/test_data.csv", test, delimiter=",")
    train_data,train_label = split_data(train)
    test_data, test_label = split_data(test)
    train_test_a_model("logistic regression", train_data, train_label, test_data, \
                          test_label, max_iter, learning_rate)


if __name__ == '__main__':
	test_logistic_regression()
