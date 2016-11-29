#python library
import numpy as np
import random


#define function
def real_function(x):
    x1 = x[0]
    x2 = x[1]
    y1 = -3*np.exp(-(((x1-2)**2)/3)-(((x2-2)**2)/3)) - 4*np.exp(-(((x1+2)**2)/4)-(((x2 +2)**2)/4))
    y2 = np.exp(-0.25 * np.sqrt(x1**2 + x2**2)) * np.cos(2 * np.sqrt(x1**2 + x2**2))
    return y1,y2


#generate dataset
def dataset_generator(n):

    #define domains
    max_x1 = 5
    min_x1 = -5
    max_x2 = 5
    min_x2 = -5

    #half noise range
    noise_range = 0.5

    y = []

    x1 = np.random.rand(n)*(max_x1-min_x1)+min_x1
    x2 = np.random.rand(n)*(max_x2-min_x2)+min_x2

    x = np.array(zip(x1,x2),dtype=np.float32)

    for e in x:
        y.append(real_function(e))

    y = np.array(y,dtype=np.float32)
    dataset = zip(x,y)

    return dataset


#separate dataset into 2 parts: input and output
def separator(train,test):

    train_x = []
    train_y = []
    test_x = []
    test_y = []

    for i in range(len(train)):
        train_x.append(train[i][0])
        train_y.append(train[i][1])

    for i in range(len(test)):
        test_x.append(test[i][0])
        test_y.append(test[i][1])

    train_x = np.array(train_x, dtype = np.float32)
    train_x = np.reshape(train_x,(len(train_x),2))
    train_y = np.array(train_y, dtype = np.float32)
    train_y = np.reshape(train_y,(len(train_y),2))

    test_x = np.array(test_x, dtype = np.float32)
    test_x = np.reshape(test_x,(len(test_x),2))
    test_y = np.array(test_y, dtype = np.float32)
    test_y = np.reshape(test_y,(len(test_y),2))

    return train_x,train_y,test_x,test_y


if __name__ == '__main__':
    dataset = dataset_generator(10)
    print dataset[0][0]
