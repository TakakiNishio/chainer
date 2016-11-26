#python library
import numpy as np
import random

#define function
def real_function(x1,x2):
    z = -3*np.exp(-(((x1-2)**2)/3)-(((x2-2)**2)/3)) - 4*np.exp(-(((x1+2)**2)/4)-(((x2 +2)**2)/4))
    #z = np.exp(-0.25 * np.sqrt(x1**2 + x2**2)) * np.cos(2 * np.sqrt(x1**2 + x2**2))
    return z


#generate dataset
def dataset_generator(n):

    #define domains
    max_x1 = 5
    min_x1 = -5
    max_x2 = 5
    min_x2 = -5

    #half noise range
    noise_range = 0.5

    x = []
    y = []

    for i in range(n):
        x1 = random.uniform(min_x1,max_x1)
        x2 = random.uniform(min_x2,max_x2)
        x.append([x1,x2])
        y.append(real_function(x1,x2))
        #y.append(real_function(x1,x2) + random.uniform(-noise_range,noise_range)) #add noise

    x = np.array(x, dtype = np.float32)
    y = np.array(y, dtype = np.float32)

    x = np.reshape(x,(len(x),2))
    y = np.reshape(y,(len(y),1))

    return x,y
