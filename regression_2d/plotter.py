import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time

"""
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
"""

#define function
def real_function(x,y):
    z = -3*np.exp(-(((x-2)**2)/3)-(((y-2)**2)/3)) - 4*np.exp(-(((x+2)**2)/4)-(((y +2)**2)/4))
    #z = np.exp(-0.25 * np.sqrt(x**2 + y**2)) * np.cos(2 * np.sqrt(x**2 + y**2))
    return z


#generate dataset
def dataset_generator(n):

    #define domains
    max_x = 5
    min_x = -5
    max_y = 5
    min_y = -5

    error_range = 0.5

    x = []
    y = []
    z = []

    for i in range(n):
        x.append(random.uniform(min_x,max_x))
        y.append(random.uniform(min_y,max_y))
        z.append(real_function(x[i],y[i]))
        #z.append(real_function(x[i],y[i])+random.uniform(-error_range,error_range))
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    return x,y,z


def real_function2(x):
    x1 = x[0]
    x2 = x[1]
    z = -3*np.exp(-(((x1-2)**2)/3)-(((x2-2)**2)/3)) - 4*np.exp(-(((x1+2)**2)/4)-(((x2 +2)**2)/4))
    #z = np.exp(-0.25 * np.sqrt(x1**2 + x2**2)) * np.cos(2 * np.sqrt(x1**2 + x2**2))
    return z


#draw 3-D graph
def visualizer(x,y,z):

    fig1 = plt.figure(1)
    ax = Axes3D(fig1)
    p = ax.scatter3D(x,y,z,color=(1.0,0,0),marker='o',s=10,label='test')
    #"""
    x_mesh = np.arange(-5,5,0.25)
    y_mesh = np.arange(-5,5,0.25)
    X,Y = np.meshgrid(x_mesh,y_mesh)
    Z = real_function(X,Y)
    w = ax.plot_wireframe(X,Y,Z,color=(0,0,1.0),label='actual')
    #"""
    #plt.xlim(-1,1)
    #plt.ylim(-1,1)
    plt.title("3-D visualization")
    plt.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


#draw 3D graph
def function_visualizer():
    fig2 = plt.figure(2)
    ax2 = Axes3D(fig2)
    x1_mesh = np.arange(-5,5,0.25)
    x2_mesh = np.arange(-5,5,0.25)
    X1,X2 = np.meshgrid(x1_mesh,x2_mesh)
    X = []
    for i in range(0,len(x1_mesh)):
        X.append([x1_mesh[i], x2_mesh[i]])
    #X = np.array(X)
    Z = real_function(X1,X2)
    #w = ax2.plot_wireframe(X1,X2,Z,color=(0,0,1.0),label='actual')
    print X1

#main
if __name__ == '__main__':

    N = 1000

    x,y,z = dataset_generator(N)

    visualizer(x,y,z)

    function_visualizer()

    plt.show()
    x1_mesh = np.arange(-5,5,0.25)
    x2_mesh = np.arange(-5,5,0.25)
    X_mesh = []
    for i in range (len(x1_mesh)):
        X_mesh.append([x1_mesh[i],x2_mesh[i]])
    X_mesh = np.array(X_mesh)
    #print X_mesh
