#python library
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

#python script
import get_dataset as d


#visualize loss reduction
def loss_visualizer():

    epoch = []
    train_loss = []
    test_loss = []

    f = open('./result/log', 'r') #load log file
    data = json.load(f)
    f.close()

    value = []

    for i in range(0,len(data)):
        value = data[i]
        epoch.append(value["epoch"])
        train_loss.append(value["main/loss"])
        test_loss.append(value["validation/main/loss"])

    plt.figure(1)
    plt.plot(epoch,train_loss,"b",label = "train LOSS")
    plt.plot(epoch,test_loss,"g",label = "test LOSS")
    plt.yscale('log')
    plt.title("LOSS reduction")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("LOSS")


#visualize regression result in draw 3-D graph
def test_result_visualizer(test_x,test_y,estimated_y):

    test_x1 = []
    test_x2 = []

    actual_y1 = []
    actual_y2 =[]

    estimated_y1 = []
    estimated_y2 =[]

    for i in range (len(test_x)):
        test_x1.append(test_x[i][0])
        test_x2.append(test_x[i][1])
        actual_y1.append(test_y[i][0])
        actual_y2.append(test_y[i][1])

    fig2 = plt.figure(figsize = plt.figaspect(0.5))

    ax1 = fig2.add_subplot(1, 2, 1, projection='3d')
    p1_1 = ax1.scatter3D(test_x1,test_x2,actual_y1,color=(1.0,0,0),marker='o',s=10,label='actual data')
    p1_2 = ax1.scatter3D(test_x1,test_x2,estimated_y1,color=(0,0,1.0),marker='o',s=10,label='estimated data')
    ax1.legend()
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_zlabel("y")

    ax2 = fig2.add_subplot(1, 2, 2, projection='3d')
    p2_1 = ax2.scatter3D(test_x1,test_x2,actual_y2,color=(1.0,0,0),marker='o',s=10,label='actual data')
    p1_2 = ax2.scatter3D(test_x1,test_x2,estimated_y2,color=(0,0,1.0),marker='o',s=10,label='estimated data')
    ax2.legend()
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("y")


#draw 3D graph
def function_visualizer(model):

    x1_mesh = np.arange(-5,5,0.25)
    x2_mesh = np.arange(-5,5,0.25)

    actual_Y1 = []
    actual_Y2 = []
    estimated_Y1 = []
    estimated_Y2 = []

    X1,X2 = np.meshgrid(x1_mesh,x2_mesh)
    X_mesh = []

    for i in range(len(x1_mesh)):
        for j in range(len(x2_mesh)):
            X_mesh.append([x1_mesh[i],x2_mesh[j]])

    for i in range(len(X_mesh)):
        actual_y1,actual_y2 = d.real_function(X_mesh[i])
        actual_Y1.append(actual_y1)
        actual_Y2.append(actual_y2)

    X_mesh = np.array(X_mesh,dtype = np.float32)

    actual_Y1 = np.array(actual_Y1,dtype = np.float32)
    actual_Y2 = np.array(actual_Y2,dtype = np.float32)
    actual_Y1 = np.array(actual_Y1.reshape(len(x1_mesh),len(x2_mesh)))
    actual_Y2 = np.array(actual_Y2.reshape(len(x1_mesh),len(x2_mesh)))

    estimated_Y = model.predictor(X_mesh).data

    for i in range(len(X_mesh)):
        estimated_Y1.append(estimated_Y[i][0])
        estimated_Y2.append(estimated_Y[i][1])

    estimated_Y1 = np.array(estimated_Y1)
    estimated_Y2 = np.array(estimated_Y2)
    estimated_Y1 = np.array(estimated_Y1.reshape((len(x1_mesh),len(x2_mesh))))
    estimated_Y2 = np.array(estimated_Y2.reshape((len(x1_mesh),len(x2_mesh))))

    #draw plot
    fig3 = plt.figure(figsize = plt.figaspect(1.0))
    fig3.set_figheight(15)
    fig3.set_figwidth(15)

    ax1 = fig3.add_subplot(2, 2, 1, projection='3d')
    w1 = ax1.plot_wireframe(X1,X2,actual_Y1,color=(1.0,0,0),label='actual function 1')
    ax1.legend()
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_zlabel("y")

    ax2 = fig3.add_subplot(2, 2, 2, projection='3d')
    w2 = ax2.plot_wireframe(X1,X2,estimated_Y1,color=(0,0,1.0),label='estimated function 1')
    ax2.legend()
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("y")

    ax3 = fig3.add_subplot(2, 2, 3, projection='3d')
    w3 = ax3.plot_wireframe(X1,X2,actual_Y2,color=(1.0,0,0),label='actual function 2')
    ax3.legend()
    ax3.set_xlabel("x1")
    ax3.set_ylabel("x2")
    ax3.set_zlabel("y")

    ax4 = fig3.add_subplot(2, 2, 4, projection='3d')
    w4 = ax4.plot_wireframe(X1,X2,estimated_Y2,color=(0,0,1.0),label='estimated function 2')
    ax4.legend()
    ax4.set_xlabel("x1")
    ax4.set_ylabel("x2")
    ax4.set_zlabel("y")

    plt.tight_layout()
