'''
    TEST SCRIPT FOR MULTIVARIATE LINEAR REGRESSION
    AUTHOR Eric Eaton, Vishnu Purushothaman Sreenivasan
'''

'''
Numpy is a standard library in python that lets you do matrix and vector operations like Matlab in python.
Check out documentation here: http://wiki.scipy.org/Tentative_NumPy_Tutorial
If you are a Matlab user this page is super useful: http://wiki.scipy.org/NumPy_for_Matlab_Users 
'''
import numpy as np
from numpy.linalg import *

#Matplotlib provides matlab like plotting tools in python 
import matplotlib.pyplot as plt

# our linear regression class
from linreg import LinearRegression

#All the modules needed for 3d surface plots
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter

#.............BARCA VO DICH...........................

#Plotting tools is already
#Tiktok : BARCELONA

def plotData3D( X, y , to_block = True) :
    fig = plt.figure()
    ax = fig.add_subplot(111,projection = '3d')
    ax.scatter(X[:,1], X[:,2], y, c= 'r', marker='o')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y')
    plt.show()

def plotRegSurface(lr_model, X, y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(X[:,1], X[:,2], y, c= 'r', marker='o', label = 'Training Data')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y')
    plt.show()

    x1_vals = np.linspace(min(X[:,1]), max(X[:,1]), 100)
    x2_vals = np.linspace(min(X[:,2]), max(X[:,2]), 100)
    X1_vals, X2_vals = np.meshgrid(x1_vals,x2_vals)
    Z = np.zeros(X1_vals.shape)

    for i in range(X1_vals.shape[0]):
        for j in range(X1_vals.shape[1]):
            instance = np.matrix([1, X1_vals[i,j], X2_vals[i,j]])
            Z[i,j] = lr_model.predict(instance)

    ax.plot_surface(X1_vals,X2_vals, X, cmap = 'viridis', alpha = 0.5,edgecolor = 'k', linewidth = 0.5)
    plt.show()

def visualizeObjective3D(lr_model, t1_vals, t2_vals, X, y):
    T1,T2 = np.meshgrid(t1_vals, t2_vals)
    n,p = T1.shape
    Z = np.zeros(T1.shape)

    for i in range(n):
        for j in range(p):
            theta_vals = np.matrix([T1[i,j], T2[i,j]])
            Z[i,j] = lr_model.computeCost(X,y,theta_vals)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(T1, T2, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('Theta1')
    ax.set_ylabel('Theta2')
    ax.set_zlabel('Cost')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()




if __name__ == "__main__":
    '''
        Main function to test multivariate linear regression
    '''
    
    # load the data
    filePath = "C:/Users/DELL/OneDrive - Hanoi University of Science and Technology/Desktop/Homework1/CIS419--Decision-Tree-Learning-Linear-Regression/data/multivariateData.dat"

    file = open(filePath,'r')
    allData = np.loadtxt(file, delimiter=',')

    X = np.matrix(allData[:,:-1])
    y = np.matrix((allData[:,-1])).T

    n,d = X.shape
    
    # Standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std
    
    # Add a row of ones for the bias term
    X = np.c_[np.ones((n,1)), X]
    
    # initialize the model
    init_theta = np.matrix(np.random.randn((d+1))).T
    n_iter = 2000
    alpha = 0.01

    # Instantiate objects
    lr_model = LinearRegression(init_theta = init_theta, alpha = alpha, n_iter = n_iter)
    lr_model.fit(X,y)

    # Compute the closed form solution in one line of code
    thetaClosedForm = (X.getT()*X).getI()*X.getT()*y
    print ("thetaClosedForm: "), thetaClosedForm



    # Visualize the objective function convex shape in 3D
    theta1_vals = np.linspace(-10, 10, 100)
    theta2_vals = np.linspace(-10, 10, 100)
    visualizeObjective3D(lr_model, theta1_vals, theta2_vals, X, y)

    # Plot the regression surface
    plotRegSurface(lr_model, X, y)



