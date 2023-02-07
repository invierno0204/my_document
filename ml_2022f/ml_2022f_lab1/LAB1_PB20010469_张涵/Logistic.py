import numpy as np
import matplotlib.pyplot as plt
class LogisticRegression:

    def __init__(self, penalty="l2", gamma=0, fit_intercept=True):
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg

    def sigmoid(self, x):
        """The logistic sigmoid function"""
        y=1./(1.+np.exp(-x))
        return(y)
        
    def fit(self, X, y, lr=0.01, tol=1e-7, max_iter=10000000):
        """
        Fit the regression coefficients via gradient descent or other methods 
        """
        X=np.mat(X)
        y=np.mat(y)
        m,d=np.shape(X)
        b=np.zeros(shape=(d+1,1))
        a=np.ones((m,1),dtype="double")
        X=np.concatenate((X,a),axis=1)
        count=1.
        while  count<=max_iter:
            count+=1.
            b_div=-(X.T)@(y-self.sigmoid(X@b))
            b_div=b_div*lr
            if np.linalg.norm(b_div) <=tol:
                print("iterative error:")
                print(np.linalg.norm(b_div))
                break
            else:
                b=b-b_div 
        return(b)

    def predict(self, X,y,b):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.
        """
        X=np.mat(X)
        y=np.mat(y)
        b=np.mat(b)
        m=len(X)
        a=np.ones((m,1),dtype="double")
        X=np.concatenate((X,a),axis=1)
        y1=X@b
        y1[y1>=0]=1
        y1[y1<0]=0
        TP=0
        FN=0
        FP=0
        TN=0
        for i in range(m):
            if y[i,0]==1 and y1[i,0]==1:
                TP+=1
            elif y[i,0]==1 and y1[i,0]==0:
                FN+=1
            elif y[i,0]==0 and y1[i,0]==1:
                FP+=1
            elif y[i,0]==0 and y1[i,0]==0:
                TN+=1
        data={'TP':TP,'FN':FN,'FP':FP,'TN':TN}
        return(data)
print("all right")


