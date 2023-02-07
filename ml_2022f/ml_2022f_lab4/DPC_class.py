import pandas as pd
import numpy as np
import math
from sklearn.metrics import davies_bouldin_score as dbs

class DPC():
    def __init__(self, dc, data, rho_rate,delta_rate,num):
        self.dc = dc
        self.data = data
        self.label = np.zeros(self.data.shape[0])
        self.rho_rate = rho_rate
        self.delta_rate=delta_rate
        self.num=num
    def Create_Rho(self, X):
        m = self.data.shape[0]
        rho = np.zeros(m)
        for i in range(m):
            for j in range(m) :
                if X[i][j] < self.dc and i!=j:
                    rho[i] += 1
        return rho
    def Create_distance_mat(self):
        m = self.data.shape[0]
        X = np.zeros([m,m])
        for i in range(m):
            for j in range(m):
                X[i][j] = math.sqrt(np.dot(self.data[i] - self.data[j],self.data[i] - self.data[j]))
        return X
    def Create_Delta(self,X, rho):
        m = self.data.shape[0]
        delta = np.zeros(m)
        for i in range(m):
            lower = float("inf")
            hint = 0
            for j in range(m):
                if rho[j] > rho[i]:
                    hint = 1
                    if X[i][j] < lower:
                        lower = X[i][j]
            if hint == 1:
                delta[i] = lower
            else:
                delta[i] = np.max(X[i])
        return delta
    def select_center(self, rho, delta):
        number = 1
        m = self.data.shape[0]
        rho_threshold = self.rho_rate * (max(rho) - min(rho)) + min(rho)
        delta_threshold = self.delta_rate * (max(delta) - min(delta)) + min(delta)
        for i in range(m):
            if rho[i] >= rho_threshold and delta[i] >= delta_threshold:
                self.label[i] = number
                number+=1
    def select_center_rd(self,rho,delta):
        m = self.data.shape[0]
        rd=np.zeros(m)
        for i in range(m):
            rd[i]=rho[i]*delta[i]
            argsort_rd =np.argsort(rd)
            e= argsort_rd[::-1]
            center=e[:self.num+1]
            for i in range(center.shape[0]):
                self.label[center[i]]=i
    def get_all_label(self, X, rho):
        m = self.data.shape[0]
        for i in range(m):
            if self.label[i] != 0:
                continue
            else:
                self.get_label(i, X, rho)
        return self.label
    def find_nearest(self, k, X, rho):
        item=0
        lower = float("inf")
        m = self.data.shape[0]
        for i in range(m):
            if rho[i] > rho[k]:
                if X[i][k] < lower:
                    lower = X[i][k]
                    item = i
        return item
    def get_label(self,k, X, rho):
        if self.label[k] != 0:
            return self.label[k]
        else: 
            self.label[k] = self.get_label(self.find_nearest(k, X, rho), X, rho)
            return self.label[k]
    def DBI(self):
        score=dbs(self.data,self.label)
        return score