#create X and y dataset for two-one mapping function
import numpy as np


class Dataset:
    def __init__(self, params):
        self.funct= params["funct"]
        #self.input= params["input_space"]
        self.num_samples= params["num_samples"]
        self.range=params["range"]

    def datasets(self):
        
        if self.funct== "Sine Function":
            #create X and y dataset for two-one mapping function
            X= np.linspace(self.range[0],self.range[1],num= self.num_samples)
            self.X=X.reshape(len(X),1)
            y= np.sin(X)
            self.y=y.reshape(len(y),1)
            return self.X,self.y
        elif self.funct == "Cosine Function":
            X= np.linspace(self.range[0],self.range[1],num= self.num_samples)
            self.X=X.reshape(len(X),1)
            y= np.cos(X)
            self.y=y.reshape(len(y),1)
            return self.X,self.y
        elif self.funct == "Quadratic Function":
            X= np.linspace(self.range[0],self.range[1],num= self.num_samples)
            self.X=X.reshape(len(X),1)
            y= np.square(X)
            self.y=y.reshape(len(y),1)
            return self.X,self.y
        elif self.funct == "Logarithmic Function":
            X= np.linspace(self.range[0],self.range[1],num= self.num_samples)
            self.X=X.reshape(len(X),1)
            y= np.log10(X)
            self.y=y.reshape(len(y),1)
            return self.X,self.y
        elif self.funct == "Exponential Function":
            X= np.linspace(self.range[0],self.range[1],num= self.num_samples)
            self.X=X.reshape(len(X),1)
            y= np.exp(X)
            self.y=y.reshape(len(y),1)
            return self.X,self.y
        elif self.funct == "Cubic Function":
            X= np.linspace(self.range[0],self.range[1],num= self.num_samples)
            self.X=X.reshape(len(X),1)
            y= np.power(X,3)
            self.y=y.reshape(len(y),1)
            return self.X,self.y
        elif self.funct == "Square Root Function":
            X= np.linspace(self.range[0],self.range[1],num= self.num_samples)
            self.X=X.reshape(len(X),1)
            y= np.sqrt(X)
            self.y=y.reshape(len(y),1)
            return self.X,self.y        
            

    def function(self):
        return self.funct
        


