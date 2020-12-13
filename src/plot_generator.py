import matplotlib
matplotlib.use('Agg')
import io
import numpy as np
from keras.utils import plot_model
from .neural_network import NeuralNetwork
from .dataset import Dataset
import random
import os, shutil
import matplotlib.pyplot as plt
from keras import backend as K
import time




class PlotGenerator:


    def __init__(self, params):
        self.model= NeuralNetwork(params["model_params"])
        self.dataset= Dataset(params["dataset_params"])
        self.X,self.y=self.dataset.datasets()
        self.funct= self.dataset.function()
        self.clear_plots()
        self.model.fit(self.X,self.y)
        self.accuracy = self.model.model.evaluate(self.X,self.y)



    
    #SHOULD BE A STATIC METHOD!!!! 
    def clear_plots(self):
        folder = './static/plots/'
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def input_plot(self):
        plt.figure(1)
        plt.plot(self.X,self.y, 'b', label=self.funct)
        plt.title(self.funct)
        plt.legend()
        plt.xlabel('x')
        plt.ylabel(self.funct)
        plt.savefig(fname="./static/plots/a_input_plot" + str(time.time())+".png")
        plt.close()



    
    def error_plot(self):
        validation_losses= [k for k in self.model.model.history.history.keys() if "val_" in k]
        training_losses=[k for k in self.model.model.history.history.keys() if "val_" not in k]

        for e in training_losses:
            for v in validation_losses:
                if e in v:
                    plt.figure()
                    plt.plot(self.model.model.history.history[e] , 'b', label= e)
                    plt.plot(self.model.model.history.history[v] , 'r', label= v)
                    plt.legend()
                    plt.xlabel('Epochs')
                    plt.ylabel('Error')
                    plt.title(e+" vs. "+ v)
                    plt.savefig(fname="./static/plots/error_plot"+e+ str(time.time())+".png")
                    plt.close()


    
    def accuracy(self):
        return self.accuracy
        
    def model_summary(self):

        plot_model(self.model.model, to_file='./static/plots/b_model'+ str(time.time())+".png")

        s = io.StringIO()
        self.model.model.summary(print_fn=lambda x: s.write(x + '\n'))
        model_summary = s.getvalue()
        s.close()
        plt.rc('figure', figsize=(8, 6))
        plt.text(0.01, 0.05, model_summary)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('./static/plots/c_model_summary'+ str(time.time())+".png")
        plt.close()
        

 
    def clear_model(self):
        K.clear_session()
        

    def prediction_plot(self):
        plt.figure(1)
        plt.plot(self.X,self.model.model.predict(self.X), 'r', label= 'Model Prediction')
        plt.plot(self.X,self.y, 'b', label= 'Actual Output' )
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title("Model Prediction vs. Actual output")
        plt.savefig(fname="./static/plots/z_prediction_plot"+ str(time.time())+".png")
        plt.close()

    def save(self):
        self.model.save()
