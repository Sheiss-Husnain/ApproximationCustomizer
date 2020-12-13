from keras.utils.generic_utils import has_arg
import keras
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



class NeuralNetwork:

    """
    	Class for building Keras models and Sklearn wrapped Keras models.

    	-----------------------------------------------------------
    	Attributes:
    		self.params:		  Dictionary that contains build, compile and fit parameters for keras model.

    		self.layers:		  Dictionary of all Keras compatible layers.

    		self.model:           Keras model

    	-----------------------------------------------------------
    	Functions:
    		fit:		Trains Keras model.

    		predict_proba:		Makes predictions (probability) using trained Keras model.

    		predict:		Makes predictions (class) using trained Keras model.

    		build_model:		returns Keras model.
    	"""

    def __init__(self, params= {}):
        self.params =params
        
        #more Keras layers can be added to this dictionary.  Visit Keras documentation for more info
        self.layers = {'dense': Dense, 'batch_norm': BatchNormalization, 'conv2d': Conv2D, 'flatten': Flatten,
                  'dropout': Dropout, 'maxpool': MaxPooling2D, 'avgpool'
                  : AveragePooling2D}

        if len(self.params)!=0:
            assert isinstance(self.params, dict), print('params parameter passed need to be a dictionary.','fail')
            assert all([k in self.params.keys() for k in ['build', 'compile', 'fit']]), print('Params must contain build, compile and fit keys','fail')
            assert len(self.params.keys())==3, print('Valid keys for params: build, compile and fit', 'fail')
            assert isinstance(self.params['build'],list), print('Value for build key must be a list', 'fail')
            assert isinstance(self.params['compile'], dict) and isinstance(self.params['fit'], dict), print('Values for compile and fit keys must be dictionaries', 'fail')
            assert all([k for d in self.params['build'] for k, v in d.items() if k in self.layers.keys()]), print('layer type is not valid','fail')

            assert all([has_arg(self.layers[k], i) for d in self.params['build'] for k, v in d.items() for i in v.keys() if i not in ['input_dim', 'name','input_shape']]), print('Hyper_parameters for layers are invalid','fail')
            assert all([has_arg(Sequential.compile, k) for k in self.params['compile'].keys()]), print('Model compilation parameters are invalid','fail')
            assert all([has_arg(Sequential.fit, k) for k in self.params['fit'].keys()]) , print('Model fit parameters are invalid','fail')
            self.model= self.build_model()

        else:
            print("params dict is empty")
        


    def fit(self, x,y):
        #Keras does not accept pd.DataFrame as input, only arrays
        try:
            self.model.fit(x,y,**self.params['fit'], verbose=0)
        except Exception as e:
            print('Values for fit parameters are invalid','fail')
            print()
            print('Error: ', e)
        self.is_fit= True
        return self

    def predict_proba(self, x, **kwargs):
        assert self.is_fit, print('Model was not fit properly.', 'fail')

        return self.model.predict_proba(x, **kwargs)


    def predict(self,x):
        assert self.is_fit, print('Model was not fit properly.', 'fail')
        return self.model.predict(x)


#automation of build function required for KerasClassifier wrapper
#in order to grid search hyperparameters using Sklearn RandomizedSearchCV, tunable hyperparameters must be listed as arguments to this build function as shown
#any new hyper_parameters to be grid_searched must be added as parameters to this function
    def build_model(self, optimizer=None, activation=None, epochs=None, batch_size=None, dropout_rate=None, momentum=None, learn_rate=None, metrics=None, loss=None):
        model = Sequential()
        for layer in self.params['build']:
            for k, v in layer.items():
                try:
                    model.add(self.layers[k](**v))
                except Exception as e:
                    print('{} are invalid values for {}'.format(v,k),'fail')
                    print()
                    print('Error: ', e)

        try:
            model.compile(**self.params['compile'])
        except Exception as e:
            print('Values for compilation parameters are invalid','fail')
            print()
            print('Error: ', e)
        return model

    def save(self):
        self.model.model.save('./static/model/my_model')



