# RegressionCustomizer

Web application that allows user to input parameters for a custom neural network for regression of most common mathematical functions. 
App displays a form to fill out for a neural network and dataset configuration.  Once the form is submitted.  The application
will construct a Keras neural network with the given configurations and train it on the selected dataset configuration.

The results page, will list various model plots such as training vs. validation error per epoch and selected metric plots.  New_config button can be clicked to restart the process and build a new model.  If satisfied with results, the trained model is saved in static/model folder and can be retrieved for further independent analysis.
Demo_Video.mp4 for video demo of app
static/model/my_model for your trained Keras neural network model

Strict Version Dependencies

     Keras version 2.2.5 
     TensorFlow version 1.14.0 


To run:
while in command line:
$ export FLASK_APP=app.py 
$ flask run 

go to listed link. ex)
 * Running on http://127.0.0.1:5000/
