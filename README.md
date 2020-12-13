# Regression Customizer Web App
<p>
Web application that allows user to input parameters for a custom neural network for regression of most common mathematical functions. </p>

<p>App displays a form to fill out for a neural network and dataset configuration.  Once the form is submitted.  The application
will construct a Keras neural network with the given configurations and train it on the selected dataset configuration.
</p>

<p>The results page, will list various model plots such as training vs. validation error per epoch and selected metric plots.  New_config button can be clicked to restart the process and build a new model.  If satisfied with results, the trained model is saved in static/model folder and can be retrieved for further independent analysis.</p>

<h3>Strict Version Dependencies</h3>
<ul>
    <li> Keras version 2.2.5 </li>
    <li> TensorFlow version 1.14.0 </li>
</ul>

<h3>To run:</h3>

<p>while in command line:</p>
<p>$ export FLASK_APP=app.py </p>
<p>$ flask run </p>
<p>
go to listed link. ex)
 * Running on http://127.0.0.1:5000/

</p>
