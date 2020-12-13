from flask import Flask, render_template, redirect, url_for, request

from src.plot_generator import PlotGenerator

import os

app = Flask(__name__, static_url_path='/static')
#src ="/static/plots/yourimage.jpg"

@app.route('/')
def root():
    return render_template("home.ejs")

@app.route('/results', methods=["POST", "GET"])
def results():
	if request.method=="POST":

		data = dict(request.form)
		package= package_builder(data)
		print(package)
		x= PlotGenerator(package)
		x.input_plot()
		x.model_summary()
		x.error_plot()
		x.prediction_plot()
		x.save()
		x.clear_model()

		data['layer_num']=int(data['layer_num'])

		imgs = os.listdir('./static/plots')
		imgs=sorted(imgs)

		payload= [imgs,data]

		return render_template("results.html",payload= payload)

	else:
		return render_template("results.html")




def package_builder(data):
	model_params={}
	model_params['build']=[{"dense":{}} for i in range(int(data['layer_num']))]
	for i in range(int(data['layer_num'])):
		model_params['build'][i]['dense']["units"]= int(data["units"+str(i)])
		model_params['build'][i]['dense']["activation"]= data["activation"+str(i)]
		if i==0:
			model_params['build'][i]['dense']["input_shape"]=(1,)


	model_params['compile']= {"loss":data["loss"], "optimizer":data["optimizer"], "metrics":[k.replace("METRIC_","") for k in data.keys() if "METRIC_" in k]}
	model_params['fit']= {"batch_size":int(data["batch_size"]),"epochs":int(data["epochs"]), "validation_split":float(data["validation_split"]) }

	dataset_params= {}
	dataset_params["funct"]= data["funct"]
	dataset_params["range"]= [int(data["range0"]), int(data["range1"])]
	dataset_params["num_samples"]= int(data["num_samples"])
	package={"model_params":model_params,"dataset_params":dataset_params}
	return package


