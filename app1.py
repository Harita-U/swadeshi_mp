#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 08:54:02 2021

@author: haritauppal
"""

from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
mp = pickle.load(open('model1.pkl', 'rb'))
#print(type(mp))

@app.route("/")
def home():
	return render_template('index.html')

@app.route("/model1", methods=['POST', 'GET'])
def model():
	if request.method == 'POST':

	#print(request.method)
		data1 = request.form['a']
		data2 = request.form['b']
		data3 = request.form['c']
		data4 = request.form['d']
		data5 = request.form['e']
		
		arr = np.array([[data1, data2, data3, data4, data5]])
		pred = mp.predict(arr)
		# int_features = [int(x) for x in request.form.values()]
		# final_features = [np.array(int_features)]
		# prediction = mp.predict(final_features)
		output = round(pred[0], 2)
		return render_template('index.html', prediction_text='water quality index is $ {}'.format(output))
	return render_template('index.html')

    
if __name__ == '__main__':
	app.run(debug=True)