from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
mp = pickle.load(open('model.pkl', 'rb'))
#print(type(mp))

@app.route("/")
def home():
	return render_template('index_final1.html')

@app.route("/predict", methods=['POST'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = mp.predict(final_features)
    output = round(prediction[0],2)
    return render_template('index_final1.html', prediction_text='Water quality index is {}'.format(output))

	#print(request.method)
# 		data1 = request.form['Temperature']
# 		data2 = request.form['pH']
# 		data3 = request.form['DO']
# 		data4 = request.form['COD']
# 		data5 = request.form['BOD']
# # 		data6 = request.form['f']
# # 		data7 = request.form['g']
# # 		data8 = request.form['h']
# # 		data9 = request.form['i']
# # 		data10 = request.form['j']
# # 		data11 = request.form['k']
# # 		data12 = request.form['l']
# # 		data13 = request.form['m']
# # 		data14 = request.form['n']
# 		arr = np.array([[data1, data2, data3, data4, data5]])
# 		pred = mp.predict(arr)
		# int_features = [int(x) for x in request.form.values()]
		# final_features = [np.array(int_features)]
		# prediction = mp.predict(final_features)
# 		output = round(pred[0], 2)
# 	return render_template('index_final1.html', prediction_text='House price will be $ {}'.format(output))
# 	return render_template('index_final1.html')

if __name__ == '__main__':
	app.run(debug=True,port=8000)