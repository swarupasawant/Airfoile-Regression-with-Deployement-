import pickle
from flask import Flask, request, app, jsonify,render_template
from flask import Response
import numpy as np 
import pandas as pd 

app = Flask(__name__)
pickle_model = pickle.load(open('model.pkl','rb'))
@app.route('/')
def main():
    #return 'Hello World'
    return render_template('main.html')

#FOR POSTMAN 
@app.route('/predict_api',methods=['POST'])
def predict_api():

    data = request.json['data']
    print(data)
    new_data = [list(data.values())]
    output = pickle_model.predict(new_data)[0]
    return jsonify(output)
    

# FOR HEROKU DEPLOYMENT
@app.route('/predict',methods=['POST'])
def predict():

    data = [float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    print(data)
    
    output = pickle_model.predict(final_features)[0]
    print(output)
    return render_template('main.html',prediction_text = "Airfoil pressure is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)