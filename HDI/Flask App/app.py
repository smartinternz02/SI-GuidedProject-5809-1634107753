import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle
app = Flask(__name__)
model = pickle.load(open('HDI.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/Prediction',methods=['POST','GET'])
def prediction():
    return render_template('indexnew.html')
@app.route('/Home',methods=['POST','GET'])
def my_home():
     return render_template('home.html')
@app.route('/predict',methods=['POST','GET'])
def predict():
     input_features = [float(x) for x in request.form.values()]
     features_value = [np.array(input_features)]
     features_name = ['Country','Life expectancy','Mean years of schooling','Gross national income (GNI) per capita','Internet Users']
     
     df = pd.DataFrame(features_value, columns=features_name)
     
     #predictions using the loaded model file
     output = model.predict(df)
     print(round(output[0][0],2))
     print(type(output))
     y_pred = round(output[0][0],2)
     if(y_pred >= 0.3 and y_pred <= 0.4) :
        return render_template("resultnew.html",prediction_text = 'Low HDI' + str(y_pred))
     elif(y_pred >= 0.4 and y_pred <= 0.7):
        return render_template("resultnew.html",prediction_text = 'Medium HDI' + str(y_pred))
     elif(y_pred >= 0.7 and y_pred <= 0.8):
        return render_template("resultnew.html",prediction_text = 'High HDI' + str(y_pred))
     elif(y_pred >= 0.8 and y_pred <= 0.94):
        return render_template("resultnew.html",prediction_text = 'Very High HDI' + str(y_pred))
     else :
        return render_template("resultnew.html",prediction_text = 'The given values do not match the range of values of the model. Try giving the values in the mentioned range'+str(y_pred))
     return render_template('result.html', prediction_text=output)
 
if __name__ == '__main__':
    app.run(debug=True,port=5000)
    