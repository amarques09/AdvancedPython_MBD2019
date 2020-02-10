import datetime as dt
from flask import Flask, request
from ie_bike_model.util import read_data
from ie_bike_model.model import predict

app = Flask(__name__)


@app.route("/")
def hello():
    name = request.args.get("name", "World")
    return "Hello, " + name + "!"


@app.route("/predict")
def get_predict():
    
    df = read_data()
    
    weather_avg = df['weathersit'].mean()
    temp_avg = df['temperature_C'].mean()
    feeling_avg = df['feeling_temperature_C'].mean()
    humidity_avg = df["humidity"].mean()
    windspeed_avg = df["windspeed"].mean()
    
    
    parameters = dict(request.args)
    parameters["date"] = dt.datetime.fromisoformat(parameters.get("date", "2011-01-01"))
    parameters["weathersit"] = int(parameters.get("weathersit", weather_avg))
    parameters["temperature_C"] = float(parameters.get("temperature_C", temp_avg))
    parameters["feeling_temperature_C"] = float(parameters.get("feeling_temperature_C",feeling_avg))
    parameters["humidity"] = float(parameters.get("humidity",humidity_avg))
    parameters["windspeed"] = float(parameters.get("windspeed",windspeed_avg))
    
    start_prediction = dt.datetime.now()
    result = predict(parameters,model = parameters.get('model'))
    end_prediction_time = dt.datetime.now() - start_prediction
    
    return {"result": result, "date": parameters['date'], "computation time/seconds": end_prediction_time.total_seconds()}
