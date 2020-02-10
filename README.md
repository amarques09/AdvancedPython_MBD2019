Amanda Marques

# Bike sharing prediction model

###how to build it from scratch:

#Individual Assignment Steps:
#
#	1. $ git clone https://github.com/IE-Advanced-Python/ie-bike-sharing-model-lib-ref.
#	2. cd ie-bike-sharing-model-lib-ref/
#	3. nano README.md 
#	4. Add your name to it
#	5. nano app.py
#	6. Copy the code of app.py
#	7. git remote remove origin
#	8. git remote add origin https://github.com/amarques09/AdvancedPython_MBD2019.git
#	9. git add .
#	10. git status
#	11. git commit -m "commiting app.py"
#	12. git push -u origin master
#	13. Go to src
#	14. Ie-bike-sharing
#	15. Nano model.py
#	16. Create a new Function:
		# New function for Ridge Regression
		
#		def train_ridge(hour):
		    # Avoid modifying the original dataset at the cost of RAM
#		    hour = hour.copy()
		
#		    hour_d = pd.get_dummies(hour)
#		    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
#		    hour_d.columns = [
#		        regex.sub("_", col) if any(x in str(col) for x in set(("[", "]", "<"))) else col
#		        for col in hour_d.columns.values
#		    ]
		
#		    hour_d = hour_d.select_dtypes(exclude="category")
		
#		    hour_d_train_x, _, hour_d_train_y, _, = split_train_test(hour_d)
		
#		    ridge = Ridge(alpha=1.0)
		
#		    ridge.fit(hour_d_train_x, hour_d_train_y)
#		    return ridge
#	17. Change the train and persist function:
#		def train_and_persist(model_dir=None, hour_path=None, model="xgboost"):
#		    hour = read_data(hour_path)
#		    hour = preprocess(hour)
#		    hour = dummify(hour)
#		    hour = postprocess(hour)
		
		    # TODO: Implement other models?
		    	
#		        if model == "xgboost":
#		                a_model = train_xgboost(hour)
#		        elif model == "ridge":
#		                a_model = train_ridge(hour)
		
		
#		    model_path = get_model_path(model_dir)
		
#		    joblib.dump(a_model, model_path)
		     	
		
#		    model_path = get_model_path(model_dir)
		
#		    joblib.dump(model, model_path)
		
#	18. Modify the predict function:
#	def predict(parameters, model_dir=None, model = 'xgboost'):
#	    """Returns model prediction.
#	
#	    """
#	    model_path = get_model_path(model_dir)
#	    if not os.path.exists(model_path):
#	        train_and_persist(model_dir=model_dir, model = model)
	
#	    model = joblib.load(model_path)
	
#	    input_dict = get_input_dict(parameters)
#	    X_input = pd.DataFrame([pd.Series(input_dict)])
	
#	    result = model.predict(X_input)
	
	    # Undo np.sqrt(hour["cnt"])
#	    return int(result ** 2)
	
#	19. Go to app.py and modify the result output and call the data - so we are able to do the mean of each column:
#		  @app.route("/predict")
#def get_predict():
#    
#    df = read_data()
#    
#    weather_avg = df['weathersit'].mean()
#    temp_avg = df['temperature_C'].mean()
#    feeling_avg = df['feeling_temperature_C'].mean()
#    humidity_avg = df["humidity"].mean()
#    windspeed_avg = df["windspeed"].mean()   
#    parameters = dict(request.args)
#    parameters["date"] = dt.datetime.fromisoformat(parameters.get("date", "2011-01-01"))
#    parameters["weathersit"] = int(parameters.get("weathersit", weather_avg))
#    parameters["temperature_C"] = float(parameters.get("temperature_C", temp_avg))
#    parameters["feeling_temperature_C"] = float(parameters.get("feeling_temperature_C",feeling_avg))
#    parameters["humidity"] = float(parameters.get("humidity",humidity_avg))
#    parameters["windspeed"] = float(parameters.get("windspeed",windspeed_avg))
#    
#    start_prediction = dt.datetime.now()
#    result = predict(parameters,model = parameters.get('model'))
#    end_prediction_time = dt.datetime.now() - start_prediction
#    
#    return {"result": result, "date": parameters['date'], "computation time/seconds": end_prediction_time.total_seconds()}#

#	20. Git add .
#	21. Git status
#	22. Git commit -m "commiting changes"
#	23. Git push -u origin master

## Usage

To install the library:

```
$ # pip install ie_bike_model  # If I ever upload this to PyPI, which I won't
$ pip install .
```

Basic usage:

```python
>>> import datetime as dt
>>> from ie_bike_model.model import train_and_persist, predict
>>> train_and_persist()
>>> predict({
...     "date": dt.datetime(2011, 1, 1, 0, 0, 0),
...     "weathersit": 1,
...     "temperature_C": 9.84,
...     "feeling_temperature_C": 14.395,
...     "humidity": 81.0,
...     "windspeed": 0.0,
... })
1
```

## Development

To install a development version of the library:

```
$ flit install --symlink --extras=dev
```

To run the tests:

```
$ pytest
```

To measure the coverage:

```
$ pytest --cov=ie_bike_model
```

## Trivia

Total implementation time: **4 hours 30 minutes** 🏁
