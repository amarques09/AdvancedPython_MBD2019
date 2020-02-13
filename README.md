Amanda Marques

# Bike sharing prediction model

	#how to build an app from scratch:


# 1.Create a repository in github
# 2.Have a reference dataset 
# 3.Have predefined functions on how to preprocess the data and how to train it (different models
	#For instance: create a new Function:
		    #def train_ridge(hour):
		    # Avoid modifying the original dataset at the cost of RAM
		    #hour = hour.copy()
		
		    #hour_d = pd.get_dummies(hour)
		    #regex = re.compile(r"\[|\]|<", re.IGNORECASE)
		    #hour_d.columns = [
		    #regex.sub("_", col) if any(x in str(col) for x in set(("[", "]", "<"))) else col
			#for col in hour_d.columns.values
		    #]
		
		    #hour_d = hour_d.select_dtypes(exclude="category")
		
		    #hour_d_train_x, _, hour_d_train_y, _, = split_train_test(hour_d)
		
		    #ridge = Ridge(alpha=1.0)
		
                    #ridge.fit(hour_d_train_x, hour_d_train_y)
                    #return ridge

# 4.Modify the predict function:
	def predict(parameters, model_dir=None, model = 'xgboost'):
	    """Returns model prediction.
	
	    """
	    model_path = get_model_path(model_dir)
	    if not os.path.exists(model_path):
	        train_and_persist(model_dir=model_dir, model = model)

	    model = joblib.load(model_path)

	    input_dict = get_input_dict(parameters)
	    X_input = pd.DataFrame([pd.Series(input_dict)])
	
	    result = model.predict(X_input)
	
	    # Undo np.sqrt(hour["cnt"])
	    return int(result ** 2)
	
# 5.Go to app.py and modify the result output and call the data - so we are able to do the mean of each column:
    For instance:

		  @app.route("/predict")
            def get_predict():
    
    df = read_data()
    
    weather_avg = df['weathersit'].mean()

    start_prediction = dt.datetime.now()
    result = predict(parameters,model = parameters.get('model'))
    end_prediction_time = dt.datetime.now() - start_prediction
    
    return {"result": result, "date": parameters['date'], "computation time/seconds": end_prediction_time.total_seconds()


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
