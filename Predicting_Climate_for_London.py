# %% [markdown]
# ![tower_bridge](tower_bridge.jpg)
# 
# As the climate changes, predicting the weather becomes ever more important for businesses. You have been asked to support on a machine learning project with the aim of building a pipeline to predict the climate in London, England. Specifically, the model should predict mean temperature in degrees Celsius (°C).
# 
# Since the weather depends on a lot of different factors, you will want to run a lot of experiments to determine what the best approach is to predict the weather. In this project, you will run experiments for different regression models predicting the mean temperature, using a combination of `sklearn` and `mlflow`.
# 
# You will be working with data stored in `london_weather.csv`, which contains the following columns:
# - **date** - recorded date of measurement - (**int**)
# - **cloud_cover** - cloud cover measurement in oktas - (**float**)
# - **sunshine** - sunshine measurement in hours (hrs) - (**float**)
# - **global_radiation** - irradiance measurement in Watt per square meter (W/m2) - (**float**)
# - **max_temp** - maximum temperature recorded in degrees Celsius (°C) - (**float**)
# - **mean_temp** - **target** mean temperature in degrees Celsius (°C) - (**float**)
# - **min_temp** - minimum temperature recorded in degrees Celsius (°C) - (**float**)
# - **precipitation** - precipitation measurement in millimeters (mm) - (**float**)
# - **pressure** - pressure measurement in Pascals (Pa) - (**float**)
# - **snow_depth** - snow depth measurement in centimeters (cm) - (**float**)

# %%
# Run this cell to import the n ules you require
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Read in the data
weather = pd.read_csv("london_weather.csv",parse_dates=['date'])

# Start coding here
# Use as many cells as you like

# %%
weather

# %%
#number of null values in each column
weather.isna().sum()

# %%
threshold = int(len(weather)*0.05)

# %%
cols_to_drop = weather.columns[weather.isna().sum()<= threshold]
weather.dropna(subset= cols_to_drop,inplace=True)

# %%
weather.isna().sum()

# %%
weather.snow_depth.nunique()

# %%
cols_with_missing_values = weather.columns[weather.isna().sum()>0]
for col in  cols_with_missing_values:
    weather[col] = weather[col].fillna(weather[col].mode()[0])

# %%
weather=weather.drop('date',axis=1)

# %%
features=weather.drop('mean_temp',axis=1).values
target = weather['mean_temp'].values

# %%
X_train , X_test, Y_train, Y_test = train_test_split(features,target,test_size=0.3,random_state=42)
Models = {"LinearRegression":LinearRegression(),"DecisonTreeRegressor":DecisionTreeRegressor(random_state=42),"RandomForestRegressor":RandomForestRegressor(random_state=42)}
MSE_values=[]
mlflow.set_experiment("Predicting London Weather")
for key,i in Models.items():
    mlflow.start_run()
    a = i
    a.fit(X_train,Y_train)
    y_predict = a.predict(X_test)
    MSE_values.append(np.round(mean_squared_error(Y_test,y_predict),2))
    mlflow.sklearn.log_model(i,key)
    mlflow.log_metric('rmse',np.round(mean_squared_error(Y_test,y_predict),2))
    mlflow.end_run()
MSE_values

# %%
experiment_name = "Predicting London Weather"
experiment = mlflow.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

# Search all runs in the experiment
experiment_results = mlflow.search_runs(experiment_ids=experiment_id)

# Convert results to a DataFrame for easier handling (optional)
experiment_results_df = pd.DataFrame(experiment_results)

print(experiment_results_df.head())


