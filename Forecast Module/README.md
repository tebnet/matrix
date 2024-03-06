ForecastClass
The ForecastClass is a Python class that provides methods for generating forecasts based on a given dataset and initial value. The class uses different methods and splitting strategies to generate the forecast.

Methods
__init__(self, data, initial_val)
The constructor for the ForecastClass. It initializes the class with the given data and initial value.

MethodUsed(self, method_used = None, **kwargs)
This method sets the method to be used for generating the forecast. If no method is provided, it defaults to the regionChange method from the DataMod module.

BoundSplit(self, region_size, recursive_split = True)
This method splits the data into regions bound (centered) on a given value. If both BoundSplit and UniformSplit methods are called, BoundSplit takes precedence.

UniformSplit(self, region_size, recursive_split = True)
This method splits the data uniformly. If both BoundSplit and UniformSplit methods are called, UniformSplit takes precedence.

GenerateForecast(self, num, period)
This method generates a forecast based on the method used, the data, and the initial value. It returns a numpy array of the forecasted values.

Usage
This will generate a 10x10 numpy array of forecasted values based on the regionChange method from the DataMod module.

# Import necessary modules
import numpy as np 
import pandas as pd
import DataMod

# Initialize data and initial value
data = pd.Series(np.random.randn(1000))
initial_val = 0

# Create an instance of ForecastClass
forecast = ForecastClass(data, initial_val)

# Set the method to be used for generating the forecast
forecast.MethodUsed(DataMod.regionChange, use_tend=True, dynamic_change=True)

# Split the data into regions bound on a given value
forecast.BoundSplit(10)

# Generate the forecast
forecast.GenerateForecast(10, 10)
