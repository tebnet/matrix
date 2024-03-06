# ForecastClass README

## Overview
The `ForecastClass` is a Python class designed for generating forecasts based on financial time series data. It provides methods for splitting the data into regions, selecting forecasting methods, and generating forecasts over a specified period.

## Dependencies
- `numpy` (imported as `np`): A powerful library for numerical operations.
- `pandas` (imported as `pd`): A data manipulation library that provides data structures for efficient data analysis.
- `logging`: A built-in Python module for logging messages.
- `Modules.DataMod` (imported as `DataMod`): A custom module containing functions for data manipulation.

## Class Attributes
- `DEFAULT_DEV_TYPE`: Default development type used in the forecast.
- `DEFAULT_SET_PROB`: Default set probability used in the forecast.

## Class Methods

### `__init__(self, data, initial_val)`
- Initializes the `ForecastClass` object with input time series data and an initial value.

### `MethodUsed(self, method_used=None, **kwargs)`
- Sets the method to be used for generating the forecast, along with optional method-specific arguments.

### `BoundSplit(self, region_size, recursive_split=True)`
- Splits the data into regions bound(centred) on a given value, using the specified region size. Optionally, the split can be recursive.

### `UniformSplit(self, region_size, recursive_split=True)`
- Splits the data uniformly using the specified region size. Optionally, the split can be recursive.

### `GenerateForecast(self, num, period)`
- Generates a forecast based on the selected method, the data, and the initial value. The number of forecasts (`num`) and the forecast period (`period`) are specified as parameters.

## Usage

```python
# Example usage of the ForecastClass

# Import necessary libraries
import numpy as np
import pandas as pd
import logging
import Modules.DataMod as DataMod
import matplotlib.pyplot as plt

# Set logging level to INFO
logging.basicConfig(level=logging.INFO)

# Define the path to the data file
FILE_PATH = r'EURAUD.ifx.csv'

# Read the data from the file and drop any missing values
data = pd.read_csv(FILE_PATH, sep='\t')['<CLOSE>'].dropna()

# Get the initial value from the data
initial_val = data.iloc[-1] if isinstance(data, pd.Series) else data[-1]

# Calculate the region size
region_size = (1/10) * np.ptp(data)

# Define the number of forecasts and the period
num = 10
period = 10

# Create an instance of the ForecastClass
instance = ForecastClass(data, initial_val)

# Split the data based on the bounds
instance.BoundSplit(region_size)

# Generate the forecast
Forecast = instance.GenerateForecast(num, period)

# Convert the forecast to a DataFrame
frame = pd.DataFrame(Forecast)

# Plot the forecast
plt.plot(Forecast.T, color='grey')
plt.show()
```

This example demonstrates how to use the `ForecastClass` to generate forecasts based on financial time series data. Customize the input parameters, such as the file path, number of forecasts, and forecast period, according to your specific requirements.
