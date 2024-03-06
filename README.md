# Forecast Generator Repository README

## Overview

The Forecast generator Repository is a Python written repository designed for generating forecasts based on a given dataset and initial value. 
This class utilizes various methods and splitting strategies to produce accurate forecasts. Below is a guide to help you understand the class and its methods.

## Installation

Before using the ForecastClass, ensure you have the necessary libraries installed:

```bash
pip install numpy pandas
```

Now you're ready to integrate and leverage the forecasting capabilities.

## Quick Start

### Import Necessary Modules

```python
import numpy as np
import pandas as pd
import DataMod
from ForecastClass import ForecastClass
```

### Set Global Variables

Initialize data and initial value:

```python
data = pd.Series(np.random.randn(1000))
initial_val = 0
```

### Create an Instance of ForecastClass

```python
forecast = ForecastClass(data, initial_val)
```

### Set the Method for Generating Forecast

Choose a method for generating the forecast. If none is provided, it defaults to the `regionChange` method from the `DataMod` module:

```python
forecast.MethodUsed(DataMod.regionChange, use_tend=True, dynamic_change=True)
```

### Split the Data into Regions (Bound Splitting)

```python
forecast.BoundSplit(10)
```

### Generate the Forecast

```python
forecast.GenerateForecast(10, 10)
```

## Methods

### `init(self, data, initial_val)`

The constructor for the ForecastClass. It initializes the class with the given data and initial value.

### `MethodUsed(self, method_used=None, **kwargs)`

Sets the method to be used for generating the forecast. If no method is provided, it defaults to the `regionChange` method from the `DataMod` module.

### `BoundSplit(self, region_size, recursive_split=True)`

Splits the data into regions bound (centered) on a given value. If both `BoundSplit` and `UniformSplit` methods are called, `BoundSplit` takes precedence.

### `UniformSplit(self, region_size, recursive_split=True)`

Splits the data uniformly. If both `BoundSplit` and `UniformSplit` methods are called, `UniformSplit` takes precedence.

### `GenerateForecast(self, num, period)`

Generates a forecast based on the method used, the data, and the initial value. It returns a numpy array of the forecasted values.

## Usage

This example generates a 10x10 numpy array of forecasted values based on the `regionChange` method from the `DataMod` module.

```python
# Import necessary modules
import numpy as np
import pandas as pd
import DataMod
from ForecastClass import ForecastClass

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
```

## Examples

Explore detailed examples and usage scenarios in the module's functions and documentation. Effortlessly adapt these examples to suit your specific data analysis and modification needs. For more information, refer to the documentation in the respective modules.
