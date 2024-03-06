ForecastClass and DataMod Modules
Overview
This repository contains two main components: the ForecastClass and the DataMod modules.

ForecastClass
The ForecastClass is a Python class designed for generating forecasts from a given dataset and an initial value. It provides a flexible and customizable way to create forecasts using different methods and splitting strategies.

The class includes the following methods:

__init__(self, data, initial_val): This is the constructor for the ForecastClass. It initializes the class with the given data and initial value.

MethodUsed(self, method_used = None, **kwargs): This method sets the method to be used for generating the forecast. If no method is provided, it defaults to the regionChange method from the DataMod module.

BoundSplit(self, region_size, recursive_split = True): This method splits the data into regions bound (centered) on a given value. If both BoundSplit and UniformSplit methods are called, BoundSplit takes precedence.

UniformSplit(self, region_size, recursive_split = True): This method splits the data uniformly. If both BoundSplit and UniformSplit methods are called, UniformSplit takes precedence.

GenerateForecast(self, num, period): This method generates a forecast based on the method used, the data, and the initial value. It returns a numpy array of the forecasted values.

DataMod Module
The DataMod module is a versatile tool that offers functionalities for comprehensive data analysis and modification. It provides a range of functions for exploring distributions, mapping regions, transforming data, and more.

The module includes the following functions:

blitzDistr(data, dev_type=2, use_tend=True, return_type='conc distr'): This function calculates the distribution of data.

diffFunction(data, n_order=1, axis=1, return_type='pos prob'): This function calculates differences of given data.

regionChange(data, dynamic_change=True, use_prob=True, set_prob=0.5): This function calculates region changes of given data.

genChange(method_data): This function generates a change based on method data.

regionMap_pair(original_data, region_values=None, pair_data=False, pair_increment=1, return_type=1): This function maps original data to specified regions.

filterPair(data, order=None, filter_flag=False, pair=False, filterFunc=None, pair_index=1, axis_index=0): This function filters and pairs the data based on specified conditions.

boundSplitFunc(data, value, region_size): This function splits the data based on specified value and region size.

uniformSplitFunc(data, val, region_size): This function splits the data uniformly based on specified value and region size.

Installation
First, ensure you have the necessary libraries installed:

Usage
Detailed usage instructions for both the ForecastClass and DataMod module are provided in the Quick Start section of this README. Examples are also provided to help you understand how to use these tools in your own projects.
