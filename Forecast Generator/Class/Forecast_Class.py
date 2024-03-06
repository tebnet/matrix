import numpy as np 
import pandas as pd
import logging
import Modules.DataMod as DataMod
import matplotlib.pyplot as plt

# Set logging level to INFO
logging.basicConfig(level=logging.INFO)

class ForecastClass():
  DEFAULT_DEV_TYPE = 2
  DEFAULT_SET_PROB = 0.5
  
  def __init__(self, data, initial_val):
    # Initialize flags and data
    self.flag = 1
    self.flag2 = 1
    self.flag3 = 1
    self.run_flag = 1
    self.data = data
    self.initial_val = initial_val
    self.MethodUsed_called = False
    self.BoundSplit_called = False
    self.UniformSplit_called = False
    # Initialize methods and arguments
    self.methods = [DataMod.regionChange, DataMod.posNegChange]
    self.method_name = [meth.__name__ for meth in self.methods]
    self.method_used = DataMod.regionChange
    self.method_args = {
    'use_tend' : False,
    'dynamic_change' : True,
    'use_prob' : True,
    'set_prob' : self.DEFAULT_SET_PROB,
    'dev_type' : self.DEFAULT_DEV_TYPE
    }

  def MethodUsed (self, method_used = None, **kwargs):
    """Set the method to be used for generating the forecast."""
    self.MethodUsed_called = True
    if method_used is None: method_used = self.method_used
    if method_used not in self.methods: raise TypeError (f'Invalid "MethodUsed" argument. Provide Valid method argument -> {self.method_name} from DataMod module')
    self.method_used = method_used  
    self.method_args.update(kwargs)

  def BoundSplit (self, region_size, recursive_split = True):
    """Split the data into regions bound(centred) on a given value."""
    self.BoundSplit_called = True ; logging.info(f'Bound Split Method called')
    if self.UniformSplit_called and self.BoundSplit_called:
      self.UniformSplit_called = False; logging.info(f'BOUND METHOD OVERWRITES UNIFORM METHOD')
    data = self.data
    initial_val = self.initial_val
    self.region_size = region_size
    self.recursive_split = recursive_split
    if initial_val < data.min() or initial_val > data.max():      
      initial_val = data.max() if initial_val > data.max() else data.min() 
    self.current_data = DataMod.boundSplitFunc(data, initial_val, region_size)
    self.method_return = self.method_used(self.current_data, **self.method_args)  

  def UniformSplit (self, region_size, recursive_split = True):
    """Split the data uniformly."""
    self.UniformSplit_called = True ; logging.info(f'Uniform Split Method called')
    if self.BoundSplit_called and self.UniformSplit_called: 
      self.BoundSplit_called = False; logging.info(f'UNIFORM NETHOD OVERWRITES BOUND METHOD')
    data = self.data
    initial_val = self.initial_val
    self.region_size = region_size
    self.recursive_split = recursive_split
    data_regions, region_values = DataMod.uniformSplitFunc(data, initial_val, region_size)
    self.region_values = region_values
    self.net_process = [self.method_used(data_regions[i], **self.method_args) for i in range(len(data_regions))]
    region_index = np.digitize(initial_val, region_values)
    self.region_index = region_index
    if region_index == len(region_values): region_index -= 1
    if region_index == 0: region_index += 1
    self.method_return = self.net_process[region_index - 1]
  
  def GenerateForecast (self, num, period):
    """Generate a forecast based on the method used, the data, and the initial value."""
    if self.flag3 == 1:
      logging.info(f'Change Function used: {self.method_used.__name__}'); self.flag3 += 1
    
    Forecast = np.zeros((num, period))
    for index1 in np.arange(num):
      Forecast[index1, 0] = self.initial_val
      current_val = self.initial_val
      for index2 in np.arange(1, period):
       
        if self.BoundSplit_called and self.recursive_split:
          # Method called flags
          if self.flag == 1: logging.info(f'Bound Recursion Split') ; self.flag+= 1; 
          data = self.data
          current_data = self.current_data
          region_size = self.region_size
          if current_val < data.min() or current_val > data.max():      
            current_val = data.max() if current_val > data.max() else data.min() 
            current_data = DataMod.boundSplitFunc(data, current_val, region_size)
          elif current_val < current_data.min() or current_val > current_data.max():
            current_data = DataMod.boundSplitFunc(data, current_val,region_size)
            self.current_data = current_data
          self.method_return = self.method_used(self.current_data, **self.method_args)
        
        elif self.UniformSplit_called and self.recursive_split:
          # Method called flags
          if self.flag2 == 1: logging.info(f'Uniform Recursion Split') ; self.flag2 += 1
          region_values = self.region_values
          region_index = self.region_index
          current_regionIndex = np.digitize(current_val, region_values)
          if current_regionIndex != region_index:
            if current_regionIndex == len(region_values): current_regionIndex -= 1
            if current_regionIndex == 0: current_regionIndex += 1
            self.region_index = current_regionIndex
          self.method_return = self.net_process[self.region_index - 1]
        
        else:
          if (not self.BoundSplit_called and not self.UniformSplit_called) and self.run_flag == 1:
              self.method_return = self.method_used(self.data, **self.method_args)
              logging.info(f'No Split Method called')        
          self.run_flag += 1
        
        current_val += DataMod.genChange(self.method_return)
        Forecast[index1, index2] = current_val
    return Forecast

if __name__ == '__main__':
  # This is a test for the ForecastClass
  
  # Define the path to the data file
  FILE_PATH = r'C:\Users\tebne\OneDrive\Programming\Languages\Python\TebnetGithub\Class\EURAUD.ifx.csv'
  
  # Read the data from the file and drop any missing values
  data = pd.read_csv(FILE_PATH, sep = '\t')['<CLOSE>'].dropna()
  
  # Get the initial value from the data
  initial_val = data.iloc[-1] if isinstance(data, pd.Series) else data[-1]
  # Calculate the region size
  region_size = (1/10)*np.ptp(data)
  
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
  plt.plot(Forecast.T, color = 'grey')
  plt.show()