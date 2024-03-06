# Importing necessary libraries
import numpy as np
import pandas as pd
import random

# Setting global variables
global blitzDistr_kwargs, regionMap_pair_kwargs
DEFAULT_DEV_TYPE = 2
INT_PAIR_INCREMENT = 1
blitzDistr_kwargs = {'dev_type': DEFAULT_DEV_TYPE, 'use_tend': False, 'return_type': 'conc distr'}
regionMap_pair_kwargs = {'pair_data': False, 'pair_increment': INT_PAIR_INCREMENT, 'return_type': 1,'distr_return': 'conc distr'}

# Methods to modify the libraries

def librModify(*libraries, **modifying_args):
  """
  Function to update the libraries with modifying arguments.
  """
  # Ensuring libraries is a tuple or list
  if not isinstance(libraries, (tuple, list)): libraries = libraries,
  net_libraries = libraries[0]
  for libr in libraries[1:]:
    net_libraries.update(libr)

  # Creating a copy of the libraries and updating it with modifying arguments
  modified_kwargs = net_libraries.copy()
  for key, value in modifying_args.items():
    key = key.lower()
    if key in net_libraries: modified_kwargs.update({key: value})
    else: raise TypeError (f'Invalid key |{key} = {value}|. Valid keys: default values ->{list(net_libraries.items())}')

  # Updating the libraries with the modified arguments
  for libr in libraries:
    [libr.update({key: modified_kwargs.get(key)}) for key in libr]
  return libraries if len(libraries) > 1 else libraries[0]

def librModify_copy(copy_libr = False, *libraries, **modifying_args):
  """
  Function to update the libraries with modifying arguments and return a copy if specified.
  """
  global var
  var = 5
  # Ensuring libraries is a tuple or list
  if not isinstance(libraries, (tuple, list)): libraries = libraries,
  net_libraries = libraries[0]
  for libr in libraries[1:]:
    net_libraries.update(libr)

  # Creating a copy of the libraries and updating it with modifying arguments
  modified_kwargs = net_libraries.copy()
  for key, value in modifying_args.items():
    key = key.lower()
    if key in net_libraries: modified_kwargs.update({key: value})
    else: raise TypeError (f'Invalid key |{key} = {value}|. Valid keys: default values ->{list(net_libraries.items())}')

  # Updating the libraries with the modified arguments and creating a copy if specified
  if copy_libr: libraries_copy = []
  for libr in libraries:
    if copy_libr:
      libr_copy = libr.copy()
      [libr_copy.update({key: modified_kwargs.get(key)}) for key in libr]
      libraries_copy.append(libr_copy)
    else: [libr.update({key: modified_kwargs.get(key)}) for key in libr]
  return libraries_copy if copy_libr else libraries

# Methods to analyze the distribution of data

def DistrFunc (data, **kwargs):
  """
  Function to calculate the distribution of data.
  """
  # Getting the arguments
  dev_type = kwargs.get('dev_type', 1)
  use_tend = kwargs.get('use_tend', True)
  return_type = kwargs.get('return_type', 'conc distr')

  # Calculating the distribution
  if use_tend:
    deviation = np.power(np.mean(np.abs(data[:, None] - data) ** dev_type, axis=1), 1/dev_type).min()
    min_indices = np.where(np.isclose(np.power(np.mean(np.abs(data[:, None] - data) ** dev_type, axis=1), 1/dev_type), deviation))
    tendency = np.unique(data[min_indices])
  else:
    tendency = np.mean(data)
    if dev_type == 1 : deviation = np.std(data)
    else:deviation = np.power(np.mean(np.abs(data - tendency) ** dev_type), 1/dev_type)

  # Creating the distribution and return library
  distribution = [tendency - deviation, tendency, tendency + deviation]
  distr_range = [tendency - deviation, tendency + deviation]
  concatenated_distribution = distribution
  conc_range = distr_range

  return_libr = {
    'dev': deviation, 
    'tend': tendency,
    'distr': distribution,
    'conc distr': concatenated_distribution,
    'conc tend': concatenated_distribution[1],
    'distr range': distr_range,
    'conc range': conc_range
  }
  if isinstance(return_type, tuple): 
    return [return_libr[key] for key in return_type]
  else: 
    return [return_libr[key] for key in [return_type]][0]

def blitzDistr (data, **kwargs):
  """
  Function to calculate the deviation, tendency, distribution, and range of given data.
  
  Parameters:
  data (list or np.array): The input data
  kwargs (dict): Additional arguments to modify the function's behavior
  
  Returns:
  list or value: Depending on the 'return_type' argument, it returns a list or a single value.
  """
  
  # Copy default arguments and update them with provided kwargs
  default_kwargs = blitzDistr_kwargs.copy()
  updated_default = librModify(default_kwargs, **kwargs)
  
  # Extract necessary parameters from updated arguments
  dev_type = updated_default['dev_type']
  use_tend = updated_default['use_tend']
  return_type = updated_default['return_type']

  # Convert data to numpy array if it's a list
  if isinstance(data, list): data = np.array(data)
  
  # Return None if data is empty
  if data.size == 0: 
    if isinstance(return_type, tuple):return [None for _ in return_type]
    else: return None
  
  # Flatten the data if it has more than one dimension
  if len(data.shape) > 1: data = np.concatenate(data)
  
  # Reshape data to 2D array
  data_array = np.array(data)[:, np.newaxis] 

  # Calculate deviation, tendency, distribution, and range based on 'use_tend' argument
  if use_tend:
    deviations = np.abs(data_array - data_array.T)**dev_type
    yield_deviation = np.mean(deviations, axis = 1)**(1/dev_type)
    min_deviation = np.min(yield_deviation)
    min_indices = np.where(yield_deviation == min_deviation)
    tendency = list(np.unique(data[min_indices[0]]))
    meanTend = np.mean(tendency) 
    distribution = [[tend - min_deviation, tend, tend + min_deviation] for tend in tendency]
    distr_range = [[tend - min_deviation, tend + min_deviation] for tend in tendency]
    concatenated_distribution = [meanTend - min_deviation, meanTend, meanTend + min_deviation]
    conc_range = [meanTend - min_deviation, meanTend + min_deviation]
    deviation = min_deviation
  else:
    tendency = np.mean(data)
    if dev_type == 1 : deviation = np.std(data)
    else:deviation = np.power(np.mean(np.abs(data - tendency) ** dev_type), 1/dev_type)

    distribution = [tendency - deviation, tendency, tendency + deviation]
    distr_range = [tendency - deviation, tendency + deviation]
    concatenated_distribution = distribution
    conc_range = distr_range
  
  # Prepare the return dictionary
  return_libr = {
  'dev': deviation, 
  'tend': tendency,
  'distr': distribution,
  'conc distr': concatenated_distribution,
  'conc tend': concatenated_distribution[1],
  'distr range': distr_range,
  'conc range': conc_range
  }
  
  # Return the requested values based on 'return_type' argument
  if isinstance(return_type, tuple): 
    return [return_libr[key] for key in return_type]
  else: 
    return [return_libr[key] for key in [return_type]][0]

def regionDensity_moveDistr(Forecast, **kwargs):
  """
  Function to calculate the moving distribution and density of a forecast.
  
  Parameters:
  Forecast (list or DataFrame): The forecast data
  kwargs (dict): Additional arguments to modify the function's behavior
  
  Returns:
  DataFrame or value: Depending on the 'return_type' argument, it returns a DataFrame or a single value.
  """
  
  # Define default and secondary arguments
  default_kwargs = {'return_type': (1,2)}
  secondary_kwargs = regionMap_pair_kwargs.copy()
  secondary_kwargs.pop('return_type')
  libr = default_kwargs.copy(), secondary_kwargs

  # Update the arguments with provided kwargs
  updated_default, secondary_kwargs = librModify(*libr, **kwargs)
  return_type = updated_default['return_type']
  
  # Convert forecast to DataFrame
  data_frame = pd.DataFrame(Forecast)
  
  # Apply regionMap_pair function to each column of the DataFrame
  yield_return = data_frame.apply(regionMap_pair, axis = 0, region_values = 'map distr', return_type = (7,3),**secondary_kwargs)
  
  # Define functions to calculate moving distribution and density
  moving_distr = lambda: pd.DataFrame([sub for sub in yield_return.iloc[0]])
  density_movingDistr = lambda: pd.DataFrame([sub for sub in yield_return.iloc[1]]).apply(np.mean, axis = 0)

  # Prepare the return list
  return_list = [moving_distr, density_movingDistr]
  
  # Return the requested values based on 'return_type' argument
  if  isinstance(return_type,tuple): 
    return [return_list[digit - 1]() for digit in return_type]
  else:  
    return [return_list[digit - 1]() for digit in [return_type]][0]

# Methods to analyse the difference begaviour of data
  
def diffFunction (data, **kwargs):
  """
  Function to calculate the differences of given data.
  
  Parameters:
  data (list or np.array): The input data
  kwargs (dict): Additional arguments to modify the function's behavior
  
  Returns:
  list or value: Depending on the 'return_type' argument, it returns a list or a single value.
  """
  
  # Copy default arguments and update them with provided kwargs
  default_kwargs = {'n_order': 1, 'axis': 1, 'return_type':0}
  updated_default = librModify(default_kwargs.copy(), **kwargs)
  
  # Extract necessary parameters from updated arguments
  n_order = updated_default['n_order']
  axis = updated_default['axis']
  return_type = updated_default['return_type']

  # Calculate differences based on the shape of the data
  data_shape = np.shape(data)
  if len(data) == 0: return print('Data is empty')
  elif len(data_shape) == 1: 
    differences =  np.diff(data, n_order)
  elif len(data_shape) == 2:
    differences =  np.diff(data, n_order, axis = axis)
  elif len(data) > 2: return print('Data shape not supported: must be less than 3D')

  # Calculate positive and negative values and their probabilities
  diffsize = differences.size
  Posvalues, Negvalues = differences[differences > 0], differences[differences < 0]
  Posprob, NegProb = Posvalues.size/diffsize, Negvalues.size/diffsize

  # Prepare the return dictionary
  return_libr = {0:differences, 1:Posvalues, 2:Negvalues, 'pos prob':Posprob, 'neg prob':NegProb}

  # Return the requested values based on 'return_type' argument
  if isinstance(return_type, tuple): 
    return [return_libr[key] for key in return_type]
  else: 
    return [return_libr[key] for key in [return_type]][0]

def regionChange (data, **kwargs):
  """
  Function to calculate the region change of given data.
  
  Parameters:
  data (list or np.array): The input data
  kwargs (dict): Additional arguments to modify the function's behavior
  
  Returns:
  tuple: Depending on the 'dynamic_change' argument, it returns a tuple of distributions or tendencies and a probability.
  """
  
  # Define default and secondary arguments
  DEFAULT_SET_PROB = 0.5
  default_kwargs = {'dynamic_change': True, 'use_prob': True, 'set_prob': DEFAULT_SET_PROB}
  secondary_kwargs = blitzDistr_kwargs.copy()
  secondary_kwargs.pop('return_type')

  # Update the arguments with provided kwargs
  libr = default_kwargs.copy(), secondary_kwargs
  updated_default, secondary_kwargs = librModify(*libr, **kwargs)  

  # Extract necessary parameters from updated arguments
  dynamic_change = updated_default['dynamic_change']
  use_prob = updated_default['use_prob']
  set_prob = updated_default['set_prob']
  
  # Calculate differences, distribution, region map, and probability
  differences = diffFunction(data, return_type = 0)
  diff_distr = blitzDistr(differences, return_type = 'conc distr', **secondary_kwargs)
  region_mapped, probability = regionMap_pair(differences , diff_distr, return_type = (1,2))
  
  # Calculate lower and upper regions and their distributions or tendencies
  lower_region = region_mapped[0]
  upper_region = region_mapped[-1]
  prob_upperRegion = probability[-1] if use_prob else set_prob
  distr_lowerRegion, tend_lowerRegion = blitzDistr(lower_region, return_type = ('conc range', 'conc tend'), **secondary_kwargs)
  distr_upperRegion, tend_upperRegion = blitzDistr(upper_region, return_type = ('conc range', 'conc tend'), **secondary_kwargs)
  
  # Return the requested values based on 'dynamic_change' argument
  if dynamic_change:
    return distr_lowerRegion, distr_upperRegion, prob_upperRegion
  else:
    return tend_lowerRegion, tend_upperRegion, prob_upperRegion

def posNegChange (data, **kwargs):
  """
  Function to calculate the positive and negative change of given data.
  
  Parameters:
  data (list or np.array): The input data
  kwargs (dict): Additional arguments to modify the function's behavior
  
  Returns:
  tuple: Depending on the 'dynamic_change' argument, it returns a tuple of distributions or tendencies and a probability.
  """
  
  # Define default and secondary arguments
  DEFAULT_SET_PROB = 0.5
  default_kwargs = {'dynamic_change': True, 'use_prob': True, 'set_prob': DEFAULT_SET_PROB}
  secondary_kwargs = blitzDistr_kwargs.copy()
  secondary_kwargs.pop('return_type')

  # Update the arguments with provided kwargs
  libr = default_kwargs.copy(), secondary_kwargs
  updated_default, secondary_kwargs = librModify(*libr, **kwargs)  

  # Extract necessary parameters from updated arguments
  dynamic_change = updated_default['dynamic_change']
  use_prob = updated_default['use_prob']
  set_prob = updated_default['set_prob']
  
  # Calculate positive and negative values and their probabilities, distributions or tendencies
  pos_val, neg_val, prob_pos = diffFunction(data, return_type = (1, 2, 'pos prob'))
  distr_pos, tend_pos = blitzDistr(pos_val, return_type = ('conc range', 'conc tend'), **secondary_kwargs)
  distr_neg, tend_neg = blitzDistr(neg_val, return_type = ('conc range', 'conc tend'), **secondary_kwargs)
  prob_pos = prob_pos if use_prob else set_prob

  # Return the requested values based on 'dynamic_change' argument
  if dynamic_change:
    return distr_neg, distr_pos, prob_pos
  else:
    return tend_neg, tend_pos, prob_pos

def genChange (method_data):
  """
  Function to generate a change based on the given method data.
  
  Parameters:
  method_data (list): The input method data. This is either the return from regionChange or posNegChange function.
  The third element of method_data is the return probability from the respective method used.
  
  Returns:
  float: The generated change.
  """
  
  # Extract the first and second method data
  methodData_1 = method_data[0]; methodData_2 = method_data[1]
  
  # If the first method data is None, return a random value from the second method data
  if methodData_1 is None: return methodData_2 if len(methodData_2) == 1 else random.uniform(*methodData_2)
  
  # If the second method data is None, return a random value from the first method data
  if methodData_2 is None: return methodData_1 if len(methodData_1) == 1 else random.uniform(*methodData_1)
  
  # Generate lower and upper changes based on the method data
  lower_change = random.uniform(*methodData_1) if isinstance(methodData_1, list) else methodData_1
  upper_change = random.uniform(*methodData_2) if isinstance(methodData_2, list) else methodData_2
  
  # Return the upper change if a random number is greater than the third element of method data (the return probability from the respective method used), else return the lower change
  return upper_change if random.random() > method_data[2] else lower_change

# Methods to modify data

def regionMap_pair(original_data, region_values=None, **kwargs):
  """
  Function to map the original data to the specified regions.

  Parameters:
  original_data (list or numpy array): The original data to be mapped.
  region_values (list or str, optional): The region values for mapping. If it's a string 'map distr', the distribution of original data will be used for mapping.
  **kwargs: Additional keyword arguments for the mapping process.

  Returns:
  list: The mapped data and related information based on the return type specified in kwargs.
  """
  # Copy the default keyword arguments and modify them based on the input kwargs
  default_kwargs = regionMap_pair_kwargs.copy()
  secondary_kwargs = blitzDistr_kwargs.copy()
  secondary_kwargs.pop('return_type')
  updated_default, secondary_kwargs = librModify(default_kwargs, secondary_kwargs, **kwargs)

  # Extract the necessary information from the updated kwargs
  pair_data = updated_default['pair_data']
  pair_increment = updated_default['pair_increment']
  return_type = updated_default['return_type']

  # Convert the original data to numpy array if it's a list
  if isinstance(original_data, list): original_data = np.array(original_data)

  # If region_values is a string 'map distr', use the distribution of original data for mapping
  if isinstance(region_values, str) and region_values.lower() == 'map distr':
    distr_return = updated_default['distr_return']
    region_values = blitzDistr(original_data, return_type=distr_return, **secondary_kwargs)

  # Initialize the mapped data
  n = len(region_values) - 1
  mapped_data = [[] for _ in range(n)]

  # Filter the original data based on the region values and map them to the regions
  filtered_data = original_data[(original_data >= min(region_values)) & (original_data <= max(region_values))]
  for val, nextval in zip(filtered_data, filtered_data[pair_increment:]):
    bin_value = np.digitize(val, region_values)
    if bin_value > n: bin_value = n
    if pair_data:
      mapped_data[bin_value - 1].append((val, nextval))
    else:
      mapped_data[bin_value - 1].append(val)

  # Calculate the size and probability of the mapped data
  Data_size = original_data.size
  net_MapData_size = len(mapped_data[0]) if len(mapped_data) == 1 else len(np.concatenate(mapped_data))
  mappedData_size = [len(sub_data) for sub_data in mapped_data]
  Total_prob = net_MapData_size / Data_size
  abs_prob = [len(sub_data) / Data_size for sub_data in mapped_data]
  relative_prob = [len(sub_data) / net_MapData_size for sub_data in mapped_data]

  # Prepare the return list based on the return type
  return_list = [mapped_data, relative_prob, abs_prob, Total_prob, mappedData_size, net_MapData_size, region_values]
  if isinstance(return_type, tuple):
    return [return_list[digit - 1] for digit in return_type]
  else:
    return [return_list[digit - 1] for digit in [return_type]][0]

def filterPair(data, **kwargs):
  """
  Function to filter and pair the data based on the specified conditions.

  Parameters:
  data (list or numpy array): The original data to be filtered and paired.
  **kwargs: Additional keyword arguments for the filtering and pairing process.

  Returns:
  numpy array: The filtered and paired data.
  """

  # Copy the default keyword arguments and modify them based on the input kwargs
  default_kwargs = {'order': None, 'filter_flag': False, 'pair': False, 'filterFunc': None, 'pair_index': 1, 'axis_index': 0}
  updated_default = librModify(default_kwargs.copy(), **kwargs)

  # Extract the necessary information from the updated kwargs
  order = updated_default['order']
  filter_flag = updated_default['filter_flag']
  pair_flag = updated_default['pair']
  filterFunc = updated_default['filterFunc']
  pair_index = updated_default['pair_index']
  axis_index = updated_default['axis_index']

  # Convert the data to numpy array if it's a list
  if isinstance(data, list): data = np.array(data)

  # If order is not None, enable the filter and pair flags
  if order is not None:
    filter_flag = True
    pair_flag = True

  # If pair_flag is True, pair the data
  if pair_flag:
    PairedData = np.transpose([data[:-pair_index], data[pair_index:]])
    if order == 0 and filterFunc is not None:
      Filtered_PairedData = PairedData[filterFunc(PairedData[:, axis_index])]
      return Filtered_PairedData
    else:
      return PairedData

  # If filter_flag is True, filter the data
  if filter_flag and filterFunc is not None:
    FilteredData = data[filterFunc(data)]
    if order == 1:
      Paired_FilteredData = np.transpose([FilteredData[:-pair_index], FilteredData[pair_index:]])
      return Paired_FilteredData
    else:
      return FilteredData

  return print('No filter or pair operation performed')

def boundSplitFunc(data, value, region_size):
  """
  Function to split the data based on the specified value and region size.

  Parameters:
  data (list or numpy array): The original data to be split.
  value (float): The value to be used for splitting.
  region_size (float): The size of the region for splitting.

  Returns:
  numpy array: The split data.
  """

  # Convert the data to numpy array if it's a list
  if isinstance(data, list): data = np.array(data)

  # Calculate the upper and lower bounds for splitting
  upper_bound = value + (abs(data.max() - value) / np.ptp(data)) * region_size
  lower_bound = value - (abs(data.min() - value) / np.ptp(data)) * region_size

  # Split the data based on the bounds
  current_data = data[(data >= lower_bound) & (data <= upper_bound)]

  return current_data

def uniformSplitFunc(data, val, region_size):
  """
  Function to split the data uniformly based on the specified value and region size.

  Parameters:
  data (list or numpy array): The original data to be split.
  val (float): The value to be used for splitting.
  region_size (float): The size of the region for splitting.

  Returns:
  tuple: The split data and the region values.
  """

  # Convert the data to numpy array if it's a list
  if isinstance(data, list): data = np.array(data)

  # Calculate the steps and region values for splitting
  steps = np.floor(np.ptp(data) / region_size) + 1
  region_values = np.linspace(data.min(), data.max(), int(steps))

  # Split the data based on the region values
  split_data = regionMap_pair(data, region_values, pair_data=True)

  # Find the region index for the specified value
  region_index = np.digitize(val, region_values)
  if region_index == len(region_values): region_index -= 1
  if region_index == 0: region_index += 1

  return split_data, region_values