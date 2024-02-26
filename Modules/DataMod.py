import numpy as np
import cProfile
import pandas as pd

def distribution(data, deviation_type,  tend_evaluation = True, return_type = 'conc distr'):
  
  if tend_evaluation:
    deviation = np.power(np.mean(np.abs(data[:, None] - data) ** deviation_type, axis=1), 1/deviation_type).min()
    min_indices = np.where(np.isclose(np.power(np.mean(np.abs(data[:, None] - data) ** deviation_type, axis=1), 1/deviation_type), deviation))
    tendency = np.unique(data[min_indices])
  else:
    tendency = np.mean(data)
    if deviation_type == 1 : deviation = np.std(data)
    else:deviation = np.power(np.mean(np.abs(data - tendency) ** deviation_type), 1/deviation_type)

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

def blitzDistr(data, deviation_type, tend_evaluation = True, return_type = 'conc distr'):
    
    if len(data.shape) > 1: np.concatenate(data)
    data_array = np.array(data)[:, np.newaxis] 

    if tend_evaluation:
      deviations = np.abs(data_array - data_array.T)**deviation_type
      yield_deviation = np.mean(deviations, axis=1)**(1/deviation_type)
      deviation = np.min(yield_deviation)
      min_indices = np.where(yield_deviation == deviation)
      tendency = list(np.unique(data[min_indices[0]]))
      minTend = np.min(tendency) 
      meanTend = np.mean(tendency) 
      maxTend = np.max(tendency)    
      distribution = [[tend - deviation, tend, tend + deviation] for tend in tendency]
      distr_range = [[tend - deviation, tend + deviation] for tend in tendency]
      concatenated_distribution = [minTend - deviation, meanTend, maxTend + deviation]
      conc_range = [minTend - deviation, maxTend + deviation]
    
    else:
      tendency = np.mean(data)
      if deviation_type == 1 : deviation = np.std(data)
      else:deviation = np.power(np.mean(np.abs(data - tendency) ** deviation_type), 1/deviation_type)

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

def region_MapPair(original_data, region_values, pair_data = False, pair_increment = 1, return_type = (4)):

  if isinstance(region_values, list): original_data = np.array(original_data)
  n = len(region_values) - 1
  mapped_data = [[] for _ in range(n)]

  filtered_data = original_data[(original_data >= min(region_values)) & (original_data <= max(region_values))]

  if not pair_data: pair_increment = 0
  for val, nextval in zip(filtered_data, filtered_data[pair_increment:]):
    bin_value = np.digitize(val, region_values)
    if bin_value > n: bin_value = n
    if pair_data and val + pair_increment < filtered_data.size:
      mapped_data[bin_value - 1].append((val, nextval))
    else: mapped_data[bin_value - 1].append(val)

  Data_size = original_data.size
  mappedData_size = filtered_data.size

  Total_prob = mappedData_size / Data_size
  abs_prob = [len(sub_data) / Data_size for sub_data in mapped_data]
  relative_prob = [len(sub_data) / mappedData_size for sub_data in mapped_data]

  return_list = [abs_prob, relative_prob, Total_prob, mapped_data]

  if  isinstance(return_type,tuple): 
    return [return_list[digit - 1] for digit in return_type]
  else:  
    return [return_list[digit - 1] for digit in [return_type]][0]

def filterPair(data, **kwargs):
  order = kwargs.get('order', None)
  filter_flag = kwargs.get('filter_flag', False)
  pair_flag = kwargs.get('pair', False)
  filterFunc = kwargs.get('filterFunc', None)
  pair_index = kwargs.get('pair_index', 1)
  axis_index = kwargs.get('axis_index', 0)

  if order is not None: 
    filter_flag = True 
    pair_flag = True

  if pair_flag:
    PairedData = np.transpose([data[:-pair_index],data[pair_index:]])
    if order == 0 and filterFunc is not None:
      Filtered_PairedData = PairedData[filterFunc(PairedData[:,axis_index])]
      return Filtered_PairedData
    else: 
      return PairedData

  if filter_flag and filterFunc is not None:
    FilteredData = data[filterFunc(data)]
    if order == 1:
      Paired_FilteredData = np.transpose([FilteredData[:-pair_index], FilteredData[pair_index:]])
      return  Paired_FilteredData
    else: 
      return FilteredData

  return print('No filter or pair operation performed') 

def diffFunction(data, **kwargs):
  
  n_order = kwargs.get('n_order', 1)
  return_type = kwargs.get('return_type', (0, 1, 2, 'pos prob', 'neg prob'))

  if isinstance(data, list): data = np.array(data)
  if len(data.shape) < 2:   
    differences =  np.diff(data, n_order, axis = 0)
  elif data.shape[1] < 3:
    differences = np.diff(data, n_order, axis = 1)
  else: raise ValueError('Data must be of shape (a,b) where b < 3')
  
  diffsize = differences.size
  Posvalues, Negvalues = differences[differences > 0], differences[differences < 0]
  Posprob, NegProb = Posvalues.size/diffsize, Negvalues.size/diffsize
  return_libr = {0:differences, 1:Posvalues, 2:Negvalues, 'pos prob':Posprob, 'neg prob':NegProb}

  if isinstance(return_type, tuple): 
    return [return_libr[key] for key in return_type]
  else: 
    return [return_libr[key] for key in [return_type]][0]


if __name__== '__main__':
  Path = r'EURAUD.ifx_M1_202402190000_202402191016.csv'
  data = pd.read_csv(Path, sep = '\t')['<CLOSE>'].dropna().to_numpy()
  
  print(f'Data size is {data.size} with max {data.max()} and min {data.min()}')

 



