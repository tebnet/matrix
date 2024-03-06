import numpy as np
import pandas as pd
import Modules.DataMod as DataMod 
import Forecast_Class as FC
import matplotlib.pyplot as plt

# Define the path to the data file
FILE_PATH = r'/workspaces/vonNuemann/Forecast Generator/Class/EURAUD.ifx.csv'

# Read the data from the file and drop any missing values
data = pd.read_csv(FILE_PATH, sep = '\t')['<CLOSE>'].dropna()
data_size = data.size
initial_val = data.iloc[-1] if isinstance(data, pd.Series) else data[-1]
region_size = (1/3)*np.ptp(data)

# Define the number of forecasts and the period
num = 100
period = data_size

# Create an instance of the ForecastClass
instance = FC.ForecastClass(data, initial_val)
# Split the data uniformly
instance.UniformSplit(region_size)
# Generate the forecast
Forecast = instance.GenerateForecast(num, period)

# Calculate the moving distribution and regional density
moving_distr, regional_density = DataMod.regionDensity_moveDistr(Forecast)

# Get the lower and upper densities
lower_density = regional_density[0]
upper_density = regional_density[1]

# Calculate the gross density
Gross_density = lower_density + upper_density

# Calculate the mean, max and min of the moving distribution
mean_movingDistr_tend = moving_distr[1].mean()
max_movingDistr = moving_distr.max().max()
min_movingDistr = moving_distr.min().min()

# Create an empty plot
empty_plt = np.arange(0)

# Define the range for the x-axis
from_ = data_size; to_ = from_ + period
x_range = np.arange(from_, to_)

# Set the title of the plot
plt.title(f'Density of Moving distrbution is {np.floor(Gross_density*100)}% [lower bound({np.floor(lower_density*100)} %): upper bound ({np.floor(upper_density*100)}%)]')

# Plot the forecast
color = 'grey'; style = '--'; width = 0.3; alpha = 0.3
plt.plot(x_range, Forecast.T, color = color,linestyle = style, linewidth = width, alpha = alpha)
plt.plot(empty_plt, label = 'Forecasts generated', color = color,linestyle = style, linewidth = 1)

# Plot the original data and the moving distribution
color = 'black'; color_2 = 'blue'; style = '-'; width = 0.8; alpha = 0.8
plt.plot(data, color = color, linestyle = style, linewidth = width, alpha = alpha, label = 'Original Data')
plt.plot(x_range, moving_distr, color = color_2, linestyle = style, linewidth = width, alpha = alpha)
plt.plot(empty_plt, label = 'Moving Distribution', color = color_2,  linestyle = style, linewidth = 1, alpha = 1)

# Plot the max, mean and min of the moving distribution
color = 'blue'; style = '--'; width = 0.8; alpha = 0.8
labels = ['Max', 'Mean tendency', 'Min']
H_lines = [max_movingDistr, mean_movingDistr_tend, min_movingDistr]
for line, label in zip(H_lines, labels):
  plt.axhline(y = line, color = color, linestyle = style, linewidth = width, alpha = alpha, label = f'{np.round(line, 3)} {label} of moving Distribution')

# Add a legend to the plot
plt.legend()

# Display the plot
plt.show()