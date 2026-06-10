# -*- coding: utf-8 -*-
"""
A sample code provided for loading ISU-ILCC battery aging data, available at
https://doi.org/10.25380/iastate.22582234

This code contains three major parts:
    1. Load raw data from JSON files and convert to dictionaries
    2. Derive capacity fade data (time vs. remaining capacity)
    3. Interpolation of discharge QV curves from C/5 RPT

This code is built on:
python - 3.7.13 (Python 3.9.9 has been verified to be compatible)
numpy - 1.21.5
scipy - 1.7.3
pandas - 1.3.5

Note:
An interpolation warning is triggered on line 164 for several cells when assigned smoothing factor ("precision_C_5") is too low. 
Without intervention, this would result in an array of NaN for the variable "tck". To resolve this problem, 
lines 165-167 loops to correct the smoothing factor until the variable "tck" is no longer an array of NaN 
to ensure correct output. Therefore, you may ignore the warning.

@author: Tingkai Li

"""

import json
import numpy as np
import pandas as pd
from scipy import interpolate

# Define a function to load RPT data and convert the data type
def convert_RPT_to_dict(cell,subfolder):
    with open('RPT_json/{}/{}.json'.format(subfolder,cell),'r') as file:
        data_dict = json.loads(json.load(file))
    
    # Convert time series data from string to np.datetime64
    for iii, start_time in enumerate(data_dict['start_stop_time']['start']):
        if start_time != '[]':
            data_dict['start_stop_time']['start'][iii] = np.datetime64(start_time)
            data_dict['start_stop_time']['stop'][iii] = np.datetime64(data_dict['start_stop_time']['stop'][iii])
        else:
            data_dict['start_stop_time']['start'][iii] = []
            data_dict['start_stop_time']['stop'][iii] = []

    for iii in range(len(data_dict['start_stop_time']['start'])):
        data_dict['QV_charge_C_2']['t'][iii] = list(map(np.datetime64,data_dict['QV_charge_C_2']['t'][iii]))
        data_dict['QV_discharge_C_2']['t'][iii] = list(map(np.datetime64,data_dict['QV_discharge_C_2']['t'][iii]))
        data_dict['QV_charge_C_5']['t'][iii] = list(map(np.datetime64,data_dict['QV_charge_C_5']['t'][iii]))
        data_dict['QV_discharge_C_5']['t'][iii] = list(map(np.datetime64,data_dict['QV_discharge_C_5']['t'][iii]))
    
    return data_dict
# Define a function to load cycling data and convert the data type
def convert_cycling_to_dict(cell,subfolder):
    with open('Cycling_json/{}/{}.json'.format(subfolder,cell),'r') as file:
        data_dict = json.loads(json.load(file))
        
    # Convert time series data from string to np.datetime64
    for iii, start_time in enumerate(data_dict['start_stop_time']['start']):
        if start_time != '[]':
            data_dict['start_stop_time']['start'][iii] = np.datetime64(start_time)
            data_dict['start_stop_time']['stop'][iii] = np.datetime64(data_dict['start_stop_time']['stop'][iii])
        else:
            data_dict['start_stop_time']['start'][iii] = []
            data_dict['start_stop_time']['stop'][iii] = []
            
    for iii, start_time in enumerate(data_dict['time_series_charge']['start']):
        if start_time != '[]':
            data_dict['time_series_charge']['start'][iii] = np.datetime64(start_time)
            data_dict['time_series_charge']['stop'][iii] = np.datetime64(data_dict['time_series_charge']['stop'][iii])
            data_dict['time_series_discharge']['start'][iii] = np.datetime64(data_dict['time_series_discharge']['start'][iii])
            data_dict['time_series_discharge']['stop'][iii] = np.datetime64(data_dict['time_series_discharge']['stop'][iii])            
        else:
            data_dict['time_series_charge']['start'][iii] = []
            data_dict['time_series_charge']['stop'][iii] = []
            data_dict['time_series_discharge']['start'][iii] = []
            data_dict['time_series_discharge']['stop'][iii] = []
    
    for iii in range(len(data_dict['time_series_charge']['start'])):
        data_dict['QV_charge']['t'][iii] = list(map(np.datetime64,data_dict['QV_charge']['t'][iii]))
        data_dict['QV_discharge']['t'][iii] = list(map(np.datetime64,data_dict['QV_discharge']['t'][iii]))
    
    return data_dict
# Define a function to calculate capacity vs time
def capacity_time_calculation(RPT_dict,cycling_dict):
    char_time_sum_week = []
    cycling_time_sum_week = []
    
    for i,time_start in enumerate(RPT_dict['start_stop_time']['start']):
        if type(time_start) != list:
            time_start = np.datetime64(time_start)
            time_stop =  np.datetime64(RPT_dict['start_stop_time']['stop'][i])
            t = (time_stop-time_start)/np.timedelta64(1, 'W')
            char_time_sum_week.append(t)
        else:
            char_time_sum_week.append(np.nan)
        
    char_time_sum_week = np.array(char_time_sum_week)
        
    for i,data in enumerate(char_time_sum_week):
        if np.isnan(data)==True:
            if np.isnan(char_time_sum_week[i-1])==False and np.isnan(char_time_sum_week[i+1])==False:
                char_time_sum_week[i] = (char_time_sum_week[i-1]+char_time_sum_week[i+1])/2
    
            elif np.isnan(char_time_sum_week[i-1])==False and np.isnan(char_time_sum_week[i+2])==False:
                char_time_sum_week[i] = (char_time_sum_week[i-1]+char_time_sum_week[i+2])/2
    
            elif np.isnan(char_time_sum_week[i-2])==False and np.isnan(char_time_sum_week[i+1])==False:
                char_time_sum_week[i] = (char_time_sum_week[i-2]+char_time_sum_week[i+1])/2
    
    for i,time_start in enumerate(cycling_dict['start_stop_time']['start']):
        if type(time_start) != list:
            time_start = np.datetime64(time_start)
            time_stop =  np.datetime64(cycling_dict['start_stop_time']['stop'][i])
            t = (time_stop-time_start)/np.timedelta64(1, 'W')
            cycling_time_sum_week.append(t)
        else:
            cycling_time_sum_week.append(np.nan)
    
    cycling_time_sum_week = np.array(cycling_time_sum_week)    
    for i,data in enumerate(cycling_time_sum_week):
        if np.isnan(data)==True:
            if np.isnan(char_time_sum_week[i-1])==False and np.isnan(char_time_sum_week[i+1])==False:
                cycling_time_sum_week[i] = (cycling_time_sum_week[i-1]+cycling_time_sum_week[i+1])/2
    
    capacity_readings = RPT_dict['capacity_discharge_C_5']
    for i,c in enumerate(capacity_readings):
        if type(c)==list:
            capacity_readings[i]=np.nan      
    capacity_readings = np.array(capacity_readings)
    
    # Calculating the cumulative time on test each week
    cum_time = [0]
    for i,cycling_time in enumerate(cycling_time_sum_week):
        t = cum_time[-1]+char_time_sum_week[i]+cycling_time
        cum_time.append(t)
    cum_time = np.array(cum_time)

    capacity_fade_data = pd.DataFrame([cum_time,capacity_readings])
    capacity_fade_data = capacity_fade_data.T
    capacity_fade_data.set_axis(['Time', 'Capacity'], axis=1, inplace=True)
    capacity_fade_data.dropna(inplace=True)
    
    return capacity_fade_data

# Define a function to interpolate QV curves
def QV_interpolation(RPT_dict,V_interpolation):
    Q_interpolate_C_5_dchg=[]
    for i,Q_raw_C_5 in enumerate(RPT_dict['QV_discharge_C_5']['Q']):
        if len(Q_raw_C_5) > 0:
            Q_raw_C_5 = np.array(Q_raw_C_5)

            if len(Q_raw_C_5) >=10000:
                precision_C_5 = 8e-4
            else:
                precision_C_5 = 2e-5
            
            V_raw_C_5 = np.array(RPT_dict['QV_discharge_C_5']['V'][i])
            indice = np.argsort(V_raw_C_5)
            
            tck = interpolate.splrep(V_raw_C_5[indice],Q_raw_C_5[indice], s=precision_C_5)
            while np.isnan(tck[1]).any():
                precision_C_5 = precision_C_5*2
                tck = interpolate.splrep(V_raw_C_5[indice],Q_raw_C_5[indice], s=precision_C_5)
            ynew = interpolate.splev(V_interpolation, tck, der=0)
            Q_interpolate_C_5_dchg.append(ynew)
            
        else:
            Q_interpolate_C_5_dchg.append(np.full_like(V_interpolation,np.nan))
    
    Q_interpolate_C_5_dchg_df = pd.DataFrame(Q_interpolate_C_5_dchg)
    Q_interpolate_C_5_dchg_df = Q_interpolate_C_5_dchg_df.T

    return Q_interpolate_C_5_dchg_df


if __name__ == '__main__':
    valid_cells = pd.read_csv('valid_cells.csv').values.flatten().tolist()
    batch2 = ['G57C1','G57C2','G57C3','G57C4','G58C1', 'G26C3','G49C1','G49C2','G49C3','G49C4','G50C1','G50C3','G50C4'] 
    
    for cell in valid_cells:
        if cell not in batch2:
            subfolder = 'Release 1.0'
        else:
            subfolder = 'Release 2.0'
        
        # Specify the location for saving capacity fade data
        capacity_fade_data_dir = 'capacity_fade/{}/'.format(subfolder)
        Q_interpolation_dir = 'Q_interpolated/{}/'.format(subfolder)
        
        # Load and convert JSON data files to dictionary
        RPT_dict = convert_RPT_to_dict(cell,subfolder)
        cycling_dict = convert_cycling_to_dict(cell,subfolder)
        # Calculate capacity fade data and save as CSV files at a specific location
        capacity_fade_data = capacity_time_calculation(RPT_dict, cycling_dict)
        capacity_fade_data.to_csv(capacity_fade_data_dir+'{}.csv'.format(cell),index=False)
        # Interpolate QV curves
        V_interpolation = np.linspace(3.0,4.18,1000)
        Q_interpolation = QV_interpolation(RPT_dict,V_interpolation)
        Q_interpolation.to_csv(Q_interpolation_dir+'{}.csv'.format(cell),index=False,header=False)
