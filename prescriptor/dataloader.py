import datetime
import pandas as pd
import numpy as np

def static (prediction_date, static_df , g):
    # prediction date is of type string 
    country = g.split('__')[0]
    static_data = static_df
    static_data = static_data.loc[static_data["CountryName"] == country]
    static_data = static_data.drop('Population', axis = 1 )
    cols_to_norm = ['Density', 'Median Age']
    static_data[cols_to_norm] = static_data[cols_to_norm].apply(lambda x:
                                 (x - x.min()) / x.max()-x.min()) 
    static_data = static_data.to_numpy()
    tmp = static_data[:,3:] 
    final_static_data = static_data[:,0:3].astype(np.float64)
    date = datetime.datetime.strptime(prediction_date, "%Y-%m-%d")
    ref_date = date - datetime.timedelta(21)
    if ref_date.month == date.month:
        temperature = tmp[:,int(date.month)].reshape((1,1)).astype(np.float64)
    else:
        temperature = ((tmp[:,int(date.month)] + tmp[:,int(ref_date.month)]) / 2
                        .reshape(1,1)).astype(np.float64)
    new_final_static_data = np.concatenate((final_static_data,temperature),axis=1)
    new_final_static_data = new_final_static_data.to_numpy().reshape((1,4)).astype(np.float64)

    return new_final_static_data

def fetch_npis (current_date, lookback_days, g, main_df):
    npi_df = main_df.loc[main_df["GeoID"] == g]
    min_date = current_date - datetime.timedelta(lookback_days)
    npi_df = npi_df[npi_df.Date >= (min_date) and npi_df.Date < (current_date))]
    npi_df.drop(columns = ['CountryName', 'RegionName', 'GeoID'])
    npi_df = npi_df.to_numpy()
    npi_df = npi_df.reshape((1,lookback_days*12)).astype(np.float64)
    return npi_df







