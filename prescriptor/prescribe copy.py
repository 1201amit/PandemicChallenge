# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import argparse

from static import *
import pandas as pd
import numpy as np
import torch
import os
from dataloader import *
from dateutil.parser import parse 
import datetime
from matplotlib import pyplot as plt
from encdec_model import Encoder, Decoder1
static_path = 'timeseries/Consolidated.csv'
static_df = pd.read_csv(static_path)

NPI_COLS = ['C1_School closing',
            'C2_Workplace closing',
            'C3_Cancel public events',
            'C4_Restrictions on gatherings',
            'C5_Close public transport',
            'C6_Stay at home requirements',
            'C7_Restrictions on internal movement',
            'C8_International travel controls',
            'H1_Public information campaigns',
            'H2_Testing policy',
            'H3_Contact tracing',
            'H6_Facial Coverings']

ID_COLS = ['CountryName',
           'RegionName',
           'GeoID',
           'Date']

NB_LOOKBACK_DLAYS = 21

def prescribe(start_date: str,
              end_date: str,
              path_to_prior_ips_file: str,
              path_to_cost_file: str,
              output_file_path) -> None:
    """
    Generates and saves a file with daily intervention plan prescriptions for the given countries, regions and prior
    intervention plans, between start_date and end_date, included.
    :param start_date: day from which to start making prescriptions, as a string, format YYYY-MM-DDD
    :param end_date: day on which to stop making prescriptions, as a string, format YYYY-MM-DDD
    :param path_to_prior_ips_file: path to a csv file containing the intervention plans between inception date
    (Jan 1 2020) and end_date, for the countries and regions for which a prescription is needed
    :param path_to_cost_file: path to a csv file containing the cost of each individual intervention, per country
    See covid_xprize/validation/data/uniform_random_costs.csv for an example
    :param output_file_path: path to file to save the prescriptions to
    :return: Nothing. Saves the generated prescriptions to an output_file_path csv file
    See 2020-08-01_2020-08-04_prescriptions_example.csv for an example
    """
    pres_df = prescribe_df(start_date, end_date, path_to_ips_file,path_to_cost_file, verbose=False)
    # Create the output path
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    # Save to a csv file
    pres_df.to_csv(output_file_path, index=False)
    print(f"Saved predictions to {output_file_path}")


def prescribe_df (start_date_str: str, end_date_str: str, path_to_ips_file: str, verbose=False):
    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')

    hist_ips_df = pd.read_csv(path_to_ips_file,
                              parse_dates=['Date'],
                              encoding="ISO-8859-1",
                              dtype={"RegionName": str},
                              error_bad_lines=True)

    hist_ips_df['GeoID'] = hist_ips_df['CountryName'] + '__' + hist_ips_df['RegionName'].astype(str)

    for col in NPI_COLS:
        hist_ips_df[col] = pd.to_numeric(hist_ips_df[col], errors='coerce') # converting object to numeric 
        hist_ips_df[col].interpolate(method='linear', inplace=True)

    for g in hist_ips_df['GeoID'].unique():
        if verbose:
            print('\nPredicting for', g)
        growthrates = []
        while current_date <= end_date:
            X = np.concatenate([fetch_npis(current_date, NB_LOOKBACK_DLAYS, g, hist_ips_df).flatten(),
                                 static(current_date, static_df, g).flatten()]) 
            X = tf.convert_to_tensor(np.asarray(X, dtype=np.float32))
            y_pred = Encoder(Decoder1(X))
            growthrate = y_pred/100
            growthrates.append(growthrate)
            current_date = current_date + np.timedelta64(1, 'D')
        


    

    return pres_df

prescribe_df('2020-08-10', '2020-09-20', 'timeseries/2020-09-30_historical_ip.csv','final.csv')


# !!! PLEASE DO NOT EDIT. THIS IS THE OFFICIAL COMPETITION API !!!
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date",
                        dest="start_date",
                        type=str,
                        required=True,
                        help="Start date from which to prescribe, included, as YYYY-MM-DD."
                             "For example 2020-08-01")
    parser.add_argument("-e", "--end_date",
                        dest="end_date",
                        type=str,
                        required=True,
                        help="End date for the last prescription, included, as YYYY-MM-DD."
                             "For example 2020-08-31")
    parser.add_argument("-ip", "--interventions_past",
                        dest="prior_ips_file",
                        type=str,
                        required=True,
                        help="The path to a .csv file of previous intervention plans")
    parser.add_argument("-c", "--intervention_costs",
                        dest="cost_file",
                        type=str,
                        required=True,
                        help="Path to a .csv file containing the cost of each IP for each geo")
    parser.add_argument("-o", "--output_file",
                        dest="output_file",
                        type=str,
                        required=True,
                        help="The path to an intervention plan .csv file")
    args = parser.parse_args()
    print(f"Generating prescriptions from {args.start_date} to {args.end_date}...")
    prescribe(args.start_date, args.end_date, args.prior_ips_file, args.cost_file, args.output_file)
    print("Done!")"""

