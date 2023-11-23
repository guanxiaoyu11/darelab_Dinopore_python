import pandas as pd

import argparse
import os
import gc
import warnings
from multiprocessing import Pool
import csv
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import h5py
parser = argparse.ArgumentParser(description='Nanopore data')
parser.add_argument('-i','--input', default='./test/xen50k.morefts.input_CNN_regression_modgen.RData', type=str,
                    help='input file name',metavar="character")
parser.add_argument('-t','--thread', default=24, type=int,
                    help='Number of cores allocated',metavar="integer")

args = parser.parse_args()

if (args.input is None):
    print(args.print_help())
    warnings.warn('Input files must be supplied (-i). ')

dinodir=os.path.dirname(os.path.abspath(__file__))
def main():

    chrmapping = pd.read_pickle("chrmapping.pkl")  # Assuming it's a pickle file, otherwise, use appropriate function
    # model1 = torch.load_weights("./model/best_pos5_mix_3class_resnet_1992.h5")
    agggrp = args.input.split('.')[0]
    # Load the Keras model
    # model1 = torch.load('./model/best_pos5_mix_3class_resnet_1992.h5',trainable=True)
    # model2 = torch.load("./model/best_pos5_mix_3c_1vs1_3010_resnet_10.h5")
    # model3 = torch.load("./model/best_regression_mixnHEK_morefts_16384_1024_b1024_init_650k_XHe0_Mus_asin06.h5")

    # Assuming 'opt' is a dictionary loaded from somewhere
    with h5py.File(args.input, 'r') as hf:
        train_matrix = {}
        test_matrix = {}

        train_matrix['x'] = hf['train_matrix']['x'][:]
        train_matrix['y'] = hf['train_matrix']['y'][:]
        train_matrix['y2'] = hf['train_matrix']['y2'][:]
        train_matrix['info'] = hf['train_matrix']['info'][:]

        test_matrix['x'] = hf['test_matrix']['x'][:]
        test_matrix['y'] = hf['test_matrix']['y'][:]
        test_matrix['y2'] = hf['test_matrix']['y2'][:]
        test_matrix['info'] = hf['test_matrix']['info'][:]

    print("Done")

    # Set the model to evaluation mode
    model1.eval()

    # Assuming 'test' is a dictionary-like object with key 'x'
    # Predict using the model on test data
    with torch.no_grad():
        input_data = test_matrix['x'][:, :, :43]  # Considering columns 1 to 43
        output = model1(input_data)
        pred = torch.argmax(output, dim=1).cpu().numpy()
    pred_df = pd.DataFrame(pred, columns=['V1', 'V2', 'V3'])
    # Convert predictions to numeric values
    pred_test = pred_df[['V1', 'V2', 'V3']].idxmax(axis=1)
    pred_test = pred_test.str.replace('V1', '0')
    pred_test = pred_test.str.replace('V2', '1')
    pred_test = pred_test.str.replace('V3', '2')
    pred_test = pred_test.astype(int)

    c01 = pred_test[pred_test != 2].index.tolist()

    # Modify pred DataFrame
    pred['n2'] = 1 - pred['V3']
    pred['c1'] = 0
    pred['c2'] = 0
    # Assuming 'test' and 'model2' are defined and 'test_x' is a torch tensor
    # Make predictions using the PyTorch model
    with torch.no_grad():
        pred2 = model2(test_matrix['x'][:, :, :43])  # Assuming 'test_x' has the required dimensions

    # Convert predictions to a pandas DataFrame
    pred2 = pd.DataFrame(pred2.numpy())

    # Assuming 'pred' is a pandas DataFrame with columns 'c01', 'c1', 'c2', 'n2', 'V1', 'V2', 'V3'
    # Perform calculations
    pred['c01'][[5, 6]] = pred2.values
    pred['V1'] = pred['c1'] * pred['n2']
    pred['V2'] = pred['c2'] * pred['n2']


    pred_2model = pd.concat([pred.iloc[:, :3], test_matrix[['y', 'info']].reset_index(drop=True)], axis=1)
    pred_2model = pred_2model.groupby(['id', 'cov', 'ref']).agg(
        {'V1': 'mean', 'V2': 'mean', 'V3': 'mean'}).reset_index()

    # Assign 'pred' column based on maximum value among 'V1', 'V2', 'V3'
    pred_2model['pred'] = pred_2model[['V1', 'V2', 'V3']].idxmax(axis=1)

    # Assuming 'chrmapping' is a pandas DataFrame with columns 'chromid', 'chroms'
    # Mapping chromosome values
    replace_dict = dict(zip(chrmapping['chroms'], chrmapping['chromid']))
    pred_2model['chr'] = pred_2model['chr'].replace(replace_dict)

    # Extract 'chr_str', 'position', 'strand' from 'id' column and manipulate other columns
    pred_2model[['chr_str', 'position']] = pred_2model['id'].str.split(':', expand=True)
    pred_2model['strand'] = np.where(pred_2model['strand'] == 1, '+', '-')
    pred_2model['cov'] = pd.to_numeric(pred_2model['cov'])

    # Adding 'contig' column based on mapping
    pred_2model['contig'] = pred_2model['chr'].map(replace_dict)

    pred2 = model2.predict(test['x'][c01, :, :43])
    pred['V5'][c01] = pred2[:, 0]
    pred['V6'][c01] = pred2[:, 1]

    pred.loc[c01, 'V1'] = pred.loc[c01, 'c1'] * pred.loc[c01, 'n2']
    pred.loc[c01, 'V2'] = pred.loc[c01, 'c2'] * pred.loc[c01, 'n2']

    #####Regression model to predict editing rate####

    # Assuming 'pred_2model' is a pandas DataFrame
    pred_2modelcov40 = pred_2model[pred_2model['cov'] >= 40]

    # Filter 'id' based on condition
    id = pred_2modelcov40.loc[pred_2modelcov40['pred'].isin(["1"]), 'id'].tolist()

    # Assuming 'test', 'test_info', and 'model3' are defined and 'test_x' is a torch tensor
    # Get indices where 'test_info' 'id' is in 'id'
    ind = test_matrix['info'][test_matrix['info']['id'].isin(id)].index.tolist()

    # Make predictions using 'model3'
    pred3 = model3(0.01 * test_matrix['x'][ind])

    # Assuming 'test_info', 'test_y', and 'test_y2' are columns in 'test' DataFrame
    # Prepare 'pred.regression' DataFrame
    pred_regression = pd.DataFrame({
        'id': test_matrix['x'].loc[ind, 'id'],
        'cov': test_matrix['info'].loc[ind, 'cov'],
        'ref': test_matrix['y'].loc[ind],
        'rate': test_matrix['y2'].loc[ind],
        'pred.rate': pred3[:, 0]  # Assuming the predictions are in the first column
    })

    # Group by 'id', 'cov', 'ref', 'rate' and calculate mean for 'pred.rate'
    pred_regression = pred_regression.groupby(['id', 'cov', 'ref', 'rate'])['pred.rate'].mean().reset_index()

    pred_all = pd.merge(pred_2model, pred_regression[['id', 'rate', 'pred.rate']], on='id', how='left')

    # Filter based on 'cov' column >= 20
    pred_all = pred_all[pred_all['cov'] >= 20]

    # Apply transformation to 'pred.rate' column using sin function
    pred_all['pred.rate'] = np.sin(pred_all['pred.rate'] ** (5 / 3))

    # Split 'pred.all' DataFrame based on 'pred' column values
    class0 = pred_all[pred_all['pred'] == 0][["contig", "position", "strand", "cov", "0", "1", "2", "ref", "pred"]]
    class1 = pred_all[pred_all['pred'] == 1][
        ["contig", "position", "strand", "cov", "0", "1", "2", "ref", "pred", "rate", "pred.rate"]]
    class2 = pred_all[pred_all['pred'] == 2][["contig", "position", "strand", "cov", "0", "1", "2", "ref", "pred"]]

    # Write DataFrames to text files
    class0.to_csv(agggrp + ".output_prediction_CNN_class0.txt", sep='\t', index=False)
    class1.to_csv(agggrp + ".output_prediction_CNN_class1.txt", sep='\t', index=False)
    class2.to_csv(agggrp + ".output_prediction_CNN_class2.txt", sep='\t', index=False)
if __name__ == '__main__':
    main()
