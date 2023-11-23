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
import pickle
parser = argparse.ArgumentParser(description='Nanopore data')
parser.add_argument('-i','--input', default='./test/xen50k.morefts.input_CNN_regression_modgen.RData', type=str,
                    help='input file name',metavar="character")
parser.add_argument('-t','--thread', default=24, type=int,
                    help='Number of cores allocated',metavar="integer")
parser.add_argument('-m','--model3c', default='./model/best_pos5_mix_3class_resnet_1992.h5', type=str,
                    help='Number of cores allocated',metavar="character")
parser.add_argument('-M','--model2c', default="./model/best_pos5_mix_3c_1vs1_3010_resnet_10.h5", type=str,
                    help='Number of cores allocated',metavar="character")
parser.add_argument('-r','--regmodel', default="./model/best_regression_mixnHEK_morefts_16384_1024_b1024_init_650k_XHe0_Mus_asin06.h5", type=str,
                    help='Number of cores allocated',metavar="character")

args = parser.parse_args()

if (args.input is None):
    print(args.print_help())
    warnings.warn('Input files, 3-class model, 2-class model and regression model must be supplied (-i,-m,-M,-r). ')

dinodir=os.path.dirname(os.path.abspath(__file__))
def main():

    chrmapping = pd.read_pickle("chrmapping.pkl")  # Assuming it's a pickle file, otherwise, use appropriate function
    # model1 = torch.load_weights("./model/best_pos5_mix_3class_resnet_1992.h5")
    agggrp = args.input.split('.')[0]
    # Load the Keras model
    model1 = torch.load(args.model3c)
    model2 = torch.load(args.model2c)
    model3 = torch.load(args.regmodel)
    # Load the data from the RDS file
    with open(args.input, 'rb') as file:
        test = pickle.load(file)

    # Assuming 'test_data' contains the loaded data from the RDS file
    test_x = test['x']  # Assuming 'x' contains the data to be reshaped

    # Extracting the required subset and reshaping it
    test_xc = test_x[:, :, :43]  # Extracting the first 43 columns
    test_xc = np.reshape(test_xc, (test_x.shape[0], 5, 43, 1))
    test_xc_tensor = torch.from_numpy(test_xc).float()
    with torch.no_grad():
        model1.eval()  # Set the model to evaluation mode
        pred_tensor = model1(test_xc_tensor)
        pred_np = pred_tensor.numpy()  # Convert predictions to NumPy array

    # Process the predictions to determine the predicted classes
    pred_test = np.argmax(pred_np, axis=1)
    pred_test[pred_test == 0] = 0
    pred_test[pred_test == 1] = 1
    pred_test[pred_test == 2] = 2

    c01 = np.where(pred_test != 2)[0]  # Indices where pred_test is not 2

    # Prepare the 'pred' DataFrame similar to R
    pred = pd.DataFrame({
        'V1': pred_np[:, 0],  # Assuming the columns of predictions in pred_np correspond to the model's output
        'V2': pred_np[:, 1],
        'V3': pred_np[:, 2],
        'n2': 1 - pred_np[:, 2],
        'c1': 0,
        'c2': 0
    })

    # Extract the subset based on c01 indices
    test_xc_c01 = test_xc[c01]

    # Convert test_xc_c01 to a PyTorch tensor
    test_xc_c01_tensor = torch.from_numpy(test_xc_c01).float()

    # Perform predictions using model2
    with torch.no_grad():
        model2.eval()  # Set the model to evaluation mode
        pred2_tensor = model2(test_xc_c01_tensor)
        pred2_np = pred2_tensor.numpy()  # Convert predictions to NumPy array

    # Convert predictions to DataFrame
    pred2 = pd.DataFrame({
        'V1': pred2_np[:, 0],  # Assuming the columns of predictions in pred2_np correspond to the model's output
        'V2': pred2_np[:, 1]
    })

    # Update 'pred' DataFrame based on 'c01' indices
    pred.loc[c01, ['V1', 'V2']] = pred2.values
    pred.loc[c01, 'V1'] = pred.loc[c01, 'c1'] * pred.loc[c01, 'n2']
    pred.loc[c01, 'V2'] = pred.loc[c01, 'c2'] * pred.loc[c01, 'n2']

    # Creating 'pred.2model' DataFrame based on grouping and summarizing 'pred' DataFrame
    pred_2model = pred[['V1', 'V2', 'V3']].copy()
    pred_2model['ref'] = test['y']
    pred_2model['id'] = test['info']["id"]
    pred_2model['cov'] = test['info']["cov"]
    pred_2model = pred_2model.groupby(['id', 'cov', 'ref']).agg({
        'V1': 'mean',
        'V2': 'mean',
        'V3': 'mean'
    }).reset_index()

    # Adding 'pred' column to 'pred.2model'
    pred_2model['pred'] = pred_2model.apply(lambda row: str(row.idxmax()), axis=1)

    # Splitting 'id' column into 'chr_str' and 'position'
    pred_2model[['chr_str', 'position']] = pred_2model['id'].str.split(':', expand=True)

    # Modifying 'strand' column based on conditions
    pred_2model['strand'] = pred_2model['chr_str'].str[-1]
    pred_2model['strand'] = pred_2model['strand'].apply(lambda x: '+' if x == '1' else '-')

    # Extracting 'chr' from 'chr_str' and converting 'cov' to numeric
    pred_2model['chr'] = pred_2model['chr_str'].str[:-1]
    pred_2model['cov'] = pd.to_numeric(pred_2model['cov'])

    # Assuming 'chrmapping' is a DataFrame with columns 'chromid' and 'chroms'
    # Mapping 'chr' values using 'chrmapping' DataFrame
    chrmapping_dict = dict(zip(chrmapping['chromid'], chrmapping['chroms']))
    pred_2model['contig'] = pred_2model['chr'].map(chrmapping_dict)

    # Filtering 'pred_2model' for cov >= 40
    pred_2model_cov40 = pred_2model[pred_2model['cov'] >= 40]

    # Filtering 'id' based on condition from 'pred_2model_cov40'
    id = pred_2model_cov40.loc[pred_2model_cov40['pred'].isin(['1']), 'id']

    # Getting indices where 'test' info id is in 'id'
    ind = test[test['info']['id'].isin(id)].index.tolist()

    # Assuming 'model3' is a PyTorch model and 'test_x' is a tensor for test data
    # Predicting using 'model3' on a subset of test data
    pred3 = model3(0.01 * test_x[ind])

    print("model 3")

    # Creating 'pred.regression' DataFrame
    pred_regression = pd.DataFrame({
        'id': test['info']['id'][ind],
        'cov': test['info']['cov'][ind],
        'ref': test['y'][ind],
        'rate': test['y2'][ind],
        'pred.rate': pred3[:, 0]  # Assuming pred3 contains predictions in column 0
    })

    # Filtering 'pred_all' DataFrame for cov >= 20 and applying mutation
    # Perform a left join on 'pred_2model' and selected columns from 'pred_regression' by 'id'
    pred_all = pd.merge(pred_2model, pred_regression[['id', 'ref', 'rate']], on='id', how='left')
    pred_all_filtered = pred_all[pred_all['cov'] >= 20].copy()
    pred_all_filtered['pred.rate'] = np.sin(pred_all_filtered['pred.rate'] ** (5 / 3))

    # Splitting 'pred_all_filtered' DataFrame based on 'pred' value
    class0 = pred_all_filtered[pred_all_filtered['pred'] == 0][
        ["contig", "position", "strand", "cov", "0", "1", "2", "ref", "pred"]]
    class1 = pred_all_filtered[pred_all_filtered['pred'] == 1][
        ["contig", "position", "strand", "cov", "0", "1", "2", "ref", "pred", "rate", "pred.rate"]]
    class2 = pred_all_filtered[pred_all_filtered['pred'] == 2][
        ["contig", "position", "strand", "cov", "0", "1", "2", "ref", "pred"]]

    # Writing DataFrames to text files
    class0.to_csv(agggrp + ".output_prediction_CNN_class0.txt", sep='\t', index=False)
    class1.to_csv(agggrp + ".output_prediction_CNN_class1.txt", sep='\t', index=False)
    class2.to_csv(agggrp + ".output_prediction_CNN_class2.txt", sep='\t', index=False)
if __name__ == '__main__':
    main()
