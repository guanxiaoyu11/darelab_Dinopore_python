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
import random
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='Nanopore data')

parser.add_argument('-t','--thread', default=24, type=int,
                    help='Number of cores allocated',metavar="integer")
parser.add_argument('-i','--input', default='./test/xen50k.Agg.morefts.10bin.inML.txt', type=str,
                    help='input file name',metavar="character")
parser.add_argument('-o','--out', default='xen50k.morefts.input_CNN_regression_modgen.RData', type=str,
                    help='output file name',metavar="character")
parser.add_argument('-c','--classref', default='./test/groundtruth_class_regression.tsv', type=str,
                    help='ground truth of class for coordinates',metavar="character")

args = parser.parse_args()

if (args.out is None) or (args.classref is None)or (args.input is None):
    print(args.print_help())
    warnings.warn('Input files must be supplied (-i,-o, -c). ')

dinodir=os.path.dirname(os.path.abspath(__file__))
def generate_before_after(x):
    x = pd.DataFrame(x)
    x[['chr', 'position', 'strand']] = x['x'].str.split(':', expand=True)
    x['position'] = x['position'].astype(int)
    x['po.le1'] = x['position'] + 1
    x['po.le2'] = x['position'] + 2
    x['po.la1'] = x['position'] - 1
    x['po.la2'] = x['position'] - 2
    x['po.le.id1'] = x['chr'] + ':' + x['po.le1'].astype(str) + ':' + x['strand']
    x['po.la.id1'] = x['chr'] + ':' + x['po.la1'].astype(str) + ':' + x['strand']
    x['po.le.id2'] = x['chr'] + ':' + x['po.le2'].astype(str) + ':' + x['strand']
    x['po.la.id2'] = x['chr'] + ':' + x['po.la2'].astype(str) + ':' + x['strand']

    y = {'ld1': x['po.le.id1'].tolist(),
         'lg1': x['po.la.id1'].tolist(),
         'ld2': x['po.le.id2'].tolist(),
         'lg2': x['po.la.id2'].tolist()}

    return y
def replace0(x):
    cols_to_replace = ['event_stdv.m', 'event_stdv.s', 'event_stdv.A', 'count.m', 'count.s',
                       'ins3.m', 'ins1.m', 'ins2.m', 'ins4.m', 'ins5.m',
                       'ins3.s', 'ins1.s', 'ins2.s', 'ins4.s', 'ins5.s']

    x[cols_to_replace] = x[cols_to_replace].replace(0, 0.00001)
    return x
def replace_list(pat, repl, x):
    y = [repl[pat[0].index(ch)] if ch in pat else ch for ch in x]
    return y

def decode(x,chroms,chromid):
    x = pd.DataFrame(x)
    x['contig'] = replace_list(chroms, chromid, x['contig'])
    x['contig'] = pd.to_numeric(x['contig'])

    x['reference_kmer'] = x['reference_kmer'].str.replace("A", "1")
    x['reference_kmer'] = x['reference_kmer'].str.replace("C", "2")
    x['reference_kmer'] = x['reference_kmer'].str.replace("G", "3")
    x['reference_kmer'] = x['reference_kmer'].str.replace("T", "4")

    x['reference_kmer'] = pd.to_numeric(x['reference_kmer'])

    patterns = ["A", "C", "G", "T","N"]
    replacement = list(map(str, range(1, 6)))

    pattern_dict = dict(zip(patterns, replacement))
    x['REF'] = x['REF'].replace(pattern_dict)
    x['REF'] = pd.to_numeric(x['REF'])

    x['strand'] = x['strand'].str.replace("p", "1")
    x['strand'] = x['strand'].str.replace("n", "2")
    x['strand'] = pd.to_numeric(x['strand'])

    x['chr.str'] = x['contig'].astype(str) + x['strand'].astype(str)
    x = x[['chr.str'] + [col for col in x.columns if col != 'chr.str']]
    x['chr.str'] = pd.to_numeric(x['chr.str'])
    return x

def transform_each_pos5_big(x, y, ncol):
    ny = y.shape[0]

    resultb1 = []
    resulta1 = []
    resultm = []
    resultb2 = []
    resulta2 = []

    for i in range(ny):
        if (y[i, 0] == x[0]) and (y[i, 1] == (x[1] - 1)):
            resultb1.append(i)
        if (y[i, 0] == x[0]) and (y[i, 1] == (x[1] + 1)):
            resulta1.append(i)
        if (y[i, 0] == x[0]) and (y[i, 1] == x[1]):
            resultm.append(i)
        if (y[i, 0] == x[0]) and (y[i, 1] == (x[1] - 2)):
            resultb2.append(i)
        if (y[i, 0] == x[0]) and (y[i, 1] == (x[1] + 2)):
            resulta2.append(i)

    result = np.zeros((50, ncol))

    for i in range(10):
        b2 = np.random.choice(len(resultb2), 1)
        b1 = np.random.choice(len(resultb1), 1)
        a1 = np.random.choice(len(resulta1), 1)
        a2 = np.random.choice(len(resulta2), 1)
        result[5 * i, :] = y[b2[0], :]
        result[5 * i + 1, :] = y[b1[0], :]
        result[5 * i + 2, :] = y[resultm[i], :]
        result[5 * i + 3, :] = y[a1[0], :]
        result[5 * i + 4, :] = y[a2[0], :]

    return result
def countnb5(refe, strand, pos, countld1, countlg1, countld2, countlg2):
    for i in range(len(pos)):
        if (refe[i] == "A" and strand[i] == "p") or (refe[i] == "T" and strand[i] == "n"):
            k = 0
            while pos[i + k + 1] == pos[i] + 1:
                k += 1
            countld1[i] = k

            h = 0
            while pos[i - h - 1] == pos[i] - 1:
                h += 1
            countlg1[i] = h

            m = 0
            while pos[i + k + m + 1] == pos[i] + 2:
                m += 1
            countld2[i] = m

            n = 0
            while pos[i - h - n - 1] == pos[i] - 2:
                n += 1
                if i - h - n - 1<0:
                    break
            countlg2[i] = n

def check_mono(x, n):
    xt = x.iloc[:, list(range(5)) + [n]].copy()
    xt.sort_values(by=["strand", "contig", "position"], inplace=True)

    p = x[((x['REF'] == "A") & (x['strand'] == "p")) | ((x['REF'] == "T") & (x['strand'] == "n"))]['posi_id']

    xt["cld1"] = 0
    xt["clg1"] = 0
    xt["cld2"] = 0
    xt["clg2"] = 0
    countnb5(xt["REF"], xt["strand"], xt["position"], xt["cld1"], xt["clg1"], xt["cld2"], xt["clg2"])
    # countnb5_optimized(xt)

    check = xt.groupby(["contig", "position", "strand", "posi_id", "REF"]).agg(
        ld1=("cld1", "max"),
        lg1=("clg1", "max"),
        ld2=("cld2", "max"),
        lg2=("clg2", "max"),
        n=("REF", "size")
    ).reset_index()

    check = check[((check['REF'] == "A") & (check['strand'] == "p")) | ((check['REF'] == "T") & (check['strand'] == "n"))]
    check = check[check["posi_id"].isin(p)]

    check["t"] = check["ld1"] * check["lg1"] * check["n"] * check["ld2"] * check["lg2"]

    return check

def transform_each_pos5(x, y, ncol):
    ny = y.shape[0]

    resultb1 = []
    resulta1 = []
    resultb2 = []
    resulta2 = []

    for i in range(ny):
        if (y[i, 0] == x[0]) and (y[i, 1] == (x[1] - 1)):
            resultb1.append(i)
        if (y[i, 0] == x[0]) and (y[i, 1] == (x[1] + 1)):
            resulta1.append(i)
        if (y[i, 0] == x[0]) and (y[i, 1] == (x[1] - 2)):
            resultb2.append(i)
        if (y[i, 0] == x[0]) and (y[i, 1] == (x[1] + 2)):
            resulta2.append(i)

    nb1 = len(resultb1)
    na1 = len(resulta1)
    nb2 = len(resultb2)
    na2 = len(resulta2)

    result = np.zeros((5 * nb1 * nb2 * na1 * na2, ncol))
    for i in range(nb1 * nb2 * na1 * na2):
        result[5 * i + 2, :] = x

    for i in range(nb2):
        for j in range(nb1):
            for h in range(na1):
                for k in range(na2):
                    ta = (nb1 * na1 * na2) * i + (na1 * na2) * j + na2 * h + k
                    result[5 * ta, :] = y[resultb2[i], :]
                    result[5 * ta + 1, :] = y[resultb1[j], :]
                    result[5 * ta + 3, :] = y[resulta1[h], :]
                    result[5 * ta + 4, :] = y[resulta2[k], :]

    return result

def transform_each_pos5_regular(x, y, ncol):
    ny = y.shape[0]

    resultb1 = []
    resulta1 = []
    resultm = []
    resultb2 = []
    resulta2 = []

    for i in range(ny):
        if (y[i, 0] == x[0]) and (y[i, 1] == (x[1] - 1)):
            resultb1.append(i)
        if (y[i, 0] == x[0]) and (y[i, 1] == (x[1] + 1)):
            resulta1.append(i)
        if (y[i, 0] == x[0]) and (y[i, 1] == x[1]):
            resultm.append(i)
        if (y[i, 0] == x[0]) and (y[i, 1] == (x[1] - 2)):
            resultb2.append(i)
        if (y[i, 0] == x[0]) and (y[i, 1] == (x[1] + 2)):
            resulta2.append(i)

    result = np.zeros((50, ncol))  # Initialize result matrix

    for i in range(10):
        b2 = np.random.choice(len(resultb2), 1)
        b1 = np.random.choice(len(resultb1), 1)
        m = np.random.choice(len(resultm), 1)
        a1 = np.random.choice(len(resulta1), 1)
        a2 = np.random.choice(len(resulta2), 1)

        result[5 * i] = y[b2[0]]
        result[5 * i + 1] = y[b1[0]]
        result[5 * i + 2] = y[m[0]]
        result[5 * i + 3] = y[a1[0]]
        result[5 * i + 4] = y[a2[0]]

    return result

def transform_dat_big(checkdf, df,chroms,chromid, ncols=70):
    random.seed(1999)
    posm = checkdf[(checkdf['t'] > 10) & (checkdf['n'] < 10)]['posi_id']
    df1 = decode(df, chroms, chromid)
    dfm = df1[df['posi_id'].isin(posm) & (
            ((df1['REF'] == 1) & (df1['strand'] == 1)) | ((df1['REF'] == 4) & (df1['strand'] == 2)))].copy()
    df1 = df1.drop(columns=['contig', 'strand', 'posi_id'])
    dfm = dfm.drop(columns=['contig', 'strand', 'posi_id'])
    dfm2 = dfm.iloc[:, [0, 1]].drop_duplicates().values
    df1 = df1.values
    lt_mt = [transform_each_pos5_regular(dfm2[i], df1, ncols) for i in tqdm(range(len(dfm2)))]
    lt_mt1 = np.vstack(lt_mt)
    lt_mt2 = np.transpose(np.reshape(np.transpose(lt_mt1), (70, 5, int(lt_mt1.shape[0] / 5))), (1, 0, 2))
    x_train = lt_mt2[:, 5:ncols, :]
    x_train = np.transpose(x_train, (2, 0, 1))
    x_train = x_train.reshape((lt_mt1.shape[0] // 5, 5, 65, 1))
    info_train = lt_mt2[:, :5, :]
    return {'x': x_train, 'info': info_train}

def transform_dat(checkdf, df, chroms,chromid,ncols=70):

    posm = checkdf[(checkdf['t'] > 0) & (checkdf['t'] <= 10)]['posi_id']
    df1 = decode(df, chroms, chromid)
    dfm = df1[df['posi_id'].isin(posm) & (
                ((df1['REF'] == 1) & (df1['strand'] == 1)) | ((df1['REF'] == 4) & (df1['strand'] == 2)))].copy()
    df1 = df1.drop(columns=['contig', 'strand', 'posi_id'])
    dfm = dfm.drop(columns=['contig', 'strand', 'posi_id'])
    dfm =  dfm.sample(frac=1).reset_index(drop=True)
    dfm=dfm.values
    df1=df1.values

    lt_mt = [transform_each_pos5(dfm[i], df1, ncols) for i in tqdm(range(len(dfm)))]
    lt_mt1 = np.vstack(lt_mt)
    lt_mt2 = np.transpose(np.reshape(np.transpose(lt_mt1), (70, 5, int(lt_mt1.shape[0] / 5))), (1, 0, 2))
    x_train = lt_mt2[:, 5:ncols, :]
    x_train = np.transpose(x_train, (2, 0, 1))
    x_train = x_train.reshape((lt_mt1.shape[0] // 5, 5, 65, 1))
    info_train = lt_mt2[:, :5, :]
    return {'x': x_train, 'info': info_train}

def transform_dat_bb(checkdf, df, chroms,chromid,ncols=70):
    np.random.seed(1999)
    posm = checkdf[(checkdf['t'] > 10) & (checkdf['n'] == 10)]['posi_id']
    df1 = decode(df, chroms, chromid)
    dfm = df1[df['posi_id'].isin(posm) & (
            ((df1['REF'] == 1) & (df1['strand'] == 1)) | ((df1['REF'] == 4) & (df1['strand'] == 2)))].copy()
    df1 = df1.drop(columns=['contig', 'strand', 'posi_id'])
    dfm = dfm.drop(columns=['contig', 'strand', 'posi_id'])
    dfm2 = dfm.iloc[:, [0, 1]].drop_duplicates().values
    df1 = df1.sample(frac=1).reset_index(drop=True).values

    lt_mt = [transform_each_pos5_big(dfm2[i], df1, ncols) for i in tqdm(range(len(dfm2)))]

    lt_mt1 = np.vstack(lt_mt)
    lt_mt2 = np.transpose(np.reshape(np.transpose(lt_mt1), (70, 5, int(lt_mt1.shape[0] / 5))), (1, 0, 2))
    x_train = lt_mt2[:, 5:ncols, :]
    x_train = np.transpose(x_train, (2, 0, 1))
    x_train = x_train.reshape((lt_mt1.shape[0] // 5, 5, 65, 1))
    info_train = lt_mt2[:, :5, :]
    return {'x': x_train, 'info': info_train}

def main():
    print(args.input)
    file_df = pd.read_csv(args.input, header=0, sep='\t', quoting=csv.QUOTE_NONE, nrows=1000000)

    df = file_df[file_df['reference_kmer'] != ""]


    # Read the truth file
    truth = pd.read_csv(args.classref, sep='\t')

    # Create a mapping for chromosomes
    chroms = np.sort(truth['contig'].unique())
    chromid = [str(i + 1) for i in range(len(chroms))][0]
    chrmapping = pd.DataFrame({'chroms': chroms, 'chromid': chromid})
    chrmapping.to_pickle("chrmapping.pkl")

    truth['posi_id'] = truth['contig'].astype(str) + ':' + truth['position'].astype(str) + ':' + truth['strand'].astype(
        str)
    truth['REF'] = 'A'
    truth2 = decode(truth,chroms,chromid)
    truth2['id'] = truth2['chr.str'] + ':' + truth2['position'].astype(str)
    truth = pd.concat([truth, truth2['id']], axis=1)

    # Split truth data into train and test sets
    pos = truth['posi_id'].unique()
    posab = generate_before_after(pos)
    truth = truth.sort_values(by='id')
    cut = int(len(truth) * 0.9)
    test_pos = truth.iloc[cut:, :]
    train_pos = truth.drop(test_pos.index)
    test_pos = truth.iloc[cut:]
    train_pos = truth.iloc[:cut]
    dat = df[df['posi_id'].isin(
        np.unique(np.concatenate((pos, posab['ld1'], posab['lg1'], posab['ld2'], posab['lg2']))))].drop_duplicates()
    del df
    gc()
    dat = replace0(dat)
    dat[["event_stdv.m", "event_stdv.s", "event_stdv.A", "count.m", "count.s", "ins3.m", "ins1.m", "ins2.m", "ins4.m",
         "ins5.m", "ins3.s", "ins1.s", "ins2.s", "ins4.s", "ins5.s"]] = \
        np.log(dat[["event_stdv.m", "event_stdv.s", "event_stdv.A", "count.m", "count.s", "ins3.m", "ins1.m", "ins2.m",
                    "ins4.m", "ins5.m", "ins3.s", "ins1.s", "ins2.s", "ins4.s", "ins5.s"]])
    dat = dat.drop(dat.columns[list(range(29, 34)) + list(range(50, 55))], axis=1)
    dat = dat.dropna().reset_index(drop=True)

    # Generate before and after data for train and test sets
    train_ab = generate_before_after(train_pos['posi_id'])
    test_ab = generate_before_after(test_pos['posi_id'])

    # Filter data based on posi_id for train and test sets
    train_dat = dat[dat['posi_id'].isin(
        np.concatenate((train_pos['posi_id'], train_ab['ld1'], train_ab['lg1'], train_ab['ld2'], train_ab['lg2'])))]
    test_dat = dat[dat['posi_id'].isin(
        np.concatenate((test_pos['posi_id'], test_ab['ld1'], test_ab['lg1'], test_ab['ld2'], test_ab['lg2'])))]

    # Define a list of columns
    cols = train_dat.columns[7:71]

    # Calculate max and min values for each column
    max_vals = train_dat[cols].max()
    min_vals = train_dat[cols].min()

    # Calculate the difference
    diff_vals = max_vals - min_vals

    # Normalize the data
    for col in cols:
        train_dat[col] = ((train_dat[col] - min_vals[col]) * 100) / diff_vals[col]
        test_dat[col] = ((test_dat[col] - min_vals[col]) * 100) / diff_vals[col]

    # Check for monotonicity
    checktrain = check_mono(train_dat, 72)
    checktest = check_mono(test_dat, 72)

    # Transform data for small, big, and bb datasets
    train_s = transform_dat(checktrain, train_dat,chroms,chromid)
    train_r = transform_dat_big(checktrain, train_dat,chroms,chromid)
    train_b = transform_dat_bb(checktrain, train_dat,chroms,chromid)

    test_s = transform_dat(checktest, test_dat,chroms,chromid)
    test_r = transform_dat_big(checktest, test_dat,chroms,chromid)
    test_b = transform_dat_bb(checktest, test_dat,chroms,chromid)
    train_tmp = {}

    # Check if train_r and train_b exist
    if 'train_r' not in locals() and 'train_b' not in locals():
        train_tmp = train_s
    elif 'train_r' not in locals():
        train_tmp['x'] = np.concatenate((train_s['x'], train_b['x']), axis=0)
        train_tmp['info'] = np.concatenate((train_s['info'], train_b['info']), axis=2)
    elif 'train_b' not in locals():
        train_tmp['x'] = np.concatenate((train_s['x'], train_r['x']), axis=0)
        train_tmp['info'] = np.concatenate((train_s['info'], train_r['info']), axis=2)
    else:
        train_tmp['x'] = np.concatenate((train_s['x'], train_r['x'], train_b['x']), axis=0)
        train_tmp['info'] = np.concatenate((train_s['info'], train_r['info'], train_b['info']), axis=2)

    # Find and remove any NaN values in train_tmp['x']
    ineli = np.isnan(train_tmp['x']).any(axis=(1, 2, 3))
    if np.sum(ineli) > 0:
        train_tmp['x'] = train_tmp['x'][~ineli]
        train_tmp['info'] = train_tmp['info'][:, :, ~ineli]
        train_tmp['x'] = np.reshape(train_tmp['x'], (len(train_tmp['x']), 5, 65, 1))

    # Define test_tmp
    test_tmp = {}

    # Check if test_r and test_b exist
    if 'test_r' not in locals() and 'test_b' not in locals():
        test_tmp = test_s
    elif 'test_r' not in locals():
        test_tmp['x'] = np.concatenate((test_s['x'], test_b['x']), axis=0)
        test_tmp['info'] = np.concatenate((test_s['info'], test_b['info']), axis=2)
    elif 'test_b' not in locals():
        test_tmp['x'] = np.concatenate((test_s['x'], test_r['x']), axis=0)
        test_tmp['info'] = np.concatenate((test_s['info'], test_r['info']), axis=2)
    else:
        test_tmp['x'] = np.concatenate((test_s['x'], test_r['x'], test_b['x']), axis=0)
        test_tmp['info'] = np.concatenate((test_s['info'], test_r['info'], test_b['info']), axis=2)
    # Train matrix
    inf = pd.DataFrame(train_tmp['info'][2, :, :].T, columns=["chr.str", "pos", "REFbase", "kmer", "cov"])
    inf['ind'] = inf.index + 1
    inf['id'] = inf['chr.str'].astype(int).astype(str) + ':' + inf['pos'].astype(int).astype(str)

    df1 = pd.merge(inf, train_pos, on='id', how='left')
    df1 = df1.dropna()
    df1 = df1.sort_values(by='ind')

    train_matrix = {}
    train_matrix['y'] = df1['edit'].values
    train_matrix['x'] = train_tmp['x'][df1['ind'] - 1, :, :, :]
    train_matrix['x'] = np.reshape(train_matrix['x'], (len(train_matrix['y']), 5, 65, 1))
    train_matrix['info'] = df1[["chr.str", "pos", "REFbase", "kmer", "cov", "ind", "id"]].values
    train_matrix['y2'] = df1['rate'].values
    # Test matrix
    inf = pd.DataFrame(test_tmp['info'][2, :, :].T, columns=["chr.str", "pos", "REFbase", "kmer", "cov"])
    inf['ind'] = inf.index + 1
    inf['id'] = inf['chr.str'].astype(int).astype(str) + ':' + inf['pos'].astype(int).astype(str)

    df1 = pd.merge(inf, test_pos, on='id', how='left')
    df1 = df1.dropna()
    df1 = df1.sort_values(by='ind')

    test_matrix = {}
    test_matrix['y'] = df1['edit'].values
    test_matrix['x'] = test_tmp['x'][df1['ind'] - 1, :, :, :]
    test_matrix['x'] = np.reshape(test_matrix['x'], (len(test_matrix['y']), 5, 65, 1))
    test_matrix['info'] = df1[["chr.str", "pos", "REFbase", "kmer", "cov", "ind", "id"]].values
    test_matrix['y2'] = df1['rate'].values

    # Save matrices
    output_dir = args.out
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "test_matrix.npy"), test_matrix)
    np.save(os.path.join(output_dir, "train_matrix.npy"), train_matrix)
    print("Done")

if __name__ == '__main__':
    main()

