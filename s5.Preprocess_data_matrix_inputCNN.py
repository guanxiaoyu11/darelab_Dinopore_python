
import argparse
import os
import warnings
import csv
from tqdm import tqdm
import numpy as np
import pandas as pd
import pyreadr
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
    x = x.copy()
    x[['chr', 'position', 'strand']] = x['x'].str.split(':', expand=True)
    x[['po.le1', 'po.le2', 'po.la1', 'po.la2']] = x['position'].astype(int).values[:, None] + np.array([1, 2, -1, -2])
    x[['po.le.id1', 'po.la.id1', 'po.le.id2', 'po.la.id2']] = x[['chr', 'po.le1', 'strand']].apply(lambda row: ':'.join(row.astype(str)), axis=1)
    return {'ld1': x['po.le.id1'], 'lg1': x['po.la.id1'], 'ld2': x['po.le.id2'], 'lg2': x['po.la.id2']}

def replace0(x):
    cols_to_replace = ['event_stdv.m', 'event_stdv.s', 'event_stdv.A', 'count.m', 'count.s', 'ins3.m', 'ins1.m', 'ins2.m', 'ins4.m', 'ins5.m', 'ins3.s', 'ins1.s', 'ins2.s', 'ins4.s', 'ins5.s']
    x[cols_to_replace] = np.where(x[cols_to_replace] == 0, 0.00001, x[cols_to_replace])
    return x

def replace_list(pat, repl, x):
    y = [repl[pat[0].index(ch)] if ch in pat else ch for ch in x]
    return y

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

def countnb5_optimized(xt):
    mask1 = (xt['REF'] == "A") & (xt['strand'] == "p")
    mask2 = (xt['REF'] == "T") & (xt['strand'] == "n")

    k = (xt['position'].diff() == 1).astype(int).cumsum()
    h = (xt['position'].diff() == -1).astype(int).cumsum()
    m = (xt['position'].diff() == 2).astype(int).cumsum()
    n = (xt['position'].diff() == -2).astype(int).cumsum()

    xt['cld1'] = k.where(mask1 | mask2, 0)
    xt['clg1'] = h.where(mask1 | mask2, 0)
    xt['cld2'] = m.where(mask1 | mask2, 0)
    xt['clg2'] = n.where(mask1 | mask2, 0)

    return xt

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
    filepath = "./test/minmax_Xen_5pos_012_10bin_morefts_combine.RData"
    result = pyreadr.read_r(filepath)
    mm_df = result['mm.df']
    print(args.input)
    file_df = pd.read_csv(args.input, header=0,sep='\t', quoting=csv.QUOTE_NONE,nrows=1000000)

    dat = file_df[file_df['reference_kmer'] != ""]
    dat=replace0(dat)

    dat[["event_stdv.m","event_stdv.s","event_stdv.A","count.m","count.s","ins3.m","ins1.m","ins2.m","ins4.m","ins5.m","ins3.s","ins1.s","ins2.s","ins4.s","ins5.s"]] = \
        np.log(dat[["event_stdv.m","event_stdv.s","event_stdv.A","count.m","count.s","ins3.m","ins1.m","ins2.m","ins4.m","ins5.m","ins3.s","ins1.s","ins2.s","ins4.s","ins5.s"]])
    dat = dat.drop(dat.columns[list(range(29, 34)) + list(range(50, 55))], axis=1)
    dat = dat.dropna().reset_index(drop=True)
    col = dat.columns[7:72]

    for i in range(len(col)):
        col_name = col[i]
        min_value = mm_df.loc[mm_df['col'] == col_name, 'min'].values[0]
        diff_value = mm_df.loc[mm_df['col'] == col_name, 'diff'].values[0]
        dat[col_name] = ((dat[col_name] - min_value) * 100) / diff_value

    checkdf = check_mono(dat, 71)


    chroms = np.sort(checkdf['contig'].unique())
    chromid = [str(i + 1) for i in range(len(chroms))][0]
    chrmapping = pd.DataFrame({'chroms': chroms, 'chromid': chromid})
    chrmapping.to_pickle("chrmapping.pkl")

    test_s = transform_dat(checkdf, dat.copy(),chroms,chromid)
    test_r = transform_dat_big(checkdf, dat.copy(),chroms,chromid)
    test_b = transform_dat_bb(checkdf, dat.copy(),chroms,chromid)

    test_tmp = {}

    if 'test_r' not in locals():
        if 'test_b' not in locals():
            test_tmp = test_s
        else:
            test_tmp['x'] = np.concatenate((test_s['x'], test_b['x']), axis=0)
            test_tmp['y'] = np.concatenate((test_s['y'], test_b['y']), axis=0)
            test_tmp['info'] = np.concatenate((test_s['info'], test_b['info']), axis=2)
    else:
        if 'test_b' not in locals():
            test_tmp['x'] = np.concatenate((test_s['x'], test_r['x']), axis=0)
            test_tmp['y'] = np.concatenate((test_s['y'], test_r['y']), axis=0)
            test_tmp['info'] = np.concatenate((test_s['info'], test_r['info']), axis=2)
        else:
            test_tmp['x'] = np.concatenate((test_s['x'], test_r['x'], test_b['x']), axis=0)
            # test_tmp['y'] = np.concatenate((test_s['y'], test_r['y'], test_b['y']), axis=0)
            test_tmp['info'] = np.concatenate((test_s['info'], test_r['info'], test_b['info']), axis=2)

    info = pd.DataFrame(test_tmp['info'][2, :, :]).T
    info.columns = ['chr.str', 'pos', 'REFbase', 'kmer', 'cov']
    info['ind'] = info.index + 1
    info['id'] = info['chr.str'].astype(int).astype(str) + ':' + info['pos'].astype(int).astype(str)

    df1 = pd.read_csv(args.classref, header=0,sep='\t')
    df2 = pd.merge(info, df1, on='id', how='left')

    df2 = df2.dropna().reset_index(drop=True)
    df2 = df2.sort_values(by='ind').reset_index(drop=True)

    test = {}

    test['y'] = df2['ref'].values
    test['x'] = test_tmp['x'][df2['ind'], :, :, None]
    test['x'] = np.reshape(test['x'], (len(test['y']), 5, 65, 1))
    test['info'] = df2.iloc[:, 1:8].values
    test['y2'] = df2['rate'].values

    # Save to file
    np.savez(args.out, **test)
    print("Done")
if __name__ == '__main__':
    main()