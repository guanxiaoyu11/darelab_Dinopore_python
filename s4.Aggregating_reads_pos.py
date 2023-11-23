import argparse
import os
import gc
import warnings
import glob
from multiprocessing import Pool
import csv
from functools import partial
import sys
import time
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import math
import concurrent.futures

warnings.filterwarnings('ignore')
warnings.simplefilter("ignore", category=RuntimeWarning)
no_cores = os.cpu_count()
parser = argparse.ArgumentParser(description='Nanopore data')

parser.add_argument('-t','--thread', default=24, type=int,
                    help='Number of cores allocated',metavar="integer")
parser.add_argument('-o','--out', default='xen50k.Agg.morefts.10bin.inML.txt', type=str,
                    help='output file name',metavar="character")
parser.add_argument('-r','--regex', default='xen50k', type=str,
                    help='output file name',metavar="character")

# fakeArgs = ['-f','--file','-t','--thread','-o','--out','-s','--size','-n','--num','-c','--chunk']
args = parser.parse_args()

if (args.out is None) or (args.regex is None):
    print(args.print_help())
    warnings.warn('Input files must be supplied (-o,-r). ')

dinodir=os.path.dirname(os.path.abspath(__file__))

###Functions###
# def impute_mean(x):
#     x = np.array(x)
#     x[np.isnan(x)] = np.mean(x[~np.isnan(x)])
#     return x.tolist()
def impute_mean(x):
     return x.fillna(np.mean(x))
def changenan(x):
    return x.fillna(' ')
def entr(x):
    y = [val * np.log2(val) if val != 0 else 0 for val in x]
    return y
def sd_c(x_m, x_s, x_n, y_m, y_s, y_n):
    al = x_n + y_n
    tmp_sd = al * ((x_n - 1) * (x_s * x_s) + (y_n - 1) * (y_s * y_s)) + y_n * x_n * (x_m - y_m) * (x_m - y_m)
    var = tmp_sd / (al * (al - 1))
    return math.sqrt(var)
def sd_d(sd, mean, freq):
    x1 = len(mean)
    if x1 == 0:
        return None
    if x1 == 1:
        sx = sd[0]
        return sx

    mx = mean[0]
    sx = sd[0]
    nx = freq[0]

    for i in range(1, x1):
        my = mean[i]
        sy = sd[i]
        ny = freq[i]

        sx = sd_c(mx, sx, nx, my, sy, ny)
        mx = (mx * nx + my * ny) / (nx + ny)
        nx = nx + ny

    return sx

def mean_c(mean, freq):
    x1 = len(mean)
    if x1 == 0:
        return 0
    if x1 == 1:
        return mean[0]

    mx = mean[0]
    nx = freq[0]

    for i in range(1, x1):
        my = mean[i]
        ny = freq[i]

        mx = (mx * nx + my * ny) / (nx + ny)
        nx = nx + ny

    return mx

def indexgroup10(ind, sizevector):
    bin_size = sizevector // 10
    for r in range(bin_size):
        for i in range(r*10, (r+1)*10):
            ind[i] = r+1
    for k in range((bin_size-1)*10, sizevector):
        ind[k] = bin_size
    return ind



def indexgroup_big(ind, sizevector):
    bin = 10
    b = sizevector // 10.0
    nr = int(b)
    for r in range(bin):
        for i in range(r * nr, (r + 1) * nr):
            ind[i] = r + 1
    for k in range((bin - 1) * nr, sizevector):
        ind[k] = bin
    return ind

def process_data(read_names, dat,group_id):
    result= []
    filtered_data = dat[dat['posi_id'].isin(read_names)]
    filtered_data[['event_level_mean', 'event_stdv', 'count']] = filtered_data.groupby('posi_id')[
        ['event_level_mean', 'event_stdv', 'count']].apply(impute_mean).reset_index(drop=True)
    # filtered_data[['reference_kmer']]= filtered_data.groupby('posi_id')[
    #     ['reference_kmer']].apply(impute_mean)
    columns_to_zero = [
        'count.m', 'count.ms', 'count.rms', 'count.s', 'cov',
        'del1.m', 'del2.m', 'del3.m', 'del4.m', 'del5.m',
        'del1.s', 'del2.s', 'del3.s', 'del4.s', 'del5.s',
        'event_level_mean.A', 'event_level_mean.m', 'event_level_mean.ms',
        'event_level_mean.rms', 'event_level_mean.s', 'event_stdv.A',
        'event_stdv.m', 'event_stdv.ms', 'event_stdv.rms','event_stdv.s', 'ins1.m',
        'ins1.s', 'ins2.m', 'ins2.s', 'ins3.m', 'ins3.s', 'ins4.m',
        'ins4.s', 'ins5.m', 'ins5.s', 'mis1.m', 'mis1.s', 'mis2.m',
        'mis2.s', 'mis3.m', 'mis3.s', 'mis4.m', 'mis4.s', 'mis5.m',
        'mis5.s', 'qual1.m', 'qual1.ms', 'qual1.s', 'qual1.rms',
        'qual2.m', 'qual2.ms', 'qual2.s', 'qual2.rms', 'qual3.m',
        'qual3.ms', 'qual3.s', 'qual3.rms', 'qual4.m', 'qual4.ms',
        'qual4.s', 'qual4.rms', 'qual5.m', 'qual5.ms', 'qual5.s',
        'qual5.rms'
    ]
    for col in columns_to_zero:
        filtered_data.loc[:,col] = 0
    agg_dict ={
        # 'reference_kmer': lambda x: '' if not x.dropna().unique().tolist() else x.dropna().unique().tolist()[0],
        'reference_kmer': lambda x: x.unique().tolist(),
        'cov' : 'size',
        'event_stdv.m' : lambda x: np.nanmean(filtered_data.loc[x.index, 'event_stdv'].dropna().values),
        'event_level_mean.m' : lambda x: np.nanmean(filtered_data.loc[x.index, 'event_level_mean'].dropna().values),
        'event_stdv.s' : lambda x:filtered_data.loc[x.index, 'event_stdv'].dropna().values.std(ddof=1),
        'event_level_mean.s' : lambda x:filtered_data.loc[x.index, 'event_level_mean'].dropna().values.std(ddof=1),
        'event_stdv.A': lambda x: sd_d(filtered_data.loc[x.index, 'event_stdv'].dropna().values, filtered_data.loc[x.index, 'event_level_mean'].dropna().values,
                                      filtered_data.loc[x.index, 'count'].dropna().values),
        'event_level_mean.A' : lambda x: mean_c(filtered_data.loc[x.index, 'event_level_mean'].values,filtered_data.loc[x.index, 'count'].values),
        'count.m' : lambda x: np.nanmean(filtered_data.loc[x.index, 'count'].dropna().values),
        'qual1.m' : lambda x: np.nanmean(filtered_data.loc[x.index, 'qual1'].dropna().values),
        'qual2.m' : lambda x: np.nanmean(filtered_data.loc[x.index, 'qual2'].dropna().values),
        'qual3.m' :lambda x: np.nanmean(filtered_data.loc[x.index, 'qual'].dropna().values),
        'qual4.m' : lambda x: np.nanmean(filtered_data.loc[x.index, 'qual4'].dropna().values),
        'qual5.m' : lambda x: np.nanmean(filtered_data.loc[x.index, 'qual5'].dropna().values),
        'del1.m' : lambda x: np.nanmean(filtered_data.loc[x.index, 'del1'].dropna().values),
        'del2.m' : lambda x: np.nanmean(filtered_data.loc[x.index, 'del2'].dropna().values),
        'del3.m' : lambda x: np.nanmean(filtered_data.loc[x.index, 'del'].dropna().values),
        'del4.m' : lambda x: np.nanmean(filtered_data.loc[x.index, 'del4'].dropna().values),
        'del5.m' : lambda x: np.nanmean(filtered_data.loc[x.index, 'del5'].dropna().values),
        'mis1.m' : lambda x: np.nanmean(filtered_data.loc[x.index, 'mis1'].dropna().values),
        'mis2.m' : lambda x: np.nanmean(filtered_data.loc[x.index, 'mis2'].dropna().values),
        'mis3.m' : lambda x: np.nanmean(filtered_data.loc[x.index, 'mis'].dropna().values),
        'mis4.m' : lambda x: np.nanmean(filtered_data.loc[x.index, 'mis4'].dropna().values),
        'mis5.m' : lambda x: np.nanmean(filtered_data.loc[x.index, 'mis5'].dropna().values),
        'ins1.m' : lambda x: np.nanmean(filtered_data.loc[x.index, 'ins1'].dropna().values),
        'ins2.m' : lambda x: np.nanmean(filtered_data.loc[x.index, 'ins2'].dropna().values),
        'ins3.m' : lambda x: np.nanmean(filtered_data.loc[x.index, 'ins'].dropna().values),
        'ins4.m' : lambda x: np.nanmean(filtered_data.loc[x.index, 'ins4'].dropna().values),
        'ins5.m' : lambda x: np.nanmean(filtered_data.loc[x.index, 'ins5'].dropna().values),
        'count.s' : lambda x:np.nanstd(filtered_data.loc[x.index, 'count'].dropna().values,ddof=1),
        'qual1.s' : lambda x:np.nanstd(filtered_data.loc[x.index, 'qual1'].dropna().values,ddof=1),
        'qual2.s' : lambda x:np.nanstd(filtered_data.loc[x.index, 'qual2'].dropna().values,ddof=1),
        'qual3.s' : lambda x:np.nanstd(filtered_data.loc[x.index, 'qual'].dropna().values,ddof=1),
        'qual4.s' : lambda x:np.nanstd(filtered_data.loc[x.index, 'qual4'].dropna().values,ddof=1),
        'qual5.s' : lambda x:np.nanstd(filtered_data.loc[x.index, 'qual5'].dropna().values,ddof=1),
        'del1.s' : lambda x:np.nanstd(filtered_data.loc[x.index, 'del1'].dropna().values,ddof=1),
        'del2.s' : lambda x:np.nanstd(filtered_data.loc[x.index, 'del2'].dropna().values,ddof=1),
        'del3.s' : lambda x:np.nanstd(filtered_data.loc[x.index, 'del'].dropna().values,ddof=1),
        'del4.s' : lambda x:np.nanstd(filtered_data.loc[x.index, 'del4'].dropna().values,ddof=1),
        'del5.s' : lambda x:np.nanstd(filtered_data.loc[x.index, 'del5'].dropna().values,ddof=1),
        'mis1.s' : lambda x:np.nanstd(filtered_data.loc[x.index, 'mis1'].dropna().values,ddof=1),
        'mis2.s' : lambda x:np.nanstd(filtered_data.loc[x.index, 'mis2'].dropna().values,ddof=1),
        'mis3.s' : lambda x:np.nanstd(filtered_data.loc[x.index, 'mis'].dropna().values,ddof=1),
        'mis4.s' : lambda x:np.nanstd(filtered_data.loc[x.index, 'mis4'].dropna().values,ddof=1),
        'mis5.s' : lambda x:np.nanstd(filtered_data.loc[x.index, 'mis5'].dropna().values,ddof=1),
        'ins1.s' : lambda x:np.nanstd(filtered_data.loc[x.index, 'ins1'].dropna().values,ddof=1),
        'ins2.s' : lambda x:np.nanstd(filtered_data.loc[x.index, 'ins2'].dropna().values,ddof=1),
        'ins3.s' : lambda x:np.nanstd(filtered_data.loc[x.index, 'ins'].dropna().values,ddof=1),
        'ins4.s' : lambda x:np.nanstd(filtered_data.loc[x.index, 'ins4'].dropna().values,ddof=1),
        'ins5.s' : lambda x:np.nanstd(filtered_data.loc[x.index, 'ins5'].dropna().values,ddof=1),
        'event_stdv.ms' : lambda x: np.nanmean(filtered_data.loc[x.index, 'event_stdv'].dropna().values ** 2),
        'event_level_mean.ms' : lambda x: np.nanmean(filtered_data.loc[x.index, 'event_level_mean'].dropna().values** 2),
        'count.ms' : lambda x: np.nanmean(filtered_data.loc[x.index, 'count'].dropna().values ** 2),
        'qual1.ms' : lambda x: np.nanmean(filtered_data.loc[x.index, 'qual1'].dropna().values ** 2),
        'qual2.ms' : lambda x: np.nanmean(filtered_data.loc[x.index, 'qual2'].dropna().values ** 2),
        'qual3.ms' : lambda x: np.nanmean(filtered_data.loc[x.index, 'qual'].dropna().values ** 2),
        'qual4.ms' : lambda x: np.nanmean(filtered_data.loc[x.index, 'qual4'].dropna().values ** 2),
        'qual5.ms' : lambda x: np.nanmean(filtered_data.loc[x.index, 'qual5'].dropna().values ** 2),
        'event_stdv.rms' : lambda x: np.sqrt(np.nanmean(filtered_data.loc[x.index, 'event_stdv'].dropna().values ** 2)),
        'event_level_mean.rms' : lambda x:np.sqrt(np.nanmean(filtered_data.loc[x.index, 'event_level_mean'].dropna().values ** 2)),
        'count.rms' : lambda x:np.sqrt(np.nanmean(filtered_data.loc[x.index, 'count'].dropna().values ** 2)),
        'qual1.rms' : lambda x:np.sqrt(np.nanmean(filtered_data.loc[x.index, 'qual1'].dropna().values ** 2)),
        'qual2.rms' : lambda x:np.sqrt(np.nanmean(filtered_data.loc[x.index, 'qual2'].dropna().values ** 2)),
        'qual3.rms' : lambda x:np.sqrt(np.nanmean(filtered_data.loc[x.index, 'qual'].dropna().values ** 2)),
        'qual4.rms' : lambda x:np.sqrt(np.nanmean(filtered_data.loc[x.index, 'qual4'].dropna().values ** 2)),
        'qual5.rms' :lambda x: np.sqrt(np.nanmean(filtered_data.loc[x.index, 'qual5'].dropna().values ** 2)),
    }
    # Apply the aggregations
    if group_id==0:
        grouped_data = filtered_data.groupby(["contig", "position", "strand", "REF", "posi_id"]).agg(agg_dict).reset_index()
    else:
        grouped_data = filtered_data.groupby(["contig", "position", "strand", "REF", "posi_id","index"]).agg(
            agg_dict).reset_index()
    check_status = []
    ref = []
    for kmer_list in grouped_data['reference_kmer']:
        if len(kmer_list) > 1:
            check_status.append(True)
            for j in kmer_list:
                if type(j) is str:
                    ref.append(j)
        else:
            check_status.append(False)
            ref.append('')

    grouped_data['reference_kmer'] = ref  # Convert to list of lists
    # Filter grouped_data1 based on check_status
    grouped_data1 = grouped_data[check_status].copy()
    grouped_datas=pd.concat([grouped_data1,grouped_data]).sort_values(by=["position"]).reset_index(drop=True)
    result.append(grouped_datas)
    return pd.concat(result)
def main():
    ###Start###
    files = glob.glob('*_grp' + args.regex)
    list_df = []
    print("loading begin time is:" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for file in files:
        # df = pd.read_csv(file, header=0,sep='\t', quoting=csv.QUOTE_NONE)
        # df.to_pickle(file+'.pkl')
        # df = pd.read_pickle(file,nrow=100000)
        # df.to_hdf(file+'.hdf', 'df')
        df = pd.read_hdf(file+'.hdf', 'df')
        list_df.append(df)
    df = pd.concat(list_df).drop_duplicates()
    del list_df

    df['REF'] = df['REF'].str.upper()
    df['event_level_mean'] = pd.to_numeric(df['event_level_mean'])
    df['event_stdv'] = pd.to_numeric(df['event_stdv'])
    df['count'] = pd.to_numeric(df['count'])
    print("loading end time is:" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("Done loading")

    ######calculate entropy####
    print("entropy begin time is:" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    A_df = df[df['BASE'] == "A"].groupby('posi_id').size().reset_index(name='cov.A')
    G_df = df[df['BASE'] == "G"].groupby('posi_id').size().reset_index(name='cov.G')
    T_df = df[df['BASE'] == "T"].groupby('posi_id').size().reset_index(name='cov.T')
    C_df = df[df['BASE'] == "C"].groupby('posi_id').size().reset_index(name='cov.C')
    df_Total = df.groupby('posi_id').size().reset_index(name='total').reset_index(drop=True)

    dat_com = pd.merge(df_Total, A_df, on='posi_id', how='outer').reset_index(drop=True)
    dat_com = pd.merge(dat_com, G_df, on='posi_id', how='outer').reset_index(drop=True)
    dat_com = pd.merge(dat_com, T_df, on='posi_id', how='outer').reset_index(drop=True)
    dat_com = pd.merge(dat_com, C_df, on='posi_id', how='outer').reset_index(drop=True)

    dat_com.fillna(0, inplace=True)

    dat_com['pct.A'] = dat_com['cov.A'] / dat_com['total']
    dat_com['pct.G'] = dat_com['cov.G'] / dat_com['total']
    dat_com['pct.C'] = dat_com['cov.C'] / dat_com['total']
    dat_com['pct.T'] = dat_com['cov.T'] / dat_com['total']

    dat_com['entr.A'] = entr(dat_com['pct.A'])
    dat_com['entr.G'] = entr(dat_com['pct.G'])
    dat_com['entr.C'] = entr(dat_com['pct.C'])
    dat_com['entr.T'] = entr(dat_com['pct.T'])
    dat_com['entropy'] = -(dat_com[['entr.A', 'entr.G', 'entr.C', 'entr.T']].sum(axis=1, skipna=True))
    dat_com = dat_com[dat_com['total'] >= 6]
    del A_df, G_df, C_df, T_df
    print("entropy end time is:" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("Done entropy")
    ##########################

    ####aggregating####
    print("under begin time is:" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    cov = df.groupby('posi_id').size().reset_index(name='n')
    cov = cov[cov['n'] >= 6].reset_index(drop=True)
    under_posi = cov[cov['n'] < 10].reset_index(drop=True)
    over_posi1 = cov[(cov['n'] >= 20) & (cov['n'] < 110)].reset_index(drop=True)
    over_posi2 = cov[cov['n'] >= 110].reset_index(drop=True)

    dat_under = df[df['posi_id'].isin(under_posi['posi_id'])]
    import dask.dataframe as dd

    dat_under_dask = dd.from_pandas(dat_under, npartitions=len(dat_under)//100)
    dat_under2_dask = dat_under_dask.groupby('posi_id').apply(lambda x: x.sample(n=10 - len(x)) if len(x) < 10 else x,
                                                              meta=dat_under)
    dat_under2 = dat_under2_dask.compute()

    del dat_under
    print("under end time is:" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("Done under")

    dat_over = df[df['posi_id'].isin(over_posi1['posi_id']) | df['posi_id'].isin(over_posi2['posi_id'])]
    # dat_over = dat_over.sample(frac=1).reset_index(drop=True)
    dat_over = dat_over.sample(n=dat_over.shape[0], replace=False).reset_index(drop=True)
    dat_over2 = pd.merge(dat_over[dat_over['posi_id'].isin(over_posi1['posi_id'])], over_posi1, on='posi_id',
                         how='left')
    dat_over3 = pd.merge(dat_over[dat_over['posi_id'].isin(over_posi2['posi_id'])], over_posi2, on='posi_id',
                         how='left')

    dat_over2['index'] = 0
    dat_over3['index'] = 0
    group=dat_over2.groupby('posi_id')
    indexed_groups =group.apply(lambda x: indexgroup10(x['index'].values, x['n'].unique()[0]))
    indexed_groups = indexed_groups.reset_index(level=0, drop=True)
    dat_over2_sorted = dat_over2.sort_values(by='posi_id').copy()
    dat_over2_sorted['index'] = np.concatenate(indexed_groups.values)
    dat_over2 = dat_over2_sorted.sort_index().reset_index(drop=True)
    group = dat_over3.groupby('posi_id')
    indexed_groups = group.apply(lambda x: indexgroup_big(x['index'].values, x['n'].unique()[0]))
    indexed_groups = indexed_groups.reset_index(level=0, drop=True)
    dat_over3_sorted = dat_over3.sort_values(by='posi_id').copy()
    dat_over3_sorted['index'] = np.concatenate(indexed_groups.values)
    dat_over3 = dat_over3_sorted.sort_index().reset_index(drop=True)
    print("Done over")

    df = df[~df['posi_id'].isin(over_posi1['posi_id']) & ~df['posi_id'].isin(over_posi2['posi_id'])]
    df = df[df['posi_id'].isin(cov['posi_id'])]
    df = pd.concat([df, dat_under2])

    dat_name = df['posi_id'].unique()
    split_size = len(dat_name) // no_cores
    split_indices = np.arange(0, len(dat_name) - 1, split_size + 1)
    name_list1 = np.split(dat_name, split_indices[1:])
    process_data(name_list1[0],df,0)
    print("undersampling begin time is:" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    with concurrent.futures.ProcessPoolExecutor(max_workers=no_cores) as executor:
        results = [executor.submit(process_data, names, df,0) for names in name_list1]
    df1 = pd.concat([result.result() for result in results]).reset_index(drop=True)
    print("undersampling end time is:" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    del df
    print("done aggregate undersampling")


    dat_over = pd.concat([dat_over2, dat_over3], ignore_index=True)
    dat_over.sort_values(by=["position"])
    dat_name = np.sort(dat_over['posi_id'].unique())
    split_size = len(dat_name) // no_cores
    split_indices = np.arange(0, len(dat_name) - 1, split_size + 1)
    # Split the list of names
    name_list2 = np.split(dat_name, split_indices[1:])
    print("aggregating begin time is:" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    with concurrent.futures.ProcessPoolExecutor(max_workers=no_cores) as executor:
        results = [executor.submit(process_data, names, dat_over,1) for names in name_list2]
    df2 = pd.concat([result.result() for result in results]).reset_index(drop=True)
    print("aggregating end time is:" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    df2 = df2.drop(columns=['index'])
    del dat_over
    print("Done aggregating")
    df_com = pd.concat([df1, df2])
    df_com = df_com.drop(columns=['cov'])
    df_com = df_com.rename(columns={'cov1': 'cov'})
    dat_com = dat_com.rename(columns={'total': 'cov'})

    all_data = pd.merge(df_com, dat_com.loc[:, ["posi_id", "cov", "pct.A", "pct.G", "pct.C", "pct.T", "entropy"]],
                        on='posi_id', how='inner')
    reference_kmer_index = all_data.columns.get_loc('reference_kmer')
    all_data.insert(reference_kmer_index + 1, 'cov', all_data.pop('cov'))
    selected_columns = all_data.iloc[:, list(range(0, 55)) + list(range(66, 76)) + list(range(55, 66))]
    all_data = selected_columns.copy()
    all_data['ms.pct'] = np.sum(all_data.iloc[:, 55:59].dropna(), axis=1) / 4
    all_data['rms.pct'] = np.sqrt(np.sum(all_data.iloc[:, 55:59].dropna(), axis=1) / 4)
    all_data['ms.mis'] = np.sum(all_data.iloc[:, 24:29].dropna(), axis=1) / 5
    all_data['rms.mis'] = np.sqrt(np.sum(all_data.iloc[:, 24:29].dropna(), axis=1) / 5)
    all_data['ms.del'] = np.sum(all_data.iloc[:, 19:24].dropna(), axis=1) / 5
    all_data = all_data[all_data['reference_kmer'] != ""]
    print("Start writing")
    all_data.to_csv(args.out, sep='\t', index=False)
    print("Done")
if __name__ == '__main__':
    main()