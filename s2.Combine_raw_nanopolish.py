import argparse
import os
import warnings
import numpy as np
import pandas as pd
import time
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')
no_cores = os.cpu_count() - 1

parser = argparse.ArgumentParser(description='Nanopore data')
parser.add_argument('-f','--file', default='./small_data/tmp.nnpl', type=str,
                    help='Event file name from Nanopolish',metavar="character")
parser.add_argument('-t','--thread', default=24, type=int,
                    help='Number of cores allocated',metavar="integer")
parser.add_argument('-o','--out', default='./small_data/xen_s9_r1_50k.gm2.nanopolish_eventAlignOut_combined.txt.part1', type=str,
                    help='output file name',metavar="character")
parser.add_argument('-s','--size', default=141548983, type=str,
                    help='number of line in input',metavar="character")
parser.add_argument('-n','--num', default=2, type=int,
                    help='order number of line in input',metavar="integer")
parser.add_argument('-c','--chunk', default=10000000, type=int,
                    help='chunk size',metavar="integer")
args = parser.parse_args()

if (args.file is None) or (args.size is None) or (args.num is None):
    print(args.print_help())
    warnings.warn('Input file and size of input must be supplied. ')

no_cores = args.thread
import math

def sd_c(x_m, x_s, x_n, y_m, y_s, y_n):
    al = x_n + y_n
    tmp_sd = al * ((x_n - 1) * (x_s * x_s) + (y_n - 1) * (y_s * y_s)) + y_n * x_n * (x_m - y_m) * (x_m - y_m)
    var = tmp_sd / (al * (al - 1))
    return np.sqrt(var)
def sd_d(sd, mean, freq):
    x1 = len(mean)

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

def process_data(names,dask_dat):
    filtered_data = dask_dat[dask_dat['read_name'].isin(names)]
    agg_dict = {
        'event_stdv': lambda x: sd_d(x.values, filtered_data.loc[x.index, 'event_level_mean'].values,
                                          filtered_data.loc[x.index, 'count'].values.astype(int)),
        'event_level_mean': lambda x: np.average(x, weights=filtered_data.loc[x.index, 'count']),
        'count': 'sum',
        'reference_kmer': lambda x: x.unique().tolist()[0]
    }
    return filtered_data.groupby(['contig', 'read_name', 'position']).agg(agg_dict).reset_index()
def main():
    head = pd.read_csv("./small_data/raw_nanopolish.header", sep="\t")
    n = args.num
    ch = args.chunk
    ind = []
    args.size = int(args.size)

    dat = pd.read_csv(args.file, header=None, sep="\t",nrows=1000000)
    print("loaded data")
    dat.columns = head.columns
    if ((n + ch) < args.size):
        read = dat['read_name'].iloc[-1]
        pos = dat['position'].iloc[-1]
        ind = dat[(dat['position'] == pos) & (dat['read_name'] == read)].index.tolist()
        dat = dat.drop(ind)

    dat = dat[(dat['strand'] == 't') & (dat['reference_kmer'] != 'NNNNN') & (dat['event_stdv'] < 50)]
    dat['count'] = round(3012 * dat['event_length'])

    dat_name = dat['read_name'].unique()
    split_size = len(dat_name) // (no_cores + 20)
    split_indices = np.arange(0, len(dat_name) - 1, split_size + 1)
    name_list = np.split(dat_name, split_indices[1:])
    print("begin time is:" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # Parallel processing using joblib
    dat_com_chunks = Parallel(n_jobs=-1,backend='loky')(delayed(process_data)(name_chunk,dat) for name_chunk in name_list)
    final_result = pd.concat(dat_com_chunks, ignore_index=True)
    print("end time is:" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    final_result.to_csv(args.out, sep='\t', header=False, index=False)
    if len(ind) > 0:
        with open("tmp.eli", "w") as file:
            file.write(str(len(ind)))

if __name__ == '__main__':
    main()


