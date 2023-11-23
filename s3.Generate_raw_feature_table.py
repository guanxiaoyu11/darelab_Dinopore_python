import argparse
import os
import gc
import warnings
from multiprocessing import Pool
import csv
from tqdm import tqdm
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import time
warnings.filterwarnings('ignore')
# no_cores <- detectCores()-1
def myascii(qual):
    qualn = [ord(char) - 1 for char in qual]
    return qualn
def mymatmis(op, ref, base):
    is_M = op == 'M'
    is_D = op == 'D'

    mat = np.where(is_M & (ref == base), 1, 0)
    mis = np.where(is_M & (ref != base), 1, 0)
    del_ = np.where(is_D, 1, 0)

    return mat.tolist(), mis.tolist(), del_.tolist()

def insertfunP(op, inse):
    indices_I = np.where(op == 'I')[0]
    consecutive = np.split(indices_I, np.where(np.diff(indices_I) != 1)[0] + 1)
    values = np.array([np.flip(np.arange(1, len(arr) + 1)) for arr in consecutive])
    indices = np.concatenate(consecutive)-1
    inse[indices] = np.concatenate(values)
    return inse
def process_data(read_names, dat):
    grouped_data = dat[dat['read_name'].isin(read_names)]
    filtered_data=grouped_data.copy()
    columns = ["qual", "del", "mis", "mat", "ins"]
    for i in range(len(columns)):
        filtered_data.loc[:,columns[i] + '1'] = filtered_data.groupby('read_name')[columns[i]].shift(2)
        filtered_data.loc[:,columns[i] + '2'] = filtered_data.groupby('read_name')[columns[i]].shift(1)
        filtered_data.loc[:,columns[i] + '4'] = filtered_data.groupby('read_name')[columns[i]].shift(-1)
        filtered_data.loc[:,columns[i] + '5'] = filtered_data.groupby('read_name')[columns[i]].shift(-2)
    return filtered_data
def insertfunN(op, inse):
    indices_I = np.where(op == 'I')[0]
    consecutive = np.split(indices_I, np.where(np.diff(indices_I) != 1)[0] + 1)
    values = np.array([np.arange(1, len(arr) + 1) for arr in consecutive])
    indices = np.concatenate(consecutive)+1
    inse[indices] = np.concatenate(values)
    return inse
# def insertfunN(op, inse):
#     for i in range(len(inse)):
#         if op[i] == "I":
#             k = 1
#             while i - k >= 0 and op[i - k] == "I":
#                 k += 1
#             if i + 1 < len(inse):
#                 inse.at[i + 1] = k
#     return inse

def complement_char(base):
    complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    base1 = [complement_dict.get(b, 'na') for b in base]
    return base1

def rev_comp(kmer):
    complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    kmer1 = []
    for seq in kmer:
        complemented_seq = ''.join([complement_dict.get(base, 'N') for base in seq])
        reversed_seq = complemented_seq[::-1]
        kmer1.append(reversed_seq)
    return kmer1

parser = argparse.ArgumentParser(description='Nanopore data')
parser.add_argument('-n','--nnpl', default='small_data/xen_s9_r1_50k.input.nanopolish_eventAlignOut_combined.txt', type=str,
                    help='nanopolish',metavar="character")
parser.add_argument('-t','--tsv', default='small_data/xen_s9_r1_50k.input.filteredSHN.tsv.txt', type=str,
                    help='tsv',metavar="character")
parser.add_argument('-o','--out', default='xen_s9_r1_50k.tsv_nnpl_inAE.txt.part1', type=str,
                    help='output file name',metavar="character")

args = parser.parse_args()

if (args.nnpl is None) or (args.tsv is None):
    print(args.print_help())
    warnings.warn('Input files must be supplied (-n,-t). ')


dinodir=os.path.dirname(os.path.abspath(__file__))

def main():
    print('Start')
    tmp_bam = pd.read_csv(f'./{args.tsv}', sep='\t', quoting=csv.QUOTE_NONE, header=0)
    print('Done loading')
    tmp_bam = tmp_bam.rename(columns={tmp_bam.columns[1]: 'strand'})
    tmp_bam['REF'] = tmp_bam['REF'].str.upper()

    tmp_bam['REF_POS'] = pd.to_numeric(tmp_bam['REF_POS'])
    tmp_bam['READ_POS'] = pd.to_numeric(tmp_bam['READ_POS'])
    tmp_bam['QUAL'] = tmp_bam['QUAL'].astype('category')

    tmp_bam['qual'] = 0
    tmp_bam['qual'] = myascii(tmp_bam['QUAL'])
    # Add mis/match + deletion columns
    tmp_bam['mat'] = 0
    tmp_bam['del'] = 0
    tmp_bam['mis'] = 0
    tmp_bam['mat'],tmp_bam['mis'], tmp_bam['del']=mymatmis(tmp_bam['OP'], tmp_bam['REF'], tmp_bam['BASE'] )

    tmp_bam['ins'] = 0
    tmp_bam_negative = tmp_bam[tmp_bam['strand'] == "n"].reset_index(drop=True)
    tmp_bam_positive = tmp_bam[tmp_bam['strand'] == "p"].reset_index(drop=True)
    del tmp_bam
    tmp_bam_positive['ins'] = insertfunP(tmp_bam_positive['OP'], tmp_bam_positive['ins'])
    tmp_bam_negative['ins']=insertfunN(tmp_bam_negative['OP'],tmp_bam_negative['ins'])
    print("Done features")

    # Filter file
    tmp_bam2_positive = tmp_bam_positive[(tmp_bam_positive['OP'] != 'I') & (tmp_bam_positive['REF_POS'] > 0)]
    readname_positive = tmp_bam2_positive['#READ_NAME'].unique()
    del tmp_bam_positive
    merge_bam_positive = tmp_bam2_positive.iloc[:, [0, 1, 2, 4, 6, 7] + list(range(9, 14))]
    del tmp_bam2_positive

    tmp_bam2_negative = tmp_bam_negative[(tmp_bam_negative['OP'] != 'I') & (tmp_bam_negative['REF_POS'] > 0)]
    del tmp_bam_negative
    readname_negative = tmp_bam2_negative['#READ_NAME'].unique()
    merge_bam_negative = tmp_bam2_negative.iloc[:, [0, 1, 2, 4, 6, 7, 9, 10, 11, 12, 13]]
    del tmp_bam2_negative

    merge_bam_negative['BASE'] = merge_bam_negative['BASE'].astype(str)
    merge_bam_negative['BASE1'] = ""
    merge_bam_negative['BASE1'] = complement_char(merge_bam_negative['BASE'])
    merge_bam_negative = merge_bam_negative.drop(columns=['BASE'])
    merge_bam_negative = merge_bam_negative.rename(columns={'BASE1': 'BASE'})
    print("Done filter bam file")

    # S2:Nanopolish file
    data_nnpl = pd.read_csv(f"./{args.nnpl}", sep="\t",
                            dtype={'col1': 'category', 'col2': 'category', 'col3': 'float64', 'col4': 'float64',
                                   'col5': 'float64', 'col6': 'float64', 'col7': 'category'})
    merge_nnpl_positive = data_nnpl[data_nnpl['read_name'].isin(readname_positive)].copy()
    merge_nnpl_positive['position'] += 2  # Set coordinate to second position in kmer
    merge_nnpl_negative = data_nnpl[data_nnpl['read_name'].isin(readname_negative)].copy()
    del data_nnpl
    print("Done positive strand for nanopolish")
    merge_nnpl_negative['position'] += 4
    merge_nnpl_negative['reference_kmer'] = merge_nnpl_negative['reference_kmer'].astype(str)
    merge_nnpl_negative['reference_kmer1'] = ""
    merge_nnpl_negative['reference_kmer1'] = rev_comp(merge_nnpl_negative['reference_kmer'])
    merge_nnpl_negative = merge_nnpl_negative.drop(columns=['reference_kmer'])
    merge_nnpl_negative.rename(columns={'reference_kmer1': 'reference_kmer'}, inplace=True)
    print("Done negative strand for nanopolish")

    # S3:merge file
    combine_positive=pd.merge(merge_nnpl_positive, merge_bam_positive,
             left_on=["read_name", "position", "contig"],
             right_on=["#READ_NAME", "REF_POS", "CHROM"], how='right')
    combine_positive['read_name']=combine_positive['#READ_NAME']
    combine_positive['position'] = combine_positive['REF_POS']
    combine_positive['contig'] = combine_positive['CHROM']
    combine_positive = combine_positive.drop(columns=['#READ_NAME'])
    combine_positive = combine_positive.drop(columns=['REF_POS'])
    combine_positive = combine_positive.drop(columns=['CHROM'])
    del merge_nnpl_positive, merge_bam_positive
    combine_negative = pd.merge(merge_nnpl_negative, merge_bam_negative,
                                left_on=["read_name", "position", "contig"],
                                right_on=["#READ_NAME", "REF_POS", "CHROM"], how='right')
    combine_negative['read_name'] = combine_negative['#READ_NAME']
    combine_negative['position'] = combine_negative['REF_POS']
    combine_negative['contig'] = combine_negative['CHROM']
    combine_negative = combine_negative.drop(columns=['#READ_NAME'])
    combine_negative = combine_negative.drop(columns=['REF_POS'])
    combine_negative = combine_negative.drop(columns=['CHROM'])
    del merge_nnpl_negative, merge_bam_negative
    print("Done merging files")

    # S4:Add neighboring
    data_sorted_positive = combine_positive.sort_values(by=['read_name', 'position']).reset_index(drop=True)
    del combine_positive
    data_sorted_negative = combine_negative.sort_values(by=['read_name', 'position']).reset_index(drop=True)
    del combine_negative
    print("Done adding neighboring positions")
    gc.collect()

    split_size = len(data_sorted_positive['read_name'].unique()) // (22 + 1)
    split_indices = np.arange(0, len(data_sorted_positive['read_name'].unique()) - 1, split_size + 1)
    # Split the list of names
    name_list = np.split(data_sorted_positive['read_name'].unique(), split_indices[1:])
    dat_com_chunks = Parallel(n_jobs=-1, backend='loky')(
        delayed(process_data)(names, data_sorted_positive) for names in name_list)
    dat_com_positive = pd.concat(dat_com_chunks, ignore_index=True)

    del data_sorted_positive


    dat_com_positive['posi_id'] = dat_com_positive['contig'] + ':' + dat_com_positive['position'].astype(str) + ':' + \
                                  dat_com_positive['strand']
    dat_com_positive.to_csv(args.out + '.positive', sep='\t', index=False, quoting=False)
    col = dat_com_positive.columns.tolist()
    del dat_com_positive
    print("Done positive strand")

    #### compile features table for negative-stranded sequences ####
    split_size = len(data_sorted_negative['read_name'].unique()) // (22 + 1)
    split_indices = np.arange(0, len(data_sorted_negative['read_name'].unique()) - 1, split_size + 1)
    # Split the list of names
    name_list = np.split(data_sorted_negative['read_name'].unique(), split_indices[1:])
    dat_com_chunks = Parallel(n_jobs=-1, backend='loky')(
        delayed(process_data)(names, data_sorted_negative) for names in name_list)
    dat_com_negative = pd.concat(dat_com_chunks, ignore_index=True)
    del data_sorted_negative
    print("Done negative strand")

    dat_com_negative['posi_id'] = dat_com_negative['contig'] + ':' + dat_com_negative['position'].astype(str) + ':' + \
                                  dat_com_negative['strand']
    dat_com_negative = dat_com_negative[col]
    dat_com_negative.to_csv(args.out + '.negative', sep='\t', index=False, header=False, mode='a', quoting=False)

    print("Done!")
if __name__ == '__main__':
    main()