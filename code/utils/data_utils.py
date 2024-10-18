import pandas as pd
from utils import phys_utils, dataset_augment_utils
import numpy as np
import pickle as pk
from glob import glob
import os


def get_experiment_specs(data_directory):
    # daq, probe, nchannels
    dir_fn = glob('%s/*/100_CH1.continuous' % data_directory)
    if len(dir_fn) == 1:
        return 'open-ephys', 'vprobe', 64
    dir_fn = glob('%s/*.ap.bin' % data_directory)
    if len(dir_fn) == 1:
        return 'spikeglx', 'neuropxl', 384

#
# def load_neural_dataset_base(subject_id='perle_hand', timebinsize=50):
#     fn = '/om/user/rishir/data/phys/%s_dataset_%dms.pkl' % (subject_id, timebinsize)
#     dat = pk.load(open(fn, 'rb'))
#     return dat


def load_neural_dataset(subject_id='perle_hand', timebinsize=50, recompute_augment=False, compute_egocentric=False):
    fn = '../data/%s_dataset_%dms.pkl' % (subject_id, timebinsize)
    # fn = '/Users/hansem/Dropbox (MIT)/MPong/phys/data/%s_dataset_%dms.pkl' % (subject_id, timebinsize)
    # fn = '/om/user/rishir/data/phys/%s_dataset_%dms.pkl' % (subject_id, timebinsize)
    dat = pk.load(open(fn, 'rb'))
    if recompute_augment:
        dat = dataset_augment_utils.augment_data_structure(dat)
        with open(fn, 'wb') as f:
            f.write(pk.dumps(dat))
    if compute_egocentric: ###
        dat = dataset_augment_utils.convert_allocentric_to_egocentric(dat)
        with open(fn, 'wb') as f:
            f.write(pk.dumps(dat))
    return dat


def load_phys_meta(data_summary_tsv_fn):
    def get_filename_for_session(df_):
        if (df_['task'] == 'pong_hand') or (df_['task'] == 'pong_hand_vis'):
            suffix2 = 'exp1_dset0'
        elif df_['task'] == 'pong_eye':
            suffix2 = 'exp0_dset0'
        # elif df_['task'] == 'pong_hand_vis':
        #     suffix2 = 'exp2_dset0'
        for suffix1 in ['ks3_1_merge_rrv3', 'ks3_merge_rrv3']:
            fn_ = '%s/phys_session_%s_%s.pkl' % (df_['data_directory'], suffix1, suffix2)
            if os.path.isfile(fn_):
                return fn_
        return None

    monk = data_summary_tsv_fn.split('/')[-1].split('.')[-2].replace('Physiology', '')
    data_dir = '/om4/group/jazlab/rishir/data/phys/%s/' % monk

    neural_meta_df = phys_utils.read_tsv(data_summary_tsv_fn)
    neural_meta_df['monk'] = monk
    neural_meta_df['AP'] = pd.to_numeric(neural_meta_df['AP'], errors='coerce')
    neural_meta_df['ML'] = pd.to_numeric(neural_meta_df['ML'], errors='coerce')
    neural_meta_df['grid_index'] = neural_meta_df['AP'] + neural_meta_df['ML'] * 100.0
    neural_meta_df['data_directory'] = data_dir + neural_meta_df['Date']
    neural_meta_df['data_fn'] = None

    for i in range(neural_meta_df.shape[0]):
        neural_meta_df['data_fn'][i] = get_filename_for_session(neural_meta_df.iloc[i])

    return neural_meta_df


def load_perle_phys_meta():
    fn = '/om4/group/jazlab/rishir/data/phys/PerlePhysiology.tsv'
    return load_phys_meta(fn)


# def load_carmen_phys_meta():
#     fn = '/om4/group/jazlab/rishir/data/phys/Carmen/CarmenPhysiology.tsv'
#     return load_phys_meta(fn)

def load_mahler_phys_meta():
    fn = '/om4/group/jazlab/rishir/data/phys/MahlerPhysiology.tsv'
    return load_phys_meta(fn)


def load_all_phys_meta():
    df1 = load_perle_phys_meta()
    # df2 = load_carmen_phys_meta()
    df3 = load_mahler_phys_meta()
    return pd.concat((df1, df3)).reset_index(drop=True)
