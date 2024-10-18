import argparse
import sys
import numpy as np
import pickle as pk
import pandas as pd
from phys import phys_utils, data_utils


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


default_save_path = '/om/user/rishir/data/phys/'

parser = argparse.ArgumentParser()
parser.add_argument('--data_in')
parser.add_argument('--downsample_rate', default=10, type=int)


class PhysDataset(object):
    def __init__(self, **kwargs):
        self.data_in = kwargs.get('data_in')
        self.downsample_rate = kwargs.get('downsample_rate', 50)
        self.save_fn = '%s/%s_dataset_%dms.pkl' % (default_save_path,
                                                   self.data_in,
                                                   self.downsample_rate)

        self.trial_mean_groupbyvar = 'py_meta_index'
        self.total_ncond = 79
        self.cutoff_ncond = 5

        self.conds_ch_dict = {
            'occ': 'occ_alpha == 1 & ignore == 0',
            'vis': 'occ_alpha != 1 & ignore == 0'}

        self.analog_ch = [
            'ball_pos_x', 'ball_pos_y',
            'paddle_pos_y', 'joy',
            'eye_v', 'eye_h',
            'eye_dv', 'eye_dh',
            'eye_dtheta', 'eye_dspeed'
        ]
        # to get "true" ball trajectory, use data from trials where the paddle doesn't hit the ball
        self.other_analog_ch = ['ball_pos_x', 'ball_pos_y']
        self.other_analog_ch_suffix = 'TRUE'
        self.cond_for_other_analog_ch = 'ignore == 1 | failure == 1'

        self.masks_ch = ['pretrial_pad0', 'start_end_pad0', 'start_occ_pad0', 'occ_end_pad0', 'f_pad0', 'occ_pad0']
        self.dat_base = {}
        return

    def get_experiment_meta(self):
        neural_meta_df = None
        search_tag = self.data_in.lower()
        if 'perle' in search_tag:
            neural_meta_df = data_utils.load_perle_phys_meta()
        # elif 'carmen' in search_tag:
        #     neural_meta_df = data_utils.load_carmen_phys_meta()
        elif 'mahler' in search_tag:
            neural_meta_df = data_utils.load_mahler_phys_meta()
        elif 'all' in search_tag:
            neural_meta_df = data_utils.load_all_phys_meta()

        query_str = ''
        if 'hand' in search_tag:
            query_str = 'task == "pong_hand" | task == "pong_hand_vis"'
            # query_str = 'task == "pong_hand"'
        elif 'eye' in search_tag:
            query_str = 'task == "pong_eye"'

        query_str_2 = ''
        if 'dmfc' in search_tag:
            query_str_2 = 'area == "DMFC"'
        elif 'ppc' in search_tag:
            query_str_2 = 'area == "PPC"'

        return neural_meta_df.query(query_str).query(query_str_2).reset_index(drop=True)

    def parse_fn_list(self, key_to_extract='data_fn'):
        if isinstance(self.data_in, list):
            return self.data_in
        else:
            neural_meta_df = self.get_experiment_meta()
            return list(neural_meta_df[key_to_extract])

    def downmsaple_binned_mean(self, x, func=np.nanmean):
        # mean statistic may nan out any bins that have at least one nan in them.
        from scipy.stats import binned_statistic
        n = self.downsample_rate
        t0 = np.arange(0, x.shape[0])
        t1 = t0[::n]
        x2 = binned_statistic(t0, x, func, bins=t1)
        # M bin edges, M-1 bin statistics : append nan
        return np.concatenate((x2.statistic, np.ones((1,)) * np.nan))

    def get_units_with_sufficient_data(self, neural_meta):
        """ enough data (enough conditions sampled in all splits) in either vis or occ conditions."""
        n_idx = []
        ncond_thres = self.total_ncond - self.cutoff_ncond
        for fk in self.conds_ch_dict.keys():
            fk_fn = self.conds_ch_dict[fk]
            n_sh1 = neural_meta['ncond_sh1_%s' % fk_fn] >= ncond_thres
            n_sh2 = neural_meta['ncond_sh2_%s' % fk_fn] >= ncond_thres
            n_all = neural_meta['ncond_%s' % fk_fn] >= ncond_thres

            t = n_sh1 & n_sh2 & n_all
            n_idx.append(np.array(t[t].index))
        n_idx_select = np.unique(np.concatenate(n_idx))
        return n_idx_select

    def load_one_session_base(self, fn):

        def select_neurons_oi(x_full, nidx_):
            x = np.array(x_full[nidx_, :, :])
            if x.shape[0] == 0:
                x = np.ones((0, x_full.shape[1], x_full.shape[2]))
            return x

        def add_experiment_meta_to_neural_meta(neural_meta_):
            exp_meta = self.get_experiment_meta()
            transfer_keys = ['Date', 'AP', 'ML', 'task', 'data_fn']
            tmp_df = exp_meta.query('data_fn == "%s"' % fn)[transfer_keys]
            for fk in transfer_keys:
                neural_meta_[fk] = tmp_df[fk].iloc[0]
            return neural_meta_

        def add_number_of_trials_for_behavior(tmp, trial_meta):
            for sfk in self.conds_ch_dict:
                sfk2 = self.conds_ch_dict[sfk]
                tmp['ntr_%s' % sfk] = [trial_meta.query(sfk2).shape[0]]
            return tmp

        dat = pk.load(open(fn, 'rb'))
        neur_idx = self.get_units_with_sufficient_data(dat['neural_meta'])
        neural_meta_per_sess = dat['neural_meta'].iloc[neur_idx].reset_index(drop=True)
        neural_meta_per_sess = add_experiment_meta_to_neural_meta(neural_meta_per_sess)

        dat_pruned = {
            'trial_meta': dat['trial_meta'],
            'neural_meta': neural_meta_per_sess,
            'neural_responses': {},
            'neural_responses_sh1': {},
            'neural_responses_sh2': {},
            'behavioral_responses': {},
            'masks': {},
            'ntr_vis': [],
            'ntr_occ': [],
        }

        dat_pruned = add_number_of_trials_for_behavior(dat_pruned, dat['trial_meta'])

        for cond_fk in self.conds_ch_dict.keys():
            cond = self.conds_ch_dict[cond_fk]
            # select neural responses to reliable units
            for nr_fk in ['neural_responses']:
                dat_pruned[nr_fk][cond_fk] = select_neurons_oi(dat[nr_fk][cond], neur_idx)
                dat_pruned['%s_sh1' % nr_fk][cond_fk] = select_neurons_oi(dat['%s_splits' % nr_fk][0][cond], neur_idx)
                dat_pruned['%s_sh2' % nr_fk][cond_fk] = select_neurons_oi(dat['%s_splits' % nr_fk][1][cond], neur_idx)

            # copy behavioral responses to select analogs
            dat_pruned['behavioral_responses'][cond_fk] = {}
            for afk in self.analog_ch:
                dat_pruned['behavioral_responses'][cond_fk][afk] = np.array(dat['behavioral_responses'][cond][afk])
            for afk in self.other_analog_ch:
                dat_pruned['behavioral_responses'][cond_fk]['%s_%s' % (afk, self.other_analog_ch_suffix)] = \
                    np.array(dat['behavioral_responses'][self.cond_for_other_analog_ch][afk])

            # copy masks for all epochs
            dat_pruned['masks'][cond_fk] = {}
            for afk in self.masks_ch:
                dat_pruned['masks'][cond_fk][afk] = np.array(dat['masks'][afk])
        return dat_pruned

    def add_session_to_dataset_base(self, fn):
        def add_to_existing_base(x_curr, x_new):
            if x_curr.ndim == 3:
                if x_new.ndim == 3:
                    # append along neurons axis (0)
                    x_out = np.concatenate((x_curr, x_new), axis=0)
                else:
                    # append along new dimension (of sessions)
                    tmp_x_new = np.expand_dims(x_new, axis=0)
                    x_out = np.concatenate((x_curr, tmp_x_new), axis=0)
            else:
                # add third dimension (of sessions)
                x_out = np.stack((x_curr, x_new))
            return x_out

        dat_to_append = self.load_one_session_base(fn)
        tmp_df_ = pd.concat([self.dat_base['trial_meta'], dat_to_append['trial_meta']])
        self.dat_base['trial_meta'] = tmp_df_.reset_index(drop=True)
        tmp_df_ = pd.concat([self.dat_base['neural_meta'], dat_to_append['neural_meta']])
        self.dat_base['neural_meta'] = tmp_df_.reset_index(drop=True)

        self.dat_base['ntr_vis'].append(dat_to_append['ntr_vis'][0])
        self.dat_base['ntr_occ'].append(dat_to_append['ntr_occ'][0])

        for cond_fk in self.conds_ch_dict.keys():
            dict_fk = 'behavioral_responses'
            analog_ch_list = self.analog_ch + ['%s_%s' % (s, self.other_analog_ch_suffix) for s in self.other_analog_ch]
            for afk in analog_ch_list:
                x = np.array(self.dat_base[dict_fk][cond_fk][afk])
                x_n = np.array(dat_to_append[dict_fk][cond_fk][afk])
                self.dat_base[dict_fk][cond_fk][afk] = add_to_existing_base(x, x_n)

            dict_fk = 'masks'
            for afk in self.masks_ch:
                x = np.array(self.dat_base[dict_fk][cond_fk][afk])
                x_n = np.array(dat_to_append[dict_fk][cond_fk][afk])
                self.dat_base[dict_fk][cond_fk][afk] = add_to_existing_base(x, x_n)

            neur_fks = ['neural_responses', 'neural_responses_sh1', 'neural_responses_sh2']
            for dict_fk in neur_fks:
                x = np.array(self.dat_base[dict_fk][cond_fk])
                x_n = np.array(dat_to_append[dict_fk][cond_fk])
                self.dat_base[dict_fk][cond_fk] = add_to_existing_base(x, x_n)

        return

    def consolidate_sessions(self):

        self.dat_base['meta'] = phys_utils.consolidate_trial_meta(self.dat_base['trial_meta'],
                                                                  self.trial_mean_groupbyvar)
        self.dat_base['neural_idx_global'] = {}
        neur_fks = ['neural_responses', 'neural_responses_sh1', 'neural_responses_sh2']

        for cond_fk in self.conds_ch_dict.keys():
            self.dat_base['neural_idx_global'][cond_fk] = {}
            for dict_fk in neur_fks:
                x = self.dat_base[dict_fk][cond_fk]
                x_nnan, idx_nnan = phys_utils.impute_nan(x, cutoff_ncond=self.cutoff_ncond)
                x_downsampled = np.array([[self.downmsaple_binned_mean(j) for j in i] for i in x_nnan])
                self.dat_base[dict_fk][cond_fk] = x_downsampled
                self.dat_base['neural_idx_global'][cond_fk][dict_fk] = idx_nnan

            for dict_fk in ['behavioral_responses', 'masks']:
                resp = self.dat_base[dict_fk]
                for afk in resp[cond_fk].keys():
                    x = resp[cond_fk][afk]
                    if x.ndim == 3:
                        x = np.nanmean(x, axis=0)
                    x_downsampled = np.array([self.downmsaple_binned_mean(i) for i in x])
                    self.dat_base[dict_fk][cond_fk][afk] = x_downsampled

        return

    def add_sessions_to_dataset(self, fns):
        # initialize
        self.dat_base = self.load_one_session_base(fns[0])

        if len(fns) > 1:
            # aggregate
            for fn in fns[1:]:
                self.add_session_to_dataset_base(fn)
        # consolidate
        self.consolidate_sessions()
        return

    def save_dataset(self):
        dat_to_save = {}
        dat_to_save.update(self.dat_base)
        dat_to_save['conds_ch_dict'] = self.conds_ch_dict

        with open(self.save_fn, 'wb') as f:
            f.write(pk.dumps(dat_to_save))
        print('dat_to_save saved to %s' % self.save_fn)
        return

    def run_all(self):
        file_list = self.parse_fn_list()
        self.add_sessions_to_dataset(file_list)
        self.save_dataset()
        return


def main(argv):
    print('Physiology session unpacking')
    flags_, _ = parser.parse_known_args(argv)
    flags = vars(flags_)

    print(flags)
    dataset = PhysDataset(**flags)
    dataset.run_all()

    return


if __name__ == "__main__":
    main(sys.argv[1:])
