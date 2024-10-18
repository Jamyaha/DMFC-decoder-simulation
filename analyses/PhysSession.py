import sys
import os
import argparse
import numpy as np
import pickle as pk
import pandas as pd
from phys import daq_utils, data_utils

from sklearn.model_selection import train_test_split
# from behavior import MentalPongBehavior as MPB
import phys.phys_utils as phys_utils
from phys.PhysBehavior import PhysBehavior
from glob import glob
from copy import deepcopy
from scipy.stats import pearsonr

PATH_ = '/om/user/rishir/lib/analysis-tools/Python3/'
sys.path.insert(0, PATH_)

parser = argparse.ArgumentParser()
parser.add_argument('--data_directory')
parser.add_argument('--dataset_idx_oi', default=0, type=int)
parser.add_argument('--experiment_idx_oi', default=1, type=int)


class PhysSessionUnpacker(object):
    def __init__(self, **kwargs):
        self.data_directory = kwargs.get('data_directory')
        self.daq, self.probe_type, self.nchannels_total = data_utils.get_experiment_specs(self.data_directory)

        self.dataset_idx_oi = kwargs.get('dataset_idx_oi', 0)  # pick out 0: pong_basic, 1: telepong
        self.experiment_idx_oi = kwargs.get('experiment_idx_oi', 1)  # pick out 1: hand, 0: eye

        self.kilosort_version = kwargs.get('kilosort_version', 3)
        self.kilosort_version_suffix = kwargs.get('kilosort_version_suffix', '_1')
        self.kilosort_subdirectory = 'ks%d_output%s' % (self.kilosort_version, self.kilosort_version_suffix)

        self.cluster_merge_suffix = kwargs.get('cluster_merge_suffix', 'v3')
        self.merge_search_suffix = 'ks%d%s_merge_rr%s' % (self.kilosort_version,
                                                          self.kilosort_version_suffix,
                                                          self.cluster_merge_suffix)

        self.photodiode_ad_fn = '100_ADC1.continuous'

        self.unique_conds = phys_utils.get_trial_meta_indices()
        self.timepoints_per_cond = phys_utils.get_trial_timepoints()

        self.max_duration = 5000  # read 5 seconds from align_event (e.g. sync), at millisecond precision
        self.baseline_duration = 500
        self.align_to_event = 't_ball_on_offset_300'

        self.minfr_per_trial_thres = 5
        self.trial_window_thres = 3
        self.min_trials = 10

        self.trial_mean_groupbyvar = 'py_meta_index'
        self.trial_mean_groups = [
            'occ_alpha == 1 & success == 1',
            'occ_alpha != 1 & success == 1',
            'occ_alpha == 1 & failure == 1',
            'occ_alpha != 1 & failure == 1',
            'occ_alpha == 1 & ignore == 0',
            'occ_alpha != 1 & ignore == 0',
            'ignore == 0',
            'ignore == 1 | failure == 1'  # trials where the paddle doesn't hit the ball, to get "true" ball trajectory

        ]

        self.neural_data_raw = {}
        self.neural_responses = {}
        self.neural_responses_splits = []
        self.neural_responses_bc = {}
        self.neural_responses_bc_splits = []

        self.behavioral_data_raw = None
        self.behavioral_responses = {}

        self.neural_data = None
        self.neural_meta = None
        self.trial_meta = None
        self.meta = None
        self.all_clusters = None

        self.alignment_conds = {}
        self.masks = {}
        self.alignment_axes = {}

        file_suffix = '%s_exp%d_dset%d' % (self.merge_search_suffix, self.experiment_idx_oi, self.dataset_idx_oi)
        self.save_fn = '%s/phys_session_%s.pkl' % (self.data_directory, file_suffix)

        return

    @staticmethod
    def subsample_data(data_in, t_subsample):
        if isinstance(data_in, pd.DataFrame):
            return data_in.iloc[t_subsample].reset_index(drop=True)
        elif isinstance(data_in, np.ndarray):
            return data_in[t_subsample, :]

    def get_trial_mean(self, x_in, y):
        # y is a pandas dataframe
        g = y.groupby(self.trial_mean_groupbyvar)
        x_in_mu, x_in_sh1, x_in_sh2 = [], [], []
        for ci in self.unique_conds:
            try:
                t = np.array(g.get_group(ci).index)
                t1, t2 = train_test_split(t, test_size=0.5, random_state=0)
                x_ = np.nanmean(x_in[t, :], axis=0)
                x_sh1 = np.nanmean(x_in[t1, :], axis=0)
                x_sh2 = np.nanmean(x_in[t2, :], axis=0)
            except Exception:
                x_ = np.ones((x_in.shape[1],)) * np.nan
                x_sh1 = np.ones((x_in.shape[1],)) * np.nan
                x_sh2 = np.ones((x_in.shape[1],)) * np.nan
            x_in_mu.append(x_)
            x_in_sh1.append(x_sh1)
            x_in_sh2.append(x_sh2)
        return {'X_mu': np.array(x_in_mu), 'X_sh1': np.array(x_in_sh1), 'X_sh2': np.array(x_in_sh2)}

    def get_mworks_raw_data(self):
        mworks_fn = glob('%s/jazlab*.mat' % self.data_directory)[0]
        max_trial_time = self.max_duration * 1000
        data = PhysBehavior(if_joystick=True, if_eye=True, max_trial_time=max_trial_time)
        data.update_with_data([mworks_fn])
        self.behavioral_data_raw = data
        self.trial_meta = self.behavioral_data_raw.datasets[0]['scalar'].copy()
        ttl = self.behavioral_data_raw.datasets[0]['ttl']['sync']
        if np.nanmean(np.log10(np.diff(ttl))) > 4:  # convert to ms from us
            ttl = ttl / 1000.0
        self.trial_meta['t_sync_on_mw'] = ttl
        return

    def get_phys_raw_data_events(self):
        if self.daq == 'open-ephys':
            t_sync_on_oe = daq_utils.get_oe_sync_on(self.data_directory)
        elif self.daq == 'spikeglx':
            t_sync_on_oe = daq_utils.get_spikeglx_sync_on(self.data_directory)
        else:
            t_sync_on_oe = daq_utils.get_oe_sync_on(self.data_directory)

        t_sync_on_mw = self.trial_meta['t_sync_on_mw']

        res_sync_align = daq_utils.match_sync_signals(t_sync_on_mw, t_sync_on_oe)
        self.trial_meta['t_sync_on_mw'] = res_sync_align['sb']
        self.trial_meta['t_sync_on_oe'] = res_sync_align['sp']
        # add some stats on how good the match is
        self.trial_meta['sync_match_r'] = res_sync_align['r']
        self.trial_meta['sync_match_delta_err'] = res_sync_align['delta_err']
        self.trial_meta['sync_match_time_err'] = res_sync_align['time_err']
        return

    def get_phys_raw_data_spikes(self):
        if self.daq == 'open-ephys':
            neural_data_raw = daq_utils.get_oe_neural_data_raw(self.data_directory, self.kilosort_subdirectory)
        elif self.daq == 'spikeglx':
            neural_data_raw = daq_utils.get_spikeglx_neural_data_raw(self.data_directory,
                                                                     self.kilosort_subdirectory)
        else:
            neural_data_raw = daq_utils.get_spikeglx_neural_data_raw(self.data_directory,
                                                                     self.kilosort_subdirectory)

        cluster_fn = '%s/%s/spike_cluster_merges.pkl' % (self.data_directory, self.merge_search_suffix)
        tmp = pk.load(open(cluster_fn, 'rb'))
        neural_data_raw['spike_clusters'] = tmp['res_clustering']
        self.neural_data_raw = neural_data_raw
        self.all_clusters = np.unique(self.neural_data_raw['spike_clusters'])
        return

    def get_phys_raw_data_photodiode(self):
        if self.daq == 'open-ephys':
            self.neural_data_raw['photodiode_oe'] = daq_utils.get_oe_photodiode(self.data_directory,
                                                                                self.photodiode_ad_fn)
        elif self.daq == 'spikeglx':
            self.neural_data_raw['photodiode_oe'] = daq_utils.get_spikeglx_photodiode(self.data_directory,
                                                                                      self.photodiode_ad_fn)
        return

    def get_phys_raw_data(self):
        self.get_phys_raw_data_events()
        self.get_phys_raw_data_spikes()
        self.get_phys_raw_data_photodiode()
        return

    def get_other_timing_variables(self):
        # by default, everything is aligned to sync. here, estimate the timing of other events (relative to sync).
        # best to align to some fixed time before ball onset, to avoid variability across trials in trial extraction

        def _get_ball_display_onset(x_):
            t_ = []
            for x_t in x_:
                tmp = np.nonzero(np.isfinite(x_t))[0]
                if len(tmp) > 0:
                    t_.append(tmp[0])
                else:  # this should be relevant on trials that will be pruned out, because the ball wasn't displayed?
                    t_.append(0)
            return np.array(t_)
            # return np.array([np.nonzero(np.isfinite(x_t))[0][0] for x_t in x_])

        def _get_photodiode_onset(x_):
            photodiode_thres = 4
            t_ = []
            for x_t in x_:
                tmp = np.nonzero(x_t > photodiode_thres)[0]
                if len(tmp) > 0:
                    t_.append(tmp[0])
                else:
                    t_.append(0)
            return np.array(t_)

        data = self.behavioral_data_raw

        ball_pos_x = np.array(data.datasets[0]['analog_sample']['ball_pos_x'])
        t_ball = _get_ball_display_onset(ball_pos_x)
        self.trial_meta['t_ball_on_relative'] = t_ball
        ball_pre_offset = 300
        self.trial_meta['t_ball_on_offset_%d_relative' % ball_pre_offset] = t_ball - ball_pre_offset

        photodiode = np.array(data.datasets[0]['analog_sample']['photodiode'])
        self.trial_meta['t_photodiode_on_relative'] = _get_photodiode_onset(photodiode)
        # don't use photodiode to align, since the photodiode onset time during trial changed between
        #  early Perle sessions (2020/06) and after.

        for fk in ['t_photodiode_on', 't_ball_on', 't_ball_on_offset_%d' % ball_pre_offset]:
            self.trial_meta['%s_mw' % fk] = self.trial_meta['t_sync_on_mw'] - self.trial_meta['%s_relative' % fk]
            self.trial_meta['%s_oe' % fk] = self.trial_meta['t_sync_on_oe'] - self.trial_meta['%s_relative' % fk]

        return

    def realign_mworks_analog_data(self):
        def shift_mworks_analog_data(analog_data, t_shift):
            analog_data_shifted = {}
            for fk_ in analog_data.keys():
                x_1 = deepcopy(analog_data[fk_])
                x_2 = np.ones(x_1.shape) * np.nan
                ntrials = x_1.shape[0]
                for i in range(ntrials):
                    idx_shift = int(t_shift[i])
                    tmp = x_1[i, idx_shift:]
                    x_2[i, :tmp.shape[0]] = tmp
                analog_data_shifted[fk_] = x_2
            return analog_data_shifted

        fk = '%s_relative' % self.align_to_event
        t_relative_to_sync = self.trial_meta[fk]
        a_data = self.behavioral_data_raw.datasets[0]['analog_sample']
        a_data_shifted = shift_mworks_analog_data(a_data, t_relative_to_sync)
        self.behavioral_data_raw.datasets[0]['analog_sample_shifted'] = a_data_shifted

        return

    def prune_trials(self):
        # find good trials (AND of conditions for good trials).
        df = self.trial_meta

        t_exp = df['experiment_idx'] == self.experiment_idx_oi
        t_dataset = df['dataset_idx'] == self.dataset_idx_oi

        t1 = np.isfinite(df['py_meta_index'])
        t2 = (df['success'] + df['failure'] + df['ignore']) > 0
        t3 = df['occ_alpha'] > 0  # weird mworks glitch in human data

        # lapse (paddle outside frame)
        max_frame_w_paddle = 10
        t4_hi = (df['joystick_output'] <= max_frame_w_paddle)
        t4_lo = (df['joystick_output'] >= -max_frame_w_paddle)
        t4 = t4_lo & t4_hi

        del_t = np.abs(df['t_sync_on_mw'] - df['t_photodiode_on_mw'])
        mu, sig = np.nanmean(del_t), np.nanstd(del_t)
        t5 = ((del_t - mu) / sig) < 3  # remove outlier display delays

        t = t1 & t2 & t3 & t4 & t5 & t_exp & t_dataset

        # remove from trial meta and on all trial data.
        # this needs to be done before computing trials of neural data
        # and before any trial-averaging.
        # also, make sure to re-index here.
        self.trial_meta = self.trial_meta[t].reset_index(drop=True)
        for fk in self.behavioral_data_raw.datasets[0]['analog_sample'].keys():
            for beh_fk in ['analog_sample', 'analog_sample_shifted']:
                subsampled_analog = self.behavioral_data_raw.datasets[0][beh_fk][fk][t, :]
                self.behavioral_data_raw.datasets[0][beh_fk][fk] = subsampled_analog
        return

    def get_behavioral_responses(self):
        def get_mworks_scalar():
            trial_meta = self.trial_meta
            trial_mean_groupbyvar = self.trial_mean_groupbyvar
            self.meta = phys_utils.consolidate_trial_meta(trial_meta,
                                                          trial_mean_groupbyvar=trial_mean_groupbyvar)
            return

        def get_mworks_analog(augment_with_eye_velocity=True):
            data_analog = self.behavioral_data_raw.datasets[0]['analog_sample_shifted']
            behavioral_responses = {}
            trial_meta = self.trial_meta
            for g_query in self.trial_mean_groups:
                y_curr = trial_meta.query(g_query)  # make sure not to re-index here
                behavioral_responses[g_query] = {}
                for fk in data_analog.keys():
                    tmp_res = self.get_trial_mean(data_analog[fk], y_curr)
                    behavioral_responses[g_query][fk] = tmp_res['X_mu']
                if augment_with_eye_velocity:
                    # estimate eye velocity on single trial data, and then average.
                    eye_v, eye_h = data_analog['eye_v'], data_analog['eye_h']
                    eye_dv = np.diff(eye_v, axis=1, append=np.nan)
                    eye_dh = np.diff(eye_h, axis=1, append=np.nan)
                    eye_dtheta = np.arctan2(eye_dv, eye_dh)
                    eye_dspeed = (eye_dv ** 2 + eye_dh ** 2) ** 0.5
                    behavioral_responses[g_query]['eye_dv'] = self.get_trial_mean(eye_dv, y_curr)['X_mu']
                    behavioral_responses[g_query]['eye_dh'] = self.get_trial_mean(eye_dh, y_curr)['X_mu']
                    behavioral_responses[g_query]['eye_dtheta'] = self.get_trial_mean(eye_dtheta, y_curr)['X_mu']
                    behavioral_responses[g_query]['eye_dspeed'] = self.get_trial_mean(eye_dspeed, y_curr)['X_mu']

            self.behavioral_responses = behavioral_responses
            return

        get_mworks_scalar()
        get_mworks_analog()
        return

    def get_neural_responses(self):
        def get_cluster_electrode_location(cidx):
            clusters = self.neural_data_raw['spike_clusters']['c_idx_temp']
            peak_time_index = 40
            templates_per_cluster = clusters[cidx]
            template_max = []
            for t_idx in templates_per_cluster:
                tmp = np.argmax(np.abs(self.neural_data_raw['templates'][t_idx][peak_time_index, :]))
                template_max.append(tmp)
            return np.nanmean(template_max)

        def get_trials_from_spiketimes_per_unit(template_idx_of_cluster, sync_var):
            spike_times = self.neural_data_raw['spike_times']
            spike_templates = self.neural_data_raw['spike_templates']
            spike_times_curr = spike_times[np.isin(spike_templates, template_idx_of_cluster)]

            sync_t = self.trial_meta[sync_var]
            ntrials = sync_t.shape[0]

            n_bins_fr = int(self.max_duration)
            fr_curr = np.ones((ntrials, n_bins_fr)) * np.nan
            fr_curr_baseline_corrected = np.ones((ntrials, n_bins_fr)) * np.nan

            for trial_idx in range(ntrials):
                ts_start = sync_t[trial_idx]
                ts_end = sync_t[trial_idx] + self.max_duration
                bs_start = sync_t[trial_idx] - 10 - self.baseline_duration
                bs_end = sync_t[trial_idx] - 10

                bins = np.linspace(ts_start, ts_end, n_bins_fr + 1)
                tmp = spike_times_curr[(spike_times_curr >= ts_start) & (spike_times_curr <= ts_end)]
                raster = np.histogram(tmp, bins)[0]
                fr_curr[trial_idx, :] = raster
                baseline_nspikes = np.nansum((spike_times_curr >= bs_start) & (spike_times_curr <= bs_end))
                tmp_baseline = baseline_nspikes / self.baseline_duration
                fr_curr_baseline_corrected[trial_idx, :] = raster - tmp_baseline
            return fr_curr, fr_curr_baseline_corrected

        def get_stable_trials_per_unit(fr_):
            boxcar_n = self.trial_window_thres
            boxcar = np.ones((boxcar_n,)) / boxcar_n
            x = np.squeeze(fr_)
            x_sum = np.nansum(x, axis=1)  # number of spikes per trial
            x_sum_conv = np.convolve(x_sum, boxcar, mode='valid')
            min_nspikes_thres = self.minfr_per_trial_thres * self.max_duration / 1000.0
            return np.nonzero(x_sum_conv > min_nspikes_thres)[0]

        def get_trial_mean_per_unit(fr_, stable_trials):
            x_ = self.subsample_data(fr_, stable_trials)
            y_ = self.subsample_data(self.trial_meta, stable_trials)

            fr_all = {'X_mu': {}, 'X_sh1': {}, 'X_sh2': {}}
            for g_query in self.trial_mean_groups:
                y_curr = y_.query(g_query)  # make sure not to re-index here
                tmp_res = self.get_trial_mean(x_, y_curr)
                for fk_ in fr_all.keys():
                    fr_all[fk_][g_query] = tmp_res[fk_]
            return fr_all

        def init_data_structure():
            fr_all = {
                'X_mu': {}, 'X_sh1': {}, 'X_sh2': {}
            }
            for shk_ in fr_all.keys():
                for fk_ in self.trial_mean_groups:
                    fr_all[shk_][fk_] = []
            return fr_all

        def insert_trial_mean_in_dictionary(fr_, trials_, res_dict):
            res_curr = get_trial_mean_per_unit(fr_, trials_)
            for shk in res_dict.keys():
                for fk in self.trial_mean_groups:
                    res_dict[shk][fk].append(res_curr[shk][fk])
            return res_dict

        def numpy_dictionary(res_dict):
            for shk in res_dict.keys():
                for fk in self.trial_mean_groups:
                    res_dict[shk][fk] = np.array(res_dict[shk][fk])
            res_dict_mu = res_dict['X_mu']
            res_dict_splits = [res_dict['X_sh1'], res_dict['X_sh2']]
            return res_dict_mu, res_dict_splits

        cluster_assignment = self.neural_data_raw['spike_clusters']['c_idx_orig']
        neur_resp = init_data_structure()
        neur_resp_bc = init_data_structure()
        use_sync_var = '%s_oe' % self.align_to_event

        neural_meta_df = []
        for c_idx, cluster_curr in enumerate(cluster_assignment):
            neural_meta_curr = {}
            fr, fr_bc = get_trials_from_spiketimes_per_unit(cluster_curr, use_sync_var)
            t_ = get_stable_trials_per_unit(fr)

            if np.nansum(t_) < self.min_trials:
                continue
            neur_resp = insert_trial_mean_in_dictionary(fr, t_, neur_resp)
            neur_resp_bc = insert_trial_mean_in_dictionary(fr_bc, t_, neur_resp_bc)

            elec_loc = get_cluster_electrode_location(c_idx)

            neural_meta_curr['cluster_idx'] = c_idx
            neural_meta_curr['templates'] = cluster_curr
            neural_meta_curr['num_stable_trials'] = t_.shape[0]
            neural_meta_curr['cluster_location'] = elec_loc
            neural_meta_df.append(neural_meta_curr)

        self.neural_responses, self.neural_responses_splits = numpy_dictionary(neur_resp)
        self.neural_responses_bc, self.neural_responses_bc_splits = numpy_dictionary(neur_resp_bc)

        self.neural_meta = pd.DataFrame(neural_meta_df)
        # self.neural_meta['cluster_location'] = cluster_electrode_location()
        return

    def get_alignment_indices(self):
        """
        by default, everything is aligned to sync, on a trial-by-trial basis.
        MentalPongBehavior automatically aligns to t_sync_on_mw,
        Open-ephys data is aligned to t_sync_on_oe, a duplicate of this.

        Above, we re-align to photodiode onset, for both.

        Here, we get indices for re-aligning trial-means.

        Since there appears to be some jitter in ball_position to sync across sessions (average 2ms),
        we use a single global reference frame (from rnn dataset, timepoints_per_cond).

        Instead of aligning data and re-saving it multiple times, just save a bunch of masks
        that can be applied on a single data matrix.
        """

        def add_pad(ts_, os_, te_, pad_length=250):
            # 250ms for post-hoc smoothing for PSTH
            padded_time = [ts_ - pad_length,
                           ts_ + pad_length,
                           os_ - pad_length,
                           os_ + pad_length,
                           te_ + pad_length]
            return padded_time

        def get_timepoints(pad_length=250):
            cond_to_use = 'ignore == 0'  # any condition should do.
            ball_x = self.behavioral_responses[cond_to_use]['ball_pos_x']
            align_timepoints = []
            trial_i = range(ball_x.shape[0])
            for i in trial_i:
                if np.nansum(np.isfinite(ball_x[i])) == 0:
                    # ensure that there were trials of this condition
                    ts, os, te = -10000, -10000, -10000
                else:
                    ts = np.nonzero(np.isfinite(ball_x[i]))[0][0]  # first index of ball_x
                    os = self.timepoints_per_cond['t_occ'][i] + ts
                    te = self.timepoints_per_cond['t_f'][i] + ts
                align_timepoints.append(add_pad(ts, os, te, pad_length=pad_length))
            return np.array(align_timepoints), trial_i

        def get_start_end_timepoints(align_timepoints):
            t_zero = [0] * align_timepoints.shape[0]
            ts_pre = list(align_timepoints[:, 0])
            ts_post = list(align_timepoints[:, 1])
            oc_pre = list(align_timepoints[:, 2])
            oc_post = list(align_timepoints[:, 3])
            te_post = list(align_timepoints[:, 4])

            te_1_step_pre = [t - 1 for t in te_post]
            oc_1_step_pre = [t - 1 for t in oc_post]

            epochs = {
                'pretrial': [t_zero, ts_post],
                'start_end': [ts_pre, te_post],
                'start_occ': [ts_pre, oc_post],
                'occ_end': [oc_pre, te_post],
                'f': [te_1_step_pre, te_post],
                'occ': [oc_1_step_pre, oc_post],
            }
            return epochs

        def get_mask_from_start_end_timepoints(t_start_, t_end_):
            cond_to_use = 'occ_alpha == 1 & success == 1'  # any condition should do.
            ball_x = self.behavioral_responses[cond_to_use]['ball_pos_x']
            blank_mask = np.ones(ball_x.shape) * np.nan
            trial_i = range(ball_x.shape[0])
            for i in trial_i:
                ts, te = int(t_start_[i]), int(t_end_[i])
                if (ts > 0) & (te > 0):  # ensure that there were trials of this condition
                    blank_mask[i, ts:te] = 1
            return blank_mask

        for pad_length_ in [0, 250]:
            align_timepoints_, trial_i_ = get_timepoints(pad_length=pad_length_)
            epochs_ = get_start_end_timepoints(align_timepoints_)
            left_axis = np.arange(-pad_length_, self.max_duration + pad_length_, 1)
            right_axis = np.arange(-pad_length_ - self.max_duration, pad_length_, 1)

            for ep in epochs_.keys():
                t_start, t_end = epochs_[ep][0], epochs_[ep][1]
                self.masks['%s_pad%d' % (ep, pad_length_)] = get_mask_from_start_end_timepoints(t_start, t_end)
                self.alignment_conds['%s_pad%d_left' % (ep, pad_length_)] = [trial_i_, t_start, t_end, True]
                self.alignment_axes['%s_pad%d_left' % (ep, pad_length_)] = left_axis
                self.alignment_conds['%s_pad%d_right' % (ep, pad_length_)] = [trial_i_, t_start, t_end, False]
                self.alignment_axes['%s_pad%d_right' % (ep, pad_length_)] = right_axis

        return

    def get_neural_response_reliability(self):
        for cond_fn in self.trial_mean_groups:
            tmp = np.isfinite(np.nanmean(self.neural_responses[cond_fn], axis=2))
            self.neural_meta['ncond_%s' % cond_fn] = np.nansum(tmp, axis=1)
            tmp = np.isfinite(np.nanmean(self.neural_responses_splits[0][cond_fn], axis=2))
            self.neural_meta['ncond_sh1_%s' % cond_fn] = np.nansum(tmp, axis=1)
            tmp = np.isfinite(np.nanmean(self.neural_responses_splits[1][cond_fn], axis=2))
            self.neural_meta['ncond_sh2_%s' % cond_fn] = np.nansum(tmp, axis=1)

        return

    def save_data(self):
        results_dict = {
            'meta': self.meta,
            'trial_meta': self.trial_meta,
            'neural_meta': self.neural_meta,
            'all_clusters': self.all_clusters,
            'trial_mean_groupbyvar': self.trial_mean_groupbyvar,
            'trial_mean_groups': self.trial_mean_groups,

            'masks': self.masks,

            'neural_responses': self.neural_responses,
            'neural_responses_splits': self.neural_responses_splits,

            'neural_responses_bc': self.neural_responses_bc,
            'neural_responses_bc_splits': self.neural_responses_bc_splits,

            'behavioral_responses': self.behavioral_responses,

            'alignment_axes': self.alignment_axes,

        }
        with open(self.save_fn, 'wb') as f:
            f.write(pk.dumps(results_dict))
        print('Saved to %s \n' % self.save_fn)
        return

    def run_all(self):
        # parse raw data from mworks and open-ephys.
        # prune trials (for bugs, lapses) simultaneously on both.
        self.get_mworks_raw_data()
        self.get_phys_raw_data()
        self.get_other_timing_variables()
        self.realign_mworks_analog_data()
        self.prune_trials()

        if self.trial_meta.shape[0] < self.min_trials:
            return

        # get trial-averaged responses of mworks analog and neural data
        self.get_behavioral_responses()
        self.get_neural_responses()

        # align trial averaged responses to trial events
        self.get_alignment_indices()

        # measure split-half reliability of neural data
        self.get_neural_response_reliability()

        # save output
        self.save_data()
        return


class ParameterSweep(object):
    def __init__(self, **kwargs):
        self.data_directory = kwargs.get('data_directory')
        self.dataset_idx_oi = kwargs.get('dataset_idx_oi', 0)
        self.experiment_idx_oi = kwargs.get('experiment_idx_oi', 1)
        self.kilosort_version = kwargs.get('kilosort_version', 3)
        # self.kilosort_version_suffix_list = kwargs.get('kilosort_version_suffix_list', [1, 2, 3, 4])
        self.kilosort_version_suffix_list = kwargs.get('kilosort_version_suffix_list', [ '_2', '_3', '_4'])
        return

    def run_merges(self):
        flags = {
            'data_directory': self.data_directory,
            'dataset_idx_oi': self.dataset_idx_oi,
            'experiment_idx_oi': self.experiment_idx_oi,
            'kilosort_version': self.kilosort_version,
        }
        #

        for v in self.kilosort_version_suffix_list:
            tmp = '%s/ks%d_output%s' % (self.data_directory, self.kilosort_version, v)
            print(tmp)
            if os.path.isdir(tmp):
                flags['kilosort_version_suffix'] = v
                psu = PhysSessionUnpacker(**flags)
                psu.run_all()
        return


def main(argv):
    print('Physiology session unpacking')
    flags_, _ = parser.parse_known_args(argv)
    flags = vars(flags_)
    ps = ParameterSweep(**flags)
    ps.run_merges()

    return


if __name__ == "__main__":
    main(sys.argv[1:])
