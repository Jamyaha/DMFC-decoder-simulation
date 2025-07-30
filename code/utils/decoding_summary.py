from utils.phys_utils import nnan_pearsonr as nnan_pearsonr
import utils.phys_utils as phys_utils
from utils import data_utils as phys_data_utils
from utils import utils as analysis_utils
partial_corr = analysis_utils.partial_corr
import matplotlib.pyplot as plt           # TBD

from glob import glob
import sys
import os
import argparse
import numpy as np
import pickle as pk
import matplotlib.pyplot as pltplot_decode_performance_xy
import seaborn as sns
from copy import deepcopy
import pandas as pd
import pingouin as pg # TBD

from utils import plot_specs
from utils import generic_plot_utils as rr_gpu
from scipy.stats import ranksums
from scipy.stats import wilcoxon
from mlxtend.evaluate import permutation_test

sns.set_style('ticks', {"axes.linewidth": "1", 'axes.yaxis.grid': False})

parser = argparse.ArgumentParser()
parser.add_argument('--timebinsize', default=50, type=int)
parser.add_argument('--subject_id', default='all_hand_dmfc') # perle_hand_dmfc
parser.add_argument('--masks_to_test_suffix', default='start_end_pad0') # occ_end_pad0
parser.add_argument('--condition', default='occ')
parser.add_argument('--k_folds', default=5, type=int)
parser.add_argument('--neural_data_to_use', default='neural_responses_reliable_FactorAnalysis_50') # neural_responses_reliable_FactorAnalysis_50 neural_responses_reliable


class PlotHelper(object):
    def __init__(self, **kwargs):
        self.timebinsize = kwargs.get('timebinsize')
        self.train_size = kwargs.get('train_size', 0.5)
        self.subject_id = kwargs.get('subject_id', 'all_hand_dmfc') # perle_hand_dmfc
        self.masks_to_test_suffix = kwargs.get('masks_to_test_suffix', 'start_end_pad0') # occ_end_pad0
        self.condition = kwargs.get('condition')

        self.neural_data_to_use = kwargs.get('neural_data_to_use', 'neural_responses_reliable')
        self.preprocess_func = kwargs.get('preprocess_func', 'none')
        self.ncomp = kwargs.get('ncomp') # 50

        figoutpath_base = plot_specs.figoutpath_base
        self.figoutpath = '%s/%s%d/%s/decode_%s%s/' % (  # '/Users/hansem/Dropbox (MIT)/MPong/figs/mpong_phys/redo_paper_scratch_202202'
            figoutpath_base, self.subject_id, self.timebinsize, self.neural_data_to_use,
            self.masks_to_test_suffix, self.condition)

        self.save_prefix = '%2.2f' % self.train_size

        if os.path.isdir(self.figoutpath) is False:
            os.makedirs(self.figoutpath)
        return

    def get_data_filename(self):
        default_save_path = '../data/'
        # default_save_path = '/Users/hansem/Dropbox (MIT)/MPong/phys/results/old_from_rishi_om/decode_results_old/' # without 'old' new dataset with ego vs allo
        # default_save_path = '/om/user/rishir/lib/MentalPong/phys/results/decode_results/'
        preproc_suffix = '_%s%d' % (self.preprocess_func, self.ncomp) if self.preprocess_func != 'none' else '' # is not
        save_fn_suffix = '%s_%s_%s_%dms_%2.2f_%s%s' % (self.subject_id,
                                                            self.condition,
                                                            self.masks_to_test_suffix,
                                                            self.timebinsize,
                                                            self.train_size, self.neural_data_to_use, preproc_suffix)
        save_fn = '%s/decode_%s.pkl' % (default_save_path, save_fn_suffix)
        
        fns = glob(save_fn)

        return fns[0]

    def load_data(self):
        fn = self.get_data_filename()
        print(fn)
        dat = pk.load(open(fn, 'rb'))
        self.dat = dat
        self.mask_fn_base = self.masks_to_test_suffix

        self.beh_to_decode = dat['beh_to_decode']
        self.nbehvar = len(self.beh_to_decode)
        self.mask_conditions = dat['mask_conditions']
        self.nmasks = len(self.mask_conditions)
        self.niter = self.dat['decoder_specs']['niter']
        self.ncond = self.dat['ncond']

        self.multi_timepoint_masks = ['start_end_pad0', 'start_occ_pad0', 'occ_end_pad0']
        self.single_timepoint_masks = ['start_pad0_roll', 'occ_pad0_roll', 'f_pad0_roll']

        if self.mask_fn_base in self.multi_timepoint_masks:
            zero_roll_mask_fn = '%s_roll0' % self.mask_fn_base
            zero_roll_mask_pair = [zero_roll_mask_fn, zero_roll_mask_fn.replace('_roll0', '')]
            self.center_mask_idx = self.mask_conditions.index(zero_roll_mask_pair)
        elif self.mask_fn_base in self.single_timepoint_masks:
            self.center_mask_idx = None
        return

    def load_ground_truth_data(self):

        data = phys_data_utils.load_neural_dataset('random', timebinsize=self.timebinsize, compute_egocentric=True) ###
        self.ground_truth = {}
        for fk in self.beh_to_decode:
            self.ground_truth[fk] = np.array(data['behavioral_responses']['occ'][fk])

        self.ground_truth['masks'] = deepcopy(data['masks']['occ'])
        self.ntime = self.ground_truth['masks']['start_end_pad0'].shape[1]

        # self.shuffled_perf = get_shuffled_null()
        return

    def get_predictions_unrolled_and_remasked_base(self, time_idx=None):
        if time_idx is None:
            time_idx = self.center_mask_idx

        def get_predictions_unflattened(res):
            y_true, y_pred = [], []
            for i in range(self.niter):
                rt = res['y_true_dist'][i]
                rp = res['y_pred_dist'][i]
                y_true.append(phys_utils.unflatten_to_3d_mat(rt, res['unflatten_res']))
                y_pred.append(phys_utils.unflatten_to_3d_mat(rp, res['unflatten_res']))
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            return y_true, y_pred

        def remask(y_true, y_pred):
            masks_to_align_to = ['start_occ_pad0', 'occ_end_pad0', 'start_end_pad0',
                                 'start_pad0_roll0', 'occ_pad0_roll0', 'f_pad0_roll0']
            res_per_mask = {'true': {}, 'pred': {}}
            for mask_fn_ in masks_to_align_to:
                m = np.array(self.ground_truth['masks'][mask_fn_])
                yt_remasked, yp_remasked = [], []
                for i in range(self.niter):
                    yt_remasked.append([phys_utils.apply_mask(y_true[i, :, :, bi], m) for bi in range(self.nbehvar)])
                    yp_remasked.append([phys_utils.apply_mask(y_pred[i, :, :, bi], m) for bi in range(self.nbehvar)])
                res_per_mask['true'][mask_fn_] = np.array(yt_remasked)
                res_per_mask['pred'][mask_fn_] = np.array(yp_remasked)
            return res_per_mask

        y_true_, y_pred_ = get_predictions_unflattened(self.dat['res_decode'][time_idx])
        return remask(y_true_, y_pred_)

    def get_performance_unrolled_and_remasked_base(self, res_remasked):

        def get_r_and_mae_over_t(x_cxt, y_cxt):
            mae_t = np.nanmean(np.abs(x_cxt - y_cxt), axis=0)
            mse_t = np.nanmean((x_cxt - y_cxt) ** 2, axis=0)
            # not worth computing r over t, since variance in beh variable changes dramatically
            r_t = np.ones(mse_t.shape) * np.nan
            # r_t = np.array([nnan_pearsonr(x_cxt[:, t_i], y_cxt[:, t_i])[0] for t_i in range(self.ntime)])
            return r_t, mae_t, mse_t

        def get_r_and_mae_pooling_t(x_cxt, y_cxt):
            mae_mu = np.nanmean(np.abs(x_cxt - y_cxt), axis=(0, 1))
            mse_mu = np.nanmean((x_cxt - y_cxt) ** 2, axis=(0, 1))
            r_mu = nnan_pearsonr(x_cxt.flatten(), y_cxt.flatten())[0]
            return r_mu, mae_mu, mse_mu

        def get_all_metrics_per_entry(y_true_cxt, y_pred_cxt):
            y_true_cxt_startalign = phys_utils.realign_masked_data(y_true_cxt, align_to_start=True)
            y_true_cxt_endalign = phys_utils.realign_masked_data(y_true_cxt, align_to_start=False)
            y_pred_cxt_startalign = phys_utils.realign_masked_data(y_pred_cxt, align_to_start=True)
            y_pred_cxt_endalign = phys_utils.realign_masked_data(y_pred_cxt, align_to_start=False)

            r_mu, mae_mu, mse_mu = get_r_and_mae_pooling_t(y_true_cxt_startalign, y_pred_cxt_startalign)
            r_t, mae_t, mse_t = get_r_and_mae_over_t(y_true_cxt_startalign, y_pred_cxt_startalign)
            r_t2, mae_t2, mse_t2 = get_r_and_mae_over_t(y_true_cxt_endalign, y_pred_cxt_endalign)

            return {
                'r_t': r_t, 'mae_t': mae_t, 'mse_t': mse_t,
                'r_t2': r_t2, 'mae_t2': mae_t2, 'mse_t2': mse_t2,
                'r_mu': r_mu, 'mae_mu': mae_mu
            }

        def get_performance_over_time(y_true_ixbxcxt, y_pred_ixbxcxt):

            metrics = {
                'r_t': np.ones((self.niter, self.ntime, self.nbehvar)) * np.nan,
                'mae_t': np.ones((self.niter, self.ntime, self.nbehvar)) * np.nan,
                'mse_t': np.ones((self.niter, self.ntime, self.nbehvar)) * np.nan,
                'r_t2': np.ones((self.niter, self.ntime, self.nbehvar)) * np.nan,
                'mae_t2': np.ones((self.niter, self.ntime, self.nbehvar)) * np.nan,
                'mse_t2': np.ones((self.niter, self.ntime, self.nbehvar)) * np.nan,
                'r_mu': np.ones((self.niter, self.nbehvar)) * np.nan,
                'mae_mu': np.ones((self.niter, self.nbehvar)) * np.nan,
                'mse_mu': np.ones((self.niter, self.nbehvar)) * np.nan,
            }

            for i_i in range(self.niter):
                for b_i in range(self.nbehvar):
                    y_true_cxt = y_true_ixbxcxt[i_i, b_i, :, :]
                    y_pred_cxt = y_pred_ixbxcxt[i_i, b_i, :, :]
                    res_curr = get_all_metrics_per_entry(y_true_cxt, y_pred_cxt)
                    for fk in res_curr.keys():
                        if metrics[fk].ndim == 3:
                            metrics[fk][i_i, :, b_i] = res_curr[fk]
                        elif metrics[fk].ndim == 2:
                            metrics[fk][i_i, b_i] = res_curr[fk]

            return metrics

        metrics_all = {}
        for mask_fn in res_remasked['true'].keys():
            y_true_ixbxcxt_, y_pred_ixbxcxt_ = res_remasked['true'][mask_fn], res_remasked['pred'][mask_fn]
            metrics_all[mask_fn] = get_performance_over_time(y_true_ixbxcxt_, y_pred_ixbxcxt_)
        return metrics_all

    def get_metrics_unrolled_and_remasked(self, time_idx=None):
        if time_idx is None:
            time_idx = self.center_mask_idx
        res_remasked = self.get_predictions_unrolled_and_remasked_base(time_idx=time_idx)
        return self.get_performance_unrolled_and_remasked_base(res_remasked)

    def get_metrics_center_bin(self):
        dat = self.dat
        unflatten_res = dat['res_decode'][self.center_mask_idx]['unflatten_res']
        dec_plot_data = {}
        for fk in ['y_pred_dist', 'y_true_dist']:
            x_ = dat['res_decode'][self.center_mask_idx][fk]
            dec_plot_data[fk] = []
            for xi in x_:
                # unflattening re-aligns every trial to start at time 0
                dec_plot_data[fk].append(phys_utils.unflatten_to_3d_mat(xi, unflatten_res))
            dec_plot_data[fk] = np.array(dec_plot_data[fk])
        return dec_plot_data

    def get_metrics_asynchronous_base(self, x_mask_prefix, label_mask_fn, time_rolls, beh_idx):
        def get_metrics_for_asynchronous_masks(mask_condition_curr_):
            # how well does N(t) predict y_tf? (r)
            # how well does y(t) predict y_tf? (r_control)
            #  r(N(t), y_tf | y(t)) (r_partial)

            mask_idx = self.mask_conditions.index(mask_condition_curr_)
            r = self.dat['res_decode'][mask_idx]['r_dist'][:, beh_idx]
            mae = self.dat['res_decode'][mask_idx]['mae_dist'][:, beh_idx]

            yt = self.dat['res_decode'][mask_idx]['y_true_dist'][:, :, beh_idx]
            yp = self.dat['res_decode'][mask_idx]['y_pred_dist'][:, :, beh_idx]
            z = self.ground_truth[self.beh_to_decode[beh_idx]]
            mask_n = mask_condition_curr_[0]
            mask = self.ground_truth['masks'][mask_n]
            zm = np.nanmean(phys_utils.apply_mask(z, mask), axis=1)

            r_control_, p_control_ = nnan_pearsonr(zm, yt[0, :])  # ground truth is same on all iter
            r_control = np.ones(r.shape) * r_control_
            p_control = np.ones(r.shape) * p_control_
            r_partial = np.array([self.partial_corr_wrapper_2(yt[i_, :], yp[i_, :], zm)[0] for i_ in range(self.niter)])

            return {'r': r, 'mae': mae, 'r_partial': r_partial, 'r_control': r_control, 'p_control': p_control}

        ntimes = time_rolls.shape[0]
        metrics = {
            'r_dist': np.ones((ntimes, self.niter)) * np.nan,
            'mae_dist': np.ones((ntimes, self.niter)) * np.nan,
            'r_partial_dist': np.ones((ntimes, self.niter)) * np.nan,
            'r_control_dist': np.ones((ntimes, self.niter)) * np.nan,
            'p_control_dist': np.ones((ntimes, self.niter)) * np.nan,
        }

        for ti, t in enumerate(time_rolls):
            x_mask_fn = '%s%d' % (x_mask_prefix, t)
            mask_condition_curr = [x_mask_fn, label_mask_fn]
            if mask_condition_curr not in self.mask_conditions:
                continue
            res_ = get_metrics_for_asynchronous_masks(mask_condition_curr)
            for fk in res_.keys():
                metrics['%s_dist' % fk][ti, :] = np.array(res_[fk])

        return metrics

    def get_metrics_asynchronous(self, max_time_roll=50, beh_idx=1):
        time_rolls = np.arange(-max_time_roll, max_time_roll)
        x_mask_prefix = '%sroll' % self.mask_fn_base.split('roll')[0]
        y_mask_fns = ['start_pad0_roll0', 'occ_pad0_roll0', 'half_pad0_roll0', 'f_pad0_roll0']
        res_unrolled_dict = {'time_rolled': [i * self.timebinsize for i in time_rolls]}
        for fk in y_mask_fns:
            res_unrolled_dict[fk] = self.get_metrics_asynchronous_base(x_mask_prefix,
                                                                       fk, time_rolls, beh_idx)
        return res_unrolled_dict

    def get_colormap_for_decoded_variables(self):
        # ['ball_pos_x_TRUE', 'ball_pos_y_TRUE',  'ball_pos_dx_TRUE', 'ball_pos_dy_TRUE', 'ball_pos_dspeed_TRUE', 'ball_pos_dtheta_TRUE',
        #  'paddle_pos_y', 'joy',
        #  'eye_v', 'eye_h',  'eye_dv', 'eye_dh',  'eye_dtheta', 'eye_dspeed',
        #  't_from_start', 't_from_occ', 't_from_end',
        #  'ball_final_y', 'ball_initial_y', 'ball_occ_start_y']

        cmap = plt.get_cmap('Dark2', 8)
        self.beh_var_groups = {
            'ball_var_idx': [0, 1, 2, 3, 4, 5, 6, 7, 27, 30, 31, 32], # [0, 1, 2, 3, 4, 5], ###
            'eye_var_idx': [10,11,12,13,14,15, 16, 17, 18, 19], # [8, 9, 10, 11, 12, 13],
            'time_var_idx': [24, 25, 26], # [14, 15, 16],
            'target_var_idx': [8, 9, 19, 20, 21, 22, 23, 28, 29], # [6, 7, 17, 18, 19],
        }

        self.dec_cols = {
            'ball_var_idx': cmap(0),
            'eye_var_idx': cmap(1),
            'time_var_idx': cmap(2),
            'target_var_idx': cmap(3),
        }
        return

    @staticmethod
    def partial_corr_wrapper_2(x, y, z):
        df_pc = pd.DataFrame({'x': x, 'y': y, 'cv': z})
        tmp = pg.partial_corr(data=df_pc, x='x', y='y', covar='cv')
        return tmp['r']['pearson'], tmp['p-val']['pearson']

    @staticmethod
    def partial_corr_wrapper(x, y, z):
        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)
        if y.ndim == 1:
            y = np.expand_dims(y, axis=1)
        if z.ndim == 1:
            z = np.expand_dims(z, axis=1)
        partial_corr = analysis_utils.partial_corr
        c = np.concatenate((x, y, z), axis=1)
        r_, p_ = partial_corr(c, add_offset=True, return_pval=True) # ,return_CI=True)    , ci_
        return r_[0, 1], p_[0, 1] # , ci_[0,1]

    """ scatter and bar plots for decode performance """

    def partial_corr_pairwise_comparison_base(self, i1, i2_list, time_idx=None):
        r_partial_dist_all = []
        if not isinstance(i2_list, list):
            i2_list = [i2_list]

        if time_idx is None:
            time_idx = self.center_mask_idx

        x0_all = self.dat['res_decode'][time_idx]['y_pred_dist'][:, :, i1]
        y0_all = self.dat['res_decode'][time_idx]['y_true_dist'][:, :, i1]

        r_raw_list = [nnan_pearsonr(x0, y0)[0] for x0, y0 in zip(x0_all, y0_all)]
        r_raw_mu, r_raw_sd = np.nanmean(r_raw_list), np.nanstd(r_raw_list)
        r_partial_dist_all.append(r_raw_list)
        # print(len(r_raw_list))

        r_partial_each_mu, r_partial_each_sd = [], []
        for i2 in i2_list:
            y1_all = self.dat['res_decode'][time_idx]['y_true_dist'][:, :, i2]
            r_partial_dist = [self.partial_corr_wrapper_2(x0, y0, y1)[0] for x0, y0, y1 in zip(x0_all, y0_all, y1_all)]
            # print(len(r_partial_dist))
            r_partial_each_mu.append(np.nanmean(r_partial_dist))
            r_partial_each_sd.append(np.nanstd(r_partial_dist))
            r_partial_dist_all.append(r_partial_dist)

        return {'r_raw_mu': r_raw_mu, 'r_raw_sd': r_raw_sd,
                'r_partial_each_mu': r_partial_each_mu, 'r_partial_each_sd': r_partial_each_sd,
                'r_partial_dist': r_partial_dist_all}


    def partial_corr_pairwise_comparison_base_v2(self, i1, i2_list, time_idx=None):
        """ running on mean over iterations/splits"""

        if not isinstance(i2_list, list):
            i2_list = [i2_list]

        if time_idx is None:
            time_idx = self.center_mask_idx

        x0 = self.dat['res_decode'][time_idx]['y_pred_mu'][:, [i1]]
        y0 = self.dat['res_decode'][time_idx]['y_true_mu'][:, [i1]]
        y1 = self.dat['res_decode'][time_idx]['y_true_mu'][:, i2_list]

        r_raw, p_raw = nnan_pearsonr(x0, y0)
        r_partial, p_partial = self.partial_corr_wrapper(x0, y0, y1)

        r_partial_each, p_partial_each = [], []
        for i2 in i2_list:
            y1 = self.dat['res_decode'][time_idx]['y_true_mu'][:, [i2]]
            r, p = self.partial_corr_wrapper(x0, y0, y1)
            r_partial_each.append(r)
            p_partial_each.append(p)

        res = {'r_raw': r_raw, 'p_raw': p_raw,
               'r_partial': r_partial, 'p_partial': p_partial,
               'r_partial_each': r_partial_each, 'p_partial_each': p_partial_each,
               }
        return res

    def plot_decode_performance_for_all_behvars(self, time_idx=None, file_suffix=''):
        if time_idx is None:
            time_idx = self.center_mask_idx

        f, axes = plt.subplots(2, 1, figsize=(6.5, 9))
        alpha = 0.2
        var_idx = [8, 9, 10, 11, 12, 13, 14, 15, 16,6, 7]
        # old
        # var_idx = [0, 1, # ball x y
        # 8, 9, 10, 11, 12, 13, # eye
        # 14, 15, 16, # time?
        # 6, 7] # paddle

        # var_idx = [10, 11, 12, 13, 14, 15, 24, 25, 26, 8, 9]
        # # 0 ['ball_pos_x_TRUE', 'ball_pos_y_TRUE',  'ball_pos_dx_TRUE', 'ball_pos_dy_TRUE', 'ball_pos_dspeed_TRUE', 'ball_pos_dtheta_TRUE',
        # # 6 7 ball_pos_ego_x_TRUE ball_pos_ego_y_TRUE
        # # 8 9 'paddle_pos_y', 'joy',
        # # 10, 11, 12, 13, 14, 15 'eye_v', 'eye_h',  'eye_dv', 'eye_dh',  'eye_dtheta', 'eye_dspeed',
        # # 24, 25, 26 't_from_start', 't_from_occ', 't_from_end',
        # #  'ball_final_y', 'ball_initial_y', 'ball_occ_start_y']

        xax, xax_labels = [], []
        for mi, mfn in enumerate(['r', 'mae']):
            ax = axes[mi]
            # Initialize lists to store combined data for all variables
            x_values = []
            y_values = []
            var_names = []

            # Generate x-locations (could be just var_idx or a sequential range)
            x_locations = range(len(var_idx))

            for vi, i in enumerate(var_idx):
                r_mu = self.dat['res_decode'][time_idx]['%s_mu' % mfn][i]
                r_sig = self.dat['res_decode'][time_idx]['%s_sd' % mfn][i]
                r_dist = self.dat['res_decode'][time_idx]['%s_dist' % mfn][:,i]

                # Append the data and corresponding x-values
                x_values.extend([x_locations[vi]] * len(r_dist))
                y_values.extend(r_dist)
                var_names.extend([i] * len(r_dist))  # Track which variable this corresponds to (optional)

                tmp_fk = [fk for fk in self.beh_var_groups.keys() if i in self.beh_var_groups[fk]]
                col_tmp = self.dec_cols[tmp_fk[0]]
                # ax.bar(vi, r_mu, yerr=r_sig, color=col_tmp)
                xax.append(vi)
                xax_labels.append(self.beh_to_decode[i])

            # Create a DataFrame for the combined data
            df = pd.DataFrame({'x': x_values, 'y': y_values, 'variable': var_names})

            sns.boxplot(x="x", y='y',
                        data=df, ax=ax, whis=100,
                        boxprops=dict(alpha=alpha), capprops=dict(alpha=alpha),
                        whiskerprops=dict(alpha=alpha), medianprops=dict(alpha=alpha))

            sns.swarmplot(data=df, x='x', y='y', ax=ax, size=2)

            np.savetxt('decode_performance_for_all_behvars_%s.csv' % (mfn), df, fmt='%5.3f', delimiter=',', newline='\n')

            rr_gpu.make_axis_nice(ax, offset=0)
            ax.set_xticks(xax)
            ax.set_xticklabels(xax_labels, rotation=90)
            if mfn == 'r':
                # ax.set_ylim([0.6, 1])
                ax.set_ylim([0, 1])
            elif mfn == 'mae':
                ax.set_ylim([0, 4])
                # ax.set_ylim([0, 10])
        plt.tight_layout()

        outfn = '%s/%s_decbar_all_mask%d%s.pdf' % (self.figoutpath, self.save_prefix,
                                                   time_idx, file_suffix)
        f.savefig(outfn)
        return

    def plot_decode_performance_xy(self, decode_var_idx, time_idx=None, file_suffix=''):
        f, axes = plt.subplots(1, 1, gridspec_kw={'width_ratios': [1]}, figsize=(4, 4))
        if time_idx is None:
            time_idx = self.center_mask_idx

        ax = axes

        cols = [plt.cm.Dark2(1), plt.cm.Dark2(2)]

        for i_col, idx in enumerate(decode_var_idx):
            x0 = self.dat['res_decode'][time_idx]['y_true_mu'][:, idx]
            y0 = self.dat['res_decode'][time_idx]['y_pred_mu'][:, idx]
            r_mu = self.dat['res_decode'][time_idx]['r_mu'][idx]
            r_sd = self.dat['res_decode'][time_idx]['r_sd'][idx]
            ax = sns.regplot(x=x0, y=y0, ax=ax, marker='.', color=cols[i_col],
                             scatter_kws={'alpha': 0.1}, line_kws={'linestyle': '--', 'color': 'k', 'lw': 1},
                             label='$\it{r=%2.2f±%2.2f}$' % (r_mu, r_sd))
            np.savetxt('%s_x.csv'%(idx),x0, fmt='%5.3f', delimiter=',', newline='\n')
            np.savetxt('%s_y.csv' % (idx), y0, fmt='%5.3f', delimiter=',', newline='\n')

        #     ax.plot(x0, y0, '.', color=self.dec_cols['ball_var_idx'], alpha=0.2)
        # lims = [np.nanmin([x0, y0]), np.nanmax([x0, y0])]
        # ax.set_xlim(lims)
        # ax.set_ylim(lims)
        plt.legend()
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        rr_gpu.make_axis_nice(ax)

        if decode_var_idx[0]==0 or decode_var_idx[0]==1: # allocentric
            file_suffix = 'allocentric'
        elif decode_var_idx[0]==7 or decode_var_idx[0]==6: # egocentric
            file_suffix = 'egocentric'

        outfn = '%s/%s_decode_xy_mask%d%s.pdf' % (self.figoutpath, self.save_prefix,
                                                  time_idx, file_suffix)
        f.savefig(outfn)
        return

    def plot_decode_performance_per_behvar_with_partial(self, decode_var_idx, time_idx=None, file_suffix=''):
        if time_idx is None:
            time_idx = self.center_mask_idx
        f, ax = plt.subplots(1, 1, gridspec_kw={'width_ratios': [1]}, figsize=(10, 4))
        # f, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 2]}, figsize=(10, 4))
        # ax = axes[0]
        # x0 = self.dat['res_decode'][time_idx]['y_true_mu'][:, decode_var_idx]
        # y0 = self.dat['res_decode'][time_idx]['y_pred_mu'][:, decode_var_idx]
        # ax.plot(x0, y0, '.', color=self.dec_cols['ball_var_idx'], alpha=0.2)
        # lims = [np.nanmin([x0, y0]), np.nanmax([x0, y0])]
        # ax.set_xlim(lims)
        # ax.set_ylim(lims)
        # ax.set_xlabel('True')
        # ax.set_ylabel('Predicted')
        # rr_gpu.make_axis_nice(ax)

        # ax = axes[1]

        # partial_corr_var_idx = [10, 11, 12, 13, 14, 15, # eye
        #                         24, 25, 26, # time
        #                         8, 9] # paddle

        # old
        partial_corr_var_idx = [8, 9, 10, 11, 12, 13, # eye
                                14, 15, 16, # time
                                6, 7] # paddle

        # Initialize lists to store combined data for all variables
        x_values = []
        y_values = []
        var_names = []

        plot_offset = 1 # 2
        alpha = 0.2
        # Raw
        res = self.partial_corr_pairwise_comparison_base(decode_var_idx, partial_corr_var_idx)
        x_values.extend([0] * len(res['r_partial_dist'][0]))
        y_values.extend(res['r_partial_dist'][0])
        var_names.extend([0]* len(res['r_partial_dist'][0]))  # Track which variable this corresponds to (optional)
        # ax.bar(0, res['r_raw_mu'], yerr=res['r_raw_sd'],
        #        color=self.dec_cols['ball_var_idx'],
        #        edgecolor=self.dec_cols['ball_var_idx'],
        #        linewidth=4)
        xax, xax_labels = [], []
        xax.append(0)
        # xax_labels.append('Raw')
        # Partial
        for i, r_i in enumerate(res['r_partial_each_mu']):
            idx_orig = partial_corr_var_idx[i]
            tmp_fk = [fk for fk in self.beh_var_groups.keys() if idx_orig in self.beh_var_groups[fk]]
            col_tmp = self.dec_cols[tmp_fk[0]]
            x_values.extend([i + plot_offset] * len(res['r_partial_dist'][i+1]))
            y_values.extend(res['r_partial_dist'][i+1])
            var_names.extend([i + plot_offset] * len(res['r_partial_dist'][i+1]))  # Track which variable this corresponds to (optional)
            # ax.bar(i + plot_offset, r_i, yerr=res['r_partial_each_sd'][i],
            #        color='w', edgecolor=col_tmp, linewidth=2, hatch='//')
            xax.append(i + plot_offset)
            # xax_labels.append(self.beh_to_decode[idx_orig])

        # Create a DataFrame for the combined data
        df = pd.DataFrame({'x': x_values, 'y': y_values, 'variable': var_names})

        sns.boxplot(x="x", y='y',
                    data = df, ax = ax, whis = 100,
                    boxprops = dict(alpha=alpha), capprops = dict(alpha=alpha),
                    whiskerprops = dict(alpha=alpha), medianprops = dict(alpha=alpha))

        sns.swarmplot(data=df, x='x', y='y', ax=ax, size=2)

        np.savetxt('decode_performance_per_behvar_with_partial_%s.csv' % (decode_var_idx), df, fmt='%5.3f', delimiter=',', newline='\n')

        rr_gpu.make_axis_nice(ax, offset=0)
        ax.set_xticks(xax)
        ax.set_xticklabels(xax_labels, rotation=90)
        ax.set_ylim([0, 1])
        plt.tight_layout()
        f.suptitle(self.beh_to_decode[decode_var_idx])
        outfn = '%s/%s_decbar_partial_beh%dmask%d%s.pdf' % (self.figoutpath, self.save_prefix,
                                                            decode_var_idx,
                                                            time_idx, file_suffix)
        f.savefig(outfn)
        return

    def plot_decode_performance_unrolled_over_time(self, time_idx=None, compute_egocentric=False):

        if compute_egocentric:  ###
            beh_idx_mat = [6, 7]
            file_suffix = 'egocentric'
        else:
            beh_idx_mat = [0, 1]
            file_suffix = 'allocentric'
        def plot_average_metric(res):
            cmap = plt.get_cmap('Dark2', 8) # green, red, blue, magenta, brightGreen, yellow, 6brown, 7grey
            f, axes = plt.subplots(1, 3, figsize=(7.5, 3))
            for ax_i, met in enumerate(['r_mu', 'mae_mu', 'mse_mu']):
                ax = axes[ax_i]
                for beh_idx in beh_idx_mat: ####
                    mu, sig = [], []
                    x_for_stat = []
                    for mi, mask_fn in enumerate(['start_occ_pad0', 'occ_end_pad0']):
                        x = res[mask_fn][met][:, beh_idx]
                        mu.append(np.nanmean(x, axis=0))
                        sig.append(np.nanstd(x, axis=0))
                        x_for_stat.append(x)

                    # ranksums
                    # stat_object = ranksums(x_for_stat[0],x_for_stat[1])
                    stat_object = wilcoxon(x_for_stat[1])
                    p_value = stat_object.pvalue

                    # # permutation_test
                    # p_value = permutation_test(x_for_stat[0],x_for_stat[1],
                    #                            method='approximate',
                    #                            num_rounds=1000,
                    #                            seed=0)

                    print(mu)
                    print(sig)
                    ax.errorbar(range(2), mu, yerr=sig, fmt='-o', mfc='w', mec=cmap(beh_idx), color=cmap(beh_idx),label=f'p={p_value:.2E}') # ,label='$\it{p=%2.6f}$' % p_value)
                    ax.legend()
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels(['Visible \n Epoch', 'Occluded \n Epoch'])
                    ax.set_xlim([-0.25, 1.25])
                    if met == 'r_mu':
                        ax.set_ylim([0, 1])
                    else:
                        ax.set_ylim([0, 10])
                    rr_gpu.make_axis_nice(ax)

            plt.tight_layout()
            return f, axes

        def plot_timecourse_of_error(res,subject_id):
            f, axes = plt.subplots(1, 4, figsize=(8, 4))
            ylim_all = [0, 7]
            for mfi, mask_fn in enumerate(['start_occ_pad0', 'occ_end_pad0']):
                rmse_alignstart = (res[mask_fn]['mse_t'][:, :, beh_idx_mat[0]] + res[mask_fn]['mse_t'][:, :, beh_idx_mat[1]]) ** 0.5 ####
                rmse_alignend = (res[mask_fn]['mse_t2'][:, :, beh_idx_mat[0]] + res[mask_fn]['mse_t2'][:, :, beh_idx_mat[1]]) ** 0.5

                if 'start' in mask_fn:
                    min_timesteps = 15
                else:
                    min_timesteps = 14

                ax = axes[mfi * 2]
                t = np.arange(0, min_timesteps, 1)
                ax.errorbar(t, np.nanmean(rmse_alignstart[:, t], axis=0),
                            np.nanstd(rmse_alignstart[:, t], axis=0), fmt='-ko')
                ax.set_ylim(ylim_all)
                ax.axvline(0, linestyle='--', color=[0.5, 0.5, 0.5])
                rr_gpu.make_axis_nice(ax)

                np.savetxt('%s_%s_alignstart_t.csv'%(subject_id,mask_fn), t, fmt='%5.3f', delimiter=',', newline='\n')
                np.savetxt('%s_%s_alignstart_mean.csv'%(subject_id,mask_fn), np.nanmean(rmse_alignstart[:, t], axis=0), fmt='%5.3f', delimiter=',', newline='\n')
                np.savetxt('%s_%s_alignstart_std.csv'%(subject_id,mask_fn), np.nanstd(rmse_alignstart[:, t], axis=0), fmt='%5.3f', delimiter=',', newline='\n')

                ax = axes[mfi * 2 + 1]
                t = np.arange(-min_timesteps, 0, 1)
                ax.errorbar(t + 1, np.nanmean(rmse_alignend[:, t], axis=0),
                            np.nanstd(rmse_alignend[:, t], axis=0), fmt='-ko')
                ax.set_ylim(ylim_all)
                ax.axvline(0, linestyle='--', color=[0.5, 0.5, 0.5])
                rr_gpu.make_axis_nice(ax)

                np.savetxt('%s_%s_alignend_t.csv'%(subject_id,mask_fn), t + 1, fmt='%5.3f', delimiter=',', newline='\n')
                np.savetxt('%s_%s_alignend_mean.csv'%(subject_id,mask_fn), np.nanmean(rmse_alignend[:, t], axis=0), fmt='%5.3f', delimiter=',', newline='\n')
                np.savetxt('%s_%s_alignend_std.csv'%(subject_id,mask_fn), np.nanstd(rmse_alignend[:, t], axis=0), fmt='%5.3f', delimiter=',', newline='\n')

            for i in [1, 2, 3]:
                axes[i].axes.get_yaxis().set_visible(False)

            return f, axes

        res_ = self.get_metrics_unrolled_and_remasked(time_idx=time_idx)

        f, axes = plot_average_metric(res_)
        outfn = '%s/%s_decode_perf_over_epochs_%s.pdf' % (self.figoutpath, self.save_prefix, file_suffix)
        f.savefig(outfn)

        f, axes = plot_timecourse_of_error(res_,self.subject_id)
        outfn = '%s/%s_decode_perf_over_time_%s.pdf' % (self.figoutpath, self.save_prefix, file_suffix)
        f.savefig(outfn)
        return

    """ for example visualization of timepoints within a condition. """

    def plot_timecourse_of_decoder_output_base(self, dec_plot_data, tr_idx, num_timebins_to_plot=5):
        time_idx_all = np.nonzero(np.isfinite(dec_plot_data['y_pred_dist'][0, tr_idx, :, 0]))[0][:-1] # somehow all nan
        idx = np.round(np.linspace(0, len(time_idx_all) - 2, num_timebins_to_plot)).astype(int)
        time_idx_sel = time_idx_all[idx]

        f, axes = plt.subplots(1, num_timebins_to_plot, figsize=(num_timebins_to_plot * 2.5, 3))
        for ti, time_idx in enumerate(time_idx_sel):
            ax = axes[ti]

            x_p = dec_plot_data['y_pred_dist'][:, tr_idx, time_idx, 0]
            y_p = dec_plot_data['y_pred_dist'][:, tr_idx, time_idx, 1]

            x_t = dec_plot_data['y_true_dist'][0, tr_idx, time_idx, 0]
            y_t = dec_plot_data['y_true_dist'][0, tr_idx, time_idx, 1]

            x_t_all = np.squeeze(dec_plot_data['y_true_dist'][0, tr_idx, :, 0])
            y_t_all = np.squeeze(dec_plot_data['y_true_dist'][0, tr_idx, :, 1])

            ax.hist2d(x_p, y_p, bins=np.arange(-12, 12, 0.66), cmap=plt.cm.Greens)
            ax.plot(x_t_all, y_t_all, 'k--')
            ax.plot(x_t, y_t, 'wo', mec='k')
            ax.plot(np.nanmean(x_p), np.nanmean(y_p), 'r*')

            rr_gpu.add_mpong_frame_to_axis(ax)

            ax.set_xticks([])
            ax.set_yticks([])

            ax.set_title('t=%d' % time_idx)
            ax.axis('off')
        plt.tight_layout()
        f.suptitle('tr_idx %d' % tr_idx, y=1.01)
        outfn = '%s/%s_decode_example_over_time_tr%d.pdf' % (self.figoutpath, self.save_prefix, tr_idx)
        f.savefig(outfn)
        return

    def plot_timecourse_of_decoder_output(self):
        dec_plot_data = self.get_metrics_center_bin()
        del_ = np.nanmean(np.abs(dec_plot_data['y_pred_dist'] - dec_plot_data['y_true_dist']), axis=(0, 2))
        plot_idx = np.argsort(del_[:, 0] + del_[:, 1])
        for i in plot_idx[:10]:
            self.plot_timecourse_of_decoder_output_base(dec_plot_data, i, num_timebins_to_plot=8)
        return

    """ decoding as a function of timepoints (single timepoint train) """

    def plot_decode_performance_over_time(self, res_unrolled_dict_all):
        cmap = plt.get_cmap('plasma', 6)
        plot_metrics = ['r_dist', 'mae_dist', 'r_partial_dist', 'r_control_dist']
        mask_choices = ['start_pad0_roll0', 'occ_pad0_roll0', 'half_pad0_roll0', 'f_pad0_roll0']
        npanels = len(plot_metrics)
        f, axes = plt.subplots(npanels, 1, figsize=(4, 3 * npanels))

        for i, mfn in enumerate(plot_metrics):
            for mi, mask_fn in enumerate(mask_choices):
                dist_metric = res_unrolled_dict_all[mask_fn][mfn]
                mu = np.nanmean(dist_metric, axis=1)
                sig = np.nanstd(dist_metric, axis=1)
                axes[i].errorbar(res_unrolled_dict_all['time_rolled'], mu, yerr=sig, color=cmap(mi + 1))
            rr_gpu.make_axis_nice(axes[i], offset=0)
            axes[i].axvline(0, linestyle=':', color='k')

        axes[0].set_ylim([-0.1, 1])
        axes[1].set_ylim([0, 10])
        axes[2].set_ylim([-0.1, 1])
        plt.tight_layout()
        outfn = '%s/%s_decode_perf_over_time.pdf' % (self.figoutpath, self.save_prefix)
        f.savefig(outfn)
        return

    def plot_decode_performance_at_tstar(self, res_unrolled_dict_all):
        # t_star is the first time at which control tests pass

        mask_choices = ['start_pad0_roll0', 'occ_pad0_roll0', 'half_pad0_roll0', 'f_pad0_roll0']
        plot_metrics = ['r_dist', 'mae_dist', 'r_partial_dist', 'r_control_dist']
        npanels = len(plot_metrics)

        def get_t_star(label_mask):
            r_ctrl = np.nanmean(res_unrolled_dict_all[label_mask]['r_control_dist'], axis=1)
            p_ctrl = np.nanmean(res_unrolled_dict_all[label_mask]['p_control_dist'], axis=1)
            t_star1 = np.nonzero(p_ctrl > 0.01)[0]
            t_star2 = np.nonzero(r_ctrl > 0.1)[0]
            if len(t_star1) > 0:
                t_star1 = t_star1[-1]
            if len(t_star2) > 0:
                t_star2 = t_star2[-1]
            return t_star1, t_star2

        def plot_bar_plot_per_t_star(t_star):
            f, axes = plt.subplots(1, npanels, figsize=(npanels * 3, 2))
            for i, mfn in enumerate(plot_metrics):
                for mi, mask_fn in enumerate(mask_choices):
                    dist_metric = res_unrolled_dict_all[mask_fn][mfn]
                    mu = np.nanmean(dist_metric, axis=1)
                    sig = np.nanstd(dist_metric, axis=1)
                    axes[i].bar(mi, mu[t_star], yerr=sig[t_star], color='k', width=0.6)
                    axes[i].set_title('%s %dms' % (mfn, res_unrolled_dict_all['time_rolled'][t_star]))
                    rr_gpu.make_axis_nice(axes[i])
            plt.tight_layout()
            return f, axes

        for mask_fn_1 in mask_choices:
            t_star1_curr, t_star2_curr = get_t_star(mask_fn_1)
            for ts_i, t_star_curr in enumerate([t_star1_curr, t_star2_curr]):
                if t_star_curr:
                    f, axes = plot_bar_plot_per_t_star(t_star_curr)
                    tag = 't_star%d_%s' % (ts_i, mask_fn_1)
                    f.suptitle(tag)
                    f.savefig('%s/%s.pdf' % (self.figoutpath, tag))

        return

    def plot_decode_performance_asynchronous(self, max_time_roll=10, beh_idx=1):
        res_unrolled_dict_all = self.get_metrics_asynchronous(max_time_roll=max_time_roll,
                                                              beh_idx=beh_idx)
        self.plot_decode_performance_over_time(res_unrolled_dict_all)
        self.plot_decode_performance_at_tstar(res_unrolled_dict_all)
        return

    def run_all(self):
        self.load_data()
        self.load_ground_truth_data()
        self.get_colormap_for_decoded_variables()
        if self.mask_fn_base in self.multi_timepoint_masks:
            # self.plot_timecourse_of_decoder_output()
            self.plot_decode_performance_for_all_behvars()
            self.plot_decode_performance_per_behvar_with_partial(0) # ball_pos_x_TRUE
            self.plot_decode_performance_per_behvar_with_partial(1) # ball_pos_y_TRUE
            self.plot_decode_performance_xy([0, 1])
            self.plot_decode_performance_unrolled_over_time(compute_egocentric=False)

            self.plot_decode_performance_per_behvar_with_partial(7)  # ball_pos_ego_x_TRUE
            self.plot_decode_performance_per_behvar_with_partial(6)  # ball_pos_ego_y_TRUE
            self.plot_decode_performance_xy([7, 6])
            self.plot_decode_performance_unrolled_over_time(compute_egocentric=True)
        elif self.mask_fn_base in self.single_timepoint_masks:
            self.plot_decode_performance_asynchronous()
        return


def main(argv):
    print('decoding summary plotting')
    flags_, _ = parser.parse_known_args(argv)
    flags = vars(flags_)
    tmp = PlotHelper(**flags)
    tmp.run_all()
    return


if __name__ == "__main__":
    main(sys.argv[1:])
