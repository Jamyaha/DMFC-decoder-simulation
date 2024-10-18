from utils.phys_utils import nnan_pearsonr as nnan_pearsonr

import sys, os
import argparse
import numpy as np
import pandas as pd
import pickle as pk
import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils import utils as analysis_utils

sns.set_style('ticks', {"axes.linewidth": "1", 'axes.yaxis.grid': False})
from utils import generic_plot_utils as rr_gpu
from utils import plot_specs

parser = argparse.ArgumentParser()
parser.add_argument('--subject_id', default='all_hand_dmfc') # perle mahler
parser.add_argument('--condition', default='occ')
parser.add_argument('--neural_data_to_use', default='neural_responses_reliable_FactorAnalysis_50')
parser.add_argument('--timebinsize', default=50, type=int)

partial_corr = analysis_utils.partial_corr # TBD


class PlotHelper(object):
    def __init__(self, **kwargs):
        self.subject_id = kwargs.get('subject_id', 'all_hand_dmfc')
        self.timebinsize = kwargs.get('timebinsize', 50)
        self.condition = kwargs.get('condition', 'occ')
        self.neural_data_to_use = kwargs.get('neural_data_to_use', 'neural_responses_reliable_FactorAnalysis_50')

        figoutpath_base = plot_specs.figoutpath_base # redo_paper_scratch_202202
        self.figoutpath = '%s/%s%d/%s/rnn_compare_%s/' % (
            figoutpath_base, self.subject_id, self.timebinsize, self.neural_data_to_use,
            self.condition)
        if os.path.isdir(self.figoutpath) is False:
            os.makedirs(self.figoutpath)

        return

    def load_data(self):
        default_save_path = '../data/'
        # default_save_path = '/Users/hansem/Dropbox (MIT)/MPong/phys/results/old_from_rishi_om/rnn_comparison_results/'
        # default_save_path = '/om/user/rishir/lib/MentalPong/phys/results/rnn_comparison_results/'
        save_fn_suffix = '%s_%s_%dms_%s' % (self.subject_id,
                                            self.condition,
                                            self.timebinsize,
                                            self.neural_data_to_use)
        save_fn = '%s/rnn_compare_%s.pkl' % (default_save_path, save_fn_suffix)
        tmp = pk.load(open(save_fn, 'rb'))
        self.data_df = tmp['summary']
        return

    def set_prefs(self):
        self.disttype = 'euclidean'
        self.cons_metrics = [
            'pdist_similarity_start_end_pad0_%s_r_xy_n_sb' % self.disttype,
            'pdist_similarity_occ_end_pad0_%s_r_xy_n_sb' % self.disttype,
        ]

        self.ylims_for_cons = rr_gpu.get_good_axis_lims(np.array(self.data_df[self.cons_metrics]).flatten())
        # self.partial_cons_metrics = [
        #     'pdist_similarity_partial_start_end_pad0_%s_r_xy_n_sb' % self.disttype,
        #     'pdist_similarity_partial_occ_end_pad0_%s_r_xy_n_sb' % self.disttype,
        # ]
        self.predictor_metrics = [
            'decode_vis-sim_to_sim_index_mae_k2',
            'error_f_mae',
            'geom_vis-sim_state_PR',
            'geom_vis-sim_state_rel_speed_full',
            'geom_vis-sim_state_rel_acc_full'
        ]
        return

    @staticmethod
    def calculate_corr_all(df):
        df = df.dropna()._get_numeric_data()
        dfcols = pd.DataFrame(columns=df.columns)
        rho_values = dfcols.transpose().join(dfcols, how='outer')
        p_values = dfcols.transpose().join(dfcols, how='outer')
        for r in df.columns:
            for c in df.columns:
                r_, p_ = nnan_pearsonr(df[r], df[c])
                rho_values[r][c] = r_
                p_values[r][c] = p_
        return rho_values, p_values

    def plot_comparison_histograms(self):
        def plot_hist_base(metrics):
            bins = np.arange(-1, 1, 0.01)
            nm = len(metrics)
            f_, axes = plt.subplots(nm, 1, figsize=(4, 4 * nm))
            for fki, fk in enumerate(metrics):
                axes[fki].hist(self.data_df[fk], bins, orientation="horizontal")
                axes[fki].set_title(fk)
                rr_gpu.make_axis_nice(axes[fki])
                axes[fki].set_ylim(self.ylims_for_cons)
            plt.tight_layout()
            return f_

        f = plot_hist_base(self.cons_metrics)
        f.savefig('%s/hist_cons_metrics.pdf' % self.figoutpath)

        # f = plot_hist_base(self.partial_cons_metrics)
        # f.savefig('%s/hist_partial_cons_metrics.pdf' % self.figoutpath)

        return

    def plot_consistency_over_groups(self):
        f, axes = plt.subplots(1, 2, figsize=(6, 4))

        df_all = self.data_df
        current_palette = sns.color_palette()
        alpha = 0.2
        palette_reordered = [current_palette[0], current_palette[2],
                             current_palette[3], current_palette[1]]
        # palette_reordered = [current_palette[1], current_palette[3],
        #                      current_palette[2], current_palette[0]] # 1320

        for i, cons_metricfn in enumerate(self.cons_metrics):
            ax = axes[i]
            sns.boxplot(x="loss_weight_type", y=cons_metricfn,
                        data=df_all, ax=ax, whis=100, palette=palette_reordered,
                        boxprops=dict(alpha=alpha), capprops=dict(alpha=alpha),
                        whiskerprops=dict(alpha=alpha), medianprops=dict(alpha=alpha),
                        order=['mov', 'vis-mov', 'vis-sim-mov', 'sim-mov' ])

            sns.swarmplot(x="loss_weight_type", y=cons_metricfn,
                          data=df_all, ax=ax, palette=palette_reordered,
                          order=['mov', 'vis-mov', 'vis-sim-mov', 'sim-mov' ])
            rr_gpu.make_axis_nice(ax)
        plt.tight_layout()
        f.savefig('%s/hist_cons_metrics_per_group.pdf' % self.figoutpath)

    def plot_variance_exp_of_consistency(self, cons_metrics, predictor_metrics,
                                         square_correlation=True, plot_xlabels=False):
        x_size = np.max([np.min([len(predictor_metrics) * 0.33, 12]), 4])
        y_size = 5
        if plot_xlabels:
            y_size = 8
        nsubplots_y = np.max([2, len(cons_metrics)])
        f, axes = plt.subplots(nsubplots_y, 2, figsize=(x_size * 2, y_size))
        cols = ['#ff00aa', '#00aaff']
        df_ = self.data_df

        for ci, cons_metric in enumerate(cons_metrics):
            ax = axes[ci, 1]
            fks = [cons_metric] + predictor_metrics

            df_for_corr = df_[fks]
            rho_raw, p_ = self.calculate_corr_all(df_for_corr)
            if square_correlation:
                r2_raw = rho_raw[cons_metric] ** 2
            else:
                r2_raw = rho_raw[cons_metric]
            p_r2_raw = p_[cons_metric]

            tmp = np.array(df_for_corr)
            rho_partial, p_partial = partial_corr(tmp, add_offset=True, return_pval=True)
            rho_partial, p_partial = rho_partial[0], p_partial[0]
            if square_correlation:
                r2_partial = rho_partial ** 2
            else:
                r2_partial = rho_partial

            plot_vals = [i for i in range(len(fks)) if i != 0]
            ax.bar(plot_vals, r2_raw[plot_vals], width=0.3, color=cols[0])
            ax.bar([i + .3 for i in plot_vals], r2_partial[plot_vals], width=0.3, color=cols[1])
            rr_gpu.make_axis_nice(ax)
            ax.set_title(cons_metric)

            if (ci == len(cons_metrics) - 1) and plot_xlabels:
                ax.set_xticks([i + .15 for i in plot_vals])
                ax.set_xticklabels(predictor_metrics, rotation=90)

            summary_stats = {'r2': np.array(r2_raw[plot_vals]),
                             'p_r2': np.array(p_r2_raw[plot_vals]),
                             'r2_partial': np.array(r2_partial[plot_vals]),
                             'p_r2_partial': np.array(p_partial[plot_vals]),
                             }

            ax = axes[ci, 0]
            rr_gpu.display_table(pd.DataFrame(summary_stats), ax)

        plt.tight_layout()
        return f, axes

    def plot_scatter_vs_consistency_base(self, ax, cons_metricn, predictor_metricn, flip_x_axis=False):
        df_ = self.data_df
        # f, axes = plt.subplots(1, 1, figsize=(4, 5))
        # ax = axes
        sns.scatterplot(data=df_, x=predictor_metricn, y=cons_metricn,
                        hue="loss_weight_type", style="rnn_type",
                        ax=ax, legend=False)
        rr_gpu.plot_least_square_line(df_[predictor_metricn],
                                      df_[cons_metricn], ax)
        ax.set_ylim(self.ylims_for_cons)
        maxval = df_[predictor_metricn].max() * 1.1
        minval = df_[predictor_metricn].min() * 0.9
        if flip_x_axis:
            ax.set_xlim([maxval, minval])
        else:
            ax.set_xlim([minval, maxval])
        rr_gpu.make_axis_nice(ax)

        return

    def plot_scatter_vs_consistency(self, cons_metricn_list, predictor_metricn_list, file_suffix=''):
        n1, n2 = len(cons_metricn_list), len(predictor_metricn_list)
        f, axes = plt.subplots(n1, n2, figsize=(4 * n2, 5 * n1))
        for i1, cons_metricn in enumerate(cons_metricn_list):
            for i2, predictor_metricn in enumerate(predictor_metricn_list):
                ax = axes[i1, i2]
                flip_xaxis = 'mae' in predictor_metricn
                self.plot_scatter_vs_consistency_base(ax, cons_metricn, predictor_metricn, flip_x_axis=flip_xaxis)
        plt.tight_layout()
        f.savefig('%s/scatter_rnn_to_dmfc_%s.pdf' % (self.figoutpath, file_suffix))
        return

    def plot_bar_rsquared(self):

        predictor_combinations = [
            self.predictor_metrics[:2],
            self.predictor_metrics[2:],
            self.predictor_metrics,
        ]
        for pci, pc in enumerate(predictor_combinations):
            fig_prefix = 'variance_exp_of_var_exp_dmfc_rnn'
            f, axes = self.plot_variance_exp_of_consistency(self.cons_metrics, pc, square_correlation=True,
                                                            plot_xlabels=True)
            f.savefig('%s/%s_%d.pdf' % (self.figoutpath, fig_prefix, pci))

        # for pci, pc in enumerate(predictor_combinations):
        #     fig_prefix = 'variance_exp_of_partial_var_exp_dmfc_rnn'
        #     f, axes = self.plot_variance_exp_of_consistency(self.partial_cons_metrics, pc, square_correlation=True,
        #                                                     plot_xlabels=True)
        #     f.savefig('%s/%s_%d.pdf' % (self.figoutpath, fig_prefix, pci))

        return

    def plot_scatter_rsquared(self):
        self.plot_scatter_vs_consistency(self.cons_metrics, self.predictor_metrics, file_suffix='raw')
        # self.plot_scatter_vs_consistency(self.partial_cons_metrics, self.predictor_metrics, file_suffix='partial')
        return

    def run_all(self):
        self.load_data()
        self.set_prefs()
        self.plot_comparison_histograms()
        self.plot_consistency_over_groups()
        self.plot_scatter_rsquared()
        self.plot_bar_rsquared()
        return


def main(argv):
    print('rnn comparison plotting')
    flags_, _ = parser.parse_known_args(argv)
    flags = vars(flags_)
    tmp = PlotHelper(**flags)
    tmp.run_all()
    return


if __name__ == "__main__":
    main(sys.argv[1:])
