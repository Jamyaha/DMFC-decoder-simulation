from utils.plot_scripts import generic_plot_utils as rr_gpu
from utils.plot_scripts import plot_specs
from utils import data_utils, phys_utils
import sys, os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.cm as mpl_cm
import seaborn as sns

sns.set_style('ticks', {"axes.linewidth": "1", 'axes.yaxis.grid': False})

parser = argparse.ArgumentParser()
parser.add_argument('--timebinsize', type=int)
parser.add_argument('--subject_id')
parser.add_argument('--condition')


class PlotHelper(object):
    def __init__(self, **kwargs): # ,timebinsize=50,subject_id='perle_hand_dmfc',condition='occ'
        # kwargs generated for shell execution?
        self.timebinsize = kwargs.get('timebinsize', 50)
        self.subject_id = kwargs.get('subject_id', 'perle_hand_dmfc')
        self.condition = kwargs.get('condition', 'occ')

        figoutpath_base = plot_specs.figoutpath_base
        self.figoutpath = '%s/%s%d/example_%s/' % (
            figoutpath_base, self.subject_id, self.timebinsize, self.condition,)

        if os.path.isdir(self.figoutpath) is False:
            os.makedirs(self.figoutpath)
        return

    def load_data(self):
        self.data_aug = data_utils.load_neural_dataset(subject_id=self.subject_id, timebinsize=self.timebinsize)
        self.all_cols = rr_gpu.get_colormap_from_metaparams(self.data_aug['meta'])
        self.cols = self.all_cols['yf']

        return

    def plot_response_over_2d_var_heatmap(self, plot_c, frame_x, frame_y,
                                          mask_fn='start_end_pad0', condition='occ',
                                          f=None, axes=None):
        plot_x = self.data_aug['behavioral_responses'][condition][frame_x]
        plot_y = self.data_aug['behavioral_responses'][condition][frame_y]
        mask = self.data_aug['masks'][condition][mask_fn]

        plot_x = phys_utils.apply_mask(plot_x, mask)
        plot_y = phys_utils.apply_mask(plot_y, mask)
        plot_c = phys_utils.apply_mask(plot_c, mask)

        cmap_limits = np.percentile(plot_c[np.isfinite(plot_c)], [1, 99])

        df_data = {
            'x': plot_x[np.isfinite(plot_x)],
            'y': plot_y[np.isfinite(plot_y)],
            'c': plot_c[np.isfinite(plot_c)],
        }
        df = pd.DataFrame(df_data)
        bins = np.arange(-11, 11, 0.5)
        df['x_bins'] = pd.cut(df['x'], bins)
        df['y_bins'] = pd.cut(df['y'], bins)

        df_mu = df.groupby(['y_bins', 'x_bins']).mean()
        c = np.array(df_mu['c'].unstack())

        if axes is None:
            f, axes = plt.subplots(1, 1, figsize=(6, 6))

        axes.imshow(c, origin='lower', extent=[bins[0], bins[-1], bins[0], bins[-1]],
                    cmap=plt.cm.viridis, vmin=cmap_limits[0], vmax=cmap_limits[1]) # plt.cm.viridis mpl_cm.viridis
        rr_gpu.add_mpong_frame_to_axis(axes)
        axes.set_xlabel(frame_x)
        axes.set_ylabel(frame_y)
        axes.set_aspect(1.0)
        rr_gpu.add_mpong_frame_to_axis(axes)
        plt.tight_layout()

        return f, axes

    def plot_response_trace_over_time(self, ax, y, mask_fn, linestyle,
                                      realign_to_start=True, ylim_mask_fn='start_end_pad0'):
        time_ax = [ti * self.timebinsize for ti in range(y.shape[1])]
        mask = self.data_aug['masks'][self.condition][mask_fn]
        rr_gpu.plot_colored_lines_masked(y, time_ax, mask, self.cols, ax, linestyle, realign=realign_to_start)

        # axis limits
        mask_full = self.data_aug['masks'][self.condition][ylim_mask_fn]
        y_masked = phys_utils.apply_mask(y, mask_full)
        ylim = rr_gpu.get_good_axis_lims(y_masked)
        ax.set_ylim(ylim)
        return

    def plot_traces_over_epochs(self, xdata_, savefn=None):
        f, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for mfni, mask_fn in enumerate(['start_end_pad0', 'start_occ_pad0', 'occ_end_pad0']):
            ax = axes[mfni]
            if mask_fn == 'start_end_pad0':
                self.plot_response_trace_over_time(ax, xdata_, 'start_occ_pad0', ':', realign_to_start=False)
                self.plot_response_trace_over_time(ax, xdata_, 'occ_end_pad0', '-', realign_to_start=False)
            else:
                self.plot_response_trace_over_time(ax, xdata_, mask_fn, '-', realign_to_start=True)
            rr_gpu.make_axis_nice(ax)
            plt.tight_layout()

        if savefn is not None:
            f.suptitle(savefn)
            f.savefig('%s/traces_%s.pdf' % (self.figoutpath, savefn))
        return

    def plot_behavioral_traces(self):
        for xfn in ['paddle_pos_y', 'joy', 'eye_v', 'eye_h', 'eye_dv', 'eye_dh', 'eye_dtheta', 'eye_dspeed']:
            xdata_ = self.data_aug['behavioral_responses'][self.condition][xfn]
            self.plot_traces_over_epochs(xdata_, savefn=xfn)
        return

    def plot_neural_components_trace_and_2dmap(self, method='FactorAnalysis', ncomp=50):

        factor_fn = 'neural_responses_reliable_%s_%d' % (method, ncomp)
        FR_embed = self.data_aug[factor_fn][self.condition]

        ncomps_to_plot = 10
        f, axes = plt.subplots(2, ncomps_to_plot, figsize=(3 * ncomps_to_plot, 5))
        for i in range(ncomps_to_plot):
            ax = axes[0, i]
            data_fr = FR_embed[i]
            self.plot_response_trace_over_time(ax, data_fr, 'start_occ_pad0', ':', realign_to_start=False)
            self.plot_response_trace_over_time(ax, data_fr, 'occ_end_pad0', '-', realign_to_start=False)
            rr_gpu.make_axis_nice(ax)

            ax = axes[1, i]
            self.plot_response_over_2d_var_heatmap(FR_embed[i], 'ball_pos_x_TRUE', 'ball_pos_y_TRUE',
                                                   mask_fn='start_end_pad0', condition=self.condition, f=f, axes=ax)
            ax.set_axis_off()
        f.savefig('%s/factors_combined_%s_%d.pdf' % (self.figoutpath, method, ncomp))
        return

    def plot_neural_components_traces(self, method='FactorAnalysis', ncomp=50):
        ncomps_to_plot = 10
        factor_fn = 'neural_responses_reliable_%s_%d' % (method, ncomp)
        FR_embed = self.data_aug[factor_fn][self.condition]

        for i in range(ncomps_to_plot):
            xdata_ = FR_embed[i]
            xfn = '%s_%d' % (factor_fn, i)
            self.plot_traces_over_epochs(xdata_, savefn=xfn)
        return

    def plot_neural_traces_all_embeddings(self):
        for method in ['FactorAnalysis']:
            for ncomp in [10, 20, 50]:
                self.plot_neural_components_traces(method=method, ncomp=ncomp)
                self.plot_neural_components_trace_and_2dmap(method=method, ncomp=ncomp)
        return

    def run_all(self):
        self.load_data()
        self.plot_behavioral_traces()
        # self.plot_neural_traces_all_embeddings()
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
