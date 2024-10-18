import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import rankdata
import utils
import pandas as pd

sns.set_style('ticks', {"axes.linewidth": "1", 'axes.yaxis.grid': False})

def get_good_axis_lims(values):
    mm, MM = np.nanmin(values), np.nanmax(values)
    dr = MM - mm
    ylim = [np.nanmin(values) - dr * 0.1, np.nanmax(values) + dr * 0.1]
    return ylim


def plot_least_square_line(x, y, ax):
    ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), 'k:')
    return

def make_axis_nice(ax, offset=5):
    sns.despine(ax=ax, offset=offset)
    ax.spines['left'].set_linewidth(2)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_color('black')
    return


def add_mpong_frame_to_axis(ax):
    ax.set_aspect(1.0)
    ax.plot([-10, 10, 10, -10, -10], [10, 10, -10, -10, 10], 'k-', lw=2)
    ax.plot([5, 5], [-10, 10], 'k-', lw=1)
    ax.set_xlim([-11, 11])
    ax.set_ylim([-11, 11])
    plt.axis('off')
    return


def smooth(x, win=4, sd=1):
    # typically 50ms sd, 200ms span, but here working on 50ms binned data.
    from scipy import signal
    filt = signal.gaussian(win, std=sd)
    filt = filt / np.nansum(filt)
    return np.convolve(x, filt, 'same')


def smooth_mat(x, win=4, sd=1):
    x2 = np.ones(x.shape) * np.nan
    for i in range(x2.shape[0]):
        x2[i, :] = smooth(x[i, :], win=win, sd=sd)
    return x2


def midrange(x):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))


def get_colormap_from_metaparams(metaparam_df):
    # metaparam_df = data_aug['meta']
    colorvar_choices = ['x0', 'y0', 'n_bounce_correct', 'yf',
                        'dx', 'dy', 'speed0', 'heading0',
                        'paddle_error', 'paddle_error_signed']
    all_cols = {}
    for colorvar in colorvar_choices:
        cax = midrange(rankdata(metaparam_df[colorvar]))
        all_cols[colorvar] = plt.cm.plasma(cax)
    return all_cols


def plot_colored_lines_masked(xdata, time_ax, mask, cols, ax, linestyle, realign=False):
    phys_utils = phys.phys_utils

    x_masked = phys_utils.apply_mask(xdata, mask)
    if realign:
        x_masked = phys_utils.realign_masked_data(x_masked)

    for xi in range(x_masked.shape[0]):
        col_curr = cols[xi]
        ax.plot(time_ax, x_masked[xi, :], color=col_curr, ls=linestyle)
        first_nnan = np.nonzero(np.isfinite(x_masked[xi, :]))[0][0]
        last_nnan = np.nonzero(np.isfinite(x_masked[xi, :]))[0][-1]
        ax.plot(time_ax[first_nnan], x_masked[xi, first_nnan], 'wo', mec=col_curr)
        ax.plot(time_ax[last_nnan], x_masked[xi, last_nnan], 'wo', mec=col_curr)
    return


def plot_2d_map(ax, x, y, c, nbins=50, func='mean'):
    df = pd.DataFrame({'x': x.flatten(), 'y': y.flatten(), 'c': c.flatten()})
    df['x_bins'] = pd.cut(df['x'], nbins)
    df['y_bins'] = pd.cut(df['y'], nbins)
    if func == 'mean':
        df_mu = df.groupby(['x_bins', 'y_bins']).mean()
    else:
        df_mu = df.groupby(['x_bins', 'y_bins']).mean()



    return

def display_table(df, ax):
    from pandas.plotting import table
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis
    table(ax, df, loc='center')  # where df is your data frame
    return ax
#
# def plot_2d_dist(x_p_, y_p_, ax):
#     from scipy.stats import multivariate_normal
#     tmp = np.array([x_p_, y_p_])
#     mu = np.nanmean(tmp, axis=1)
#     cov = np.cov(tmp)
#     rv = multivariate_normal(mu, cov)
#     x, y = np.mgrid[-11:11:.1, -11:11:.1]
#     pos = np.dstack((x, y))
#     ax.contourf(x, y, rv.pdf(pos), cmap=plt.cm.Greens)
#     return

#
# def plot_lines_with_color(x, cols, smooth=False, axes=None):
#     """ this isn't used now?"""
#     if smooth:
#         x = smooth_mat(x)
#     ylimits = [np.nanmin(x) * 0.95, np.nanmax(x) * 1.1]
#     if axes is None:
#         f, axes = plt.subplots(1, 1, figsize=(4, 3))
#         for tr_i in range(x.shape[0]):
#             axes.plot(x[tr_i, :], color=cols[tr_i], alpha=0.5)
#         axes.set_ylim(ylimits)
#         axes.set_yticks([])
#         axes.set_xticklabels([])
#         sns.despine(ax=axes, offset=5)
#
#     plt.tight_layout()
#     return
