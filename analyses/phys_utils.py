import csv
import pandas as pd
import numpy as np
from scipy import signal

from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.manifold import Isomap

from sklearn.metrics.pairwise import nan_euclidean_distances

from sklearn.model_selection import GroupKFold

# from rnn_analysis import utils as rnn_analysis_utils

PONG_BASIC_META_IDX = [
    29337, 55062, 58244, 59920, 68220, 72780, 77046, 77944, 90751,
    93053, 99902, 105802, 118879, 122957, 126479, 158705, 159613, 163248,
    167848, 179553, 182748, 183171, 184642, 187617, 197632, 199564, 204372,
    204678, 206128, 217817, 218791, 226533, 232533, 241919, 242380, 244437,
    245662, 248856, 251629, 258471, 269179, 270569, 273073, 273515, 287936,
    297652, 301600, 320246, 325813, 328386, 330797, 332165, 340684, 351612,
    356164, 356765, 370777, 377134, 379400, 381514, 388571, 394672, 396358,
    400995, 401673, 406460, 413189, 413513, 420020, 428615, 435203, 448412,
    454477, 456038, 460651, 463174, 464785, 465431, 469268,
]


def get_trial_meta_indices():
    return PONG_BASIC_META_IDX


def consolidate_trial_meta(trial_meta, trial_mean_groupbyvar='py_meta_index'):
    # average over py_meta_index, and order
    unique_conds = get_trial_meta_indices()
    meta = trial_meta.groupby(trial_mean_groupbyvar).mean()
    meta['dx'] = meta['ball_speed'] * np.cos(np.deg2rad(meta['ball_heading']))
    meta['dy'] = meta['ball_speed'] * np.sin(np.deg2rad(meta['ball_heading']))
    missing_idx = [pidx for pidx in unique_conds if pidx not in meta.index]
    for m in missing_idx:
        meta.loc[m] = np.nan
    meta_ordered = meta.loc[unique_conds]
    meta_ordered[trial_mean_groupbyvar] = meta_ordered.index
    return meta_ordered


def get_trial_timepoints():
    dfn = '/om/user/rishir/data/pong_basic/RF/occluded_pong_bounce1_pad8_4speed/valid_meta_sample_full.pkl'
    meta_augmented = pd.read_pickle(dfn)
    scaling_factor = 41  # 41ms for each rnn time-step.
    t_f, t_occ = [], []
    for pmidx in PONG_BASIC_META_IDX:
        i = meta_augmented['meta_index'] == pmidx
        t_f.append(np.squeeze(meta_augmented['t_f'][i]) * scaling_factor)
        t_occ.append(np.squeeze(meta_augmented['t_occ'][i]) * scaling_factor)
    return {'t_f': t_f, 't_occ': t_occ}


def read_tsv(tsv_fn):
    tsv_file = open(tsv_fn)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    tsv_data = []
    for row in read_tsv:
        tsv_data.append(row)
    tsv_file.close()
    names = tsv_data[0]
    return pd.DataFrame(tsv_data[1:], columns=names)


def realign_responses_base(X, i, j_start, j_end, align_to_start=True):
    # i,j are the rows and indices for realignment
    X_out = np.ones(X.shape) * np.nan
    for ii, jjs, jje in zip(i, j_start, j_end):
        ii_, jjs_, jje_ = int(ii), int(jjs), int(jje)
        tmp = X[ii_, jjs_:jje_]
        if align_to_start:
            idx_new = np.arange(0, tmp.shape[0])
        else:
            idx_new = np.arange(X.shape[1] - tmp.shape[0], X.shape[1])
        X_out[ii_, idx_new] = tmp
    return X_out


def realign_responses(X_dict, i, j_start, j_end, align_to_start=True):
    def run_one(XX):
        return realign_responses_base(XX, i, j_start, j_end,
                                      align_to_start=align_to_start)

    X_out = {}
    for fk in X_dict.keys():
        X_ = X_dict[fk]
        # behavioral data has dict of analog variables
        if isinstance(X_, dict):
            X_out[fk] = {}
            for fk2 in X_.keys():
                X_curr = X_[fk2]
                X_out[fk][fk2] = run_one(X_curr)
        # population: neurons are first dimension
        elif X_.ndim == 3:
            X_out[fk] = []
            for nidx in range(X_.shape[0]):
                X_curr = X_[nidx]
                X_out[fk].append(run_one(X_curr))
            X_out[fk] = np.array(X_out[fk])
        # single feature matrix
        elif X_.ndim == 2:
            X_curr = X_
            X_out[fk] = run_one(X_curr)

    return X_out


def apply_mask(X, mask):
    X2 = np.array(X)
    X2[np.isnan(mask)] = np.nan
    return X2


def get_mask(mask):
    ms = mask.shape
    mask_2 = np.reshape(mask, (ms[0] * ms[1], ms[2]))
    return np.nonzero(np.isfinite(np.mean(mask_2, axis=1)))[0]


def flatten_to_mat(x, mask):
    # conditions x time x units
    xs = x.shape
    x2 = np.reshape(x, (xs[0] * xs[1], xs[2]))
    idx = get_mask(mask)
    x3 = x2[idx, :]
    return {'X': x3, 'idx': idx, 's': xs}


def unflatten_to_3d_mat(X, res):
    x_, idx, xs = res['X'], res['idx'], res['s']
    xs = list(xs)
    if np.prod(X.shape) != np.prod(xs):
        xs[-1] = X.shape[-1]  # third dimension has changed
    x2 = np.ones((xs[0] * xs[1], xs[2])) * np.nan
    x2[idx, :] = X
    x = np.reshape(x2, xs)
    return x


def get_embedding(x_nxcxt, mask_cxt=None, n_components=None, method='PCA', random_state=0):
    x_cxtxn = np.transpose(x_nxcxt, (1, 2, 0))
    if mask_cxt is None:
        y_cxtx1 = np.array(x_cxtxn[:, :, [0]])
    else:
        y_cxtx1 = np.expand_dims(mask_cxt, axis=2)

    x_res = flatten_to_mat(x_cxtxn, y_cxtx1)
    if n_components is None:
        n_components = np.nanmin([x_res['X'].shape[0], x_res['X'].shape[1]])

    x_ctxn = np.array(x_res['X'])
    if method == 'PCA':
        transformer = PCA(n_components=n_components, svd_solver='full')
    elif method == 'Isomap':
        transformer = Isomap(n_components=n_components)
    elif method == 'FastICA':
        transformer = FastICA(n_components=n_components, random_state=random_state)
    elif method == 'FactorAnalysis':
        transformer = FactorAnalysis(n_components=n_components, random_state=random_state, rotation=None)
    elif method == 'FactorAnalysis_wrotation':
        transformer = FactorAnalysis(n_components=n_components, random_state=random_state, rotation='varimax')
    else:
        transformer = PCA(n_components=n_components, svd_solver='full')

    x_ctxnf = transformer.fit_transform(x_ctxn)
    x_cxtxnf = unflatten_to_3d_mat(x_ctxnf, x_res)
    x_nfxcxt = np.transpose(x_cxtxnf, (2, 0, 1))

    return x_nfxcxt, transformer


def realign_masked_data(X, align_to_start=True):
    """
        align to first non-nan entry. this is dangerous if the data matrix is supposed to have nans
        (e.g. ball not displayed yet)
        """

    X_out = np.ones(X.shape) * np.nan
    for i in range(X.shape[0]):
        nn_idx = np.nonzero(np.isfinite(X[i, :]))[0]
        if len(nn_idx) == 0:
            continue
        start_win = nn_idx[0]
        end_win = nn_idx[-1] + 1

        tmp = X[i, start_win:end_win]
        if align_to_start:
            idx_new = np.arange(0, tmp.shape[0])
        else:
            idx_new = np.arange(X.shape[1] - tmp.shape[0], X.shape[1])
        X_out[i, idx_new] = tmp
    return X_out


def smooth_gaussian(x, win=250, sd=50):
    filt = signal.gaussian(win, std=sd)
    filt = filt / np.nansum(filt)
    return np.convolve(x, filt, 'same')


def midrange(x):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))


def impute_nan(x_in, cutoff_ncond=5, return_kept_idx=True):
    """ x is nneur x cond x time.
    For each neuron, set FR of missing conditions to mean over time and conditions.
    Note that all time-points are filled with mean, so don't use this as a nan-mask.
    """
    nnancond_per_neuron = np.nansum(np.isnan(np.nanmean(x_in, axis=2)), axis=1)
    i_cutoff = np.nonzero(nnancond_per_neuron <= cutoff_ncond)[0]
    X_m = x_in[i_cutoff, :, :]

    if X_m.shape[0] == 0:
        X_m_nn = np.ones((0, x_in.shape[1], x_in.shape[2]))
    else:
        i, j = np.nonzero(~np.isfinite(np.nanmean(X_m, axis=2)))
        X_m_nn = np.array(X_m)
        for ii, jj in zip(i, j):
            x = np.nanmean(X_m[ii, :, :])
            # time_ax = np.isfinite(mask[jj,:])
            if np.isnan(x):
                X_m_nn[ii, jj, :] = 0
                # X_m_nn[ii, jj, time_ax] = 0
            else:
                X_m_nn[ii, jj, :] = x
                # X_m_nn[ii, jj, time_ax] = x
    if return_kept_idx:
        return X_m_nn, i_cutoff
    else:
        return X_m_nn


def flatten_pca(x_, n_comp=3):
    """ x is nneur x cond x time."""
    s = x_.shape
    xs = np.reshape(x_, (s[0], s[1] * s[2])).T
    ys = np.ones((s[1] * s[2], n_comp)) * np.nan
    idx = np.isfinite(np.nanmean(xs, axis=1))
    xxs = xs[idx]
    pca = PCA(n_components=n_comp)
    yys = pca.fit_transform(xxs)
    ys[idx] = yys
    y_recon = np.reshape(ys.T, (n_comp, s[1], s[2]))
    return y_recon, pca


def apply_flattened_pca(x_, pca_trans, n_comp=3):
    s = x_.shape
    xs = np.reshape(x_, (s[0], s[1] * s[2])).T
    ys = np.ones((s[1] * s[2], n_comp)) * np.nan
    idx = np.isfinite(np.nanmean(xs, axis=1))
    xxs = xs[idx]
    yys = pca_trans.transform(xxs)
    ys[idx] = yys
    y_recon = np.reshape(ys.T, (n_comp, s[1], s[2]))
    return y_recon


def nnan_pearsonr(x, y):
    ind = np.isfinite(x) & np.isfinite(y)
    if np.nansum(ind) < 2:
        return np.nan, np.nan
    x, y = x[ind], y[ind]
    return pearsonr(x, y)


def nnan_spearmanr(x, y):
    ind = np.isfinite(x) & np.isfinite(y)
    if np.nansum(ind) < 2:
        return np.nan, np.nan
    x, y = x[ind], y[ind]
    return spearmanr(x, y)


def linear_regress_grouped(X, y, group=None, k_folds=2, use_pls=True, max_n_pls_components=25):
    """
    Note: fitting only the intercept can lead to spurious correlations in small datasets.
    since the results over all folds are pooled before computing a correlation (so the mean of each fold is fit).
    """

    if group is None:
        group = np.arange(X.shape[0])
    all_splits = GroupKFold(n_splits=k_folds)

    y_pred = np.ones(y.shape) * np.nan
    coeff = []

    n_pls_components = np.min([max_n_pls_components, X.shape[0], X.shape[1], y.shape[1]])

    for train_index, test_index in all_splits.split(X, y, group):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if use_pls:
            pls2 = PLSRegression(n_components=n_pls_components)
            pls2.fit(X_train, y_train)
            y_pred[test_index] = pls2.predict(X_test)
        else:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
            reg = LinearRegression()
            reg.fit(X_train_s, y_train)
            coeff.append(reg.coef_)
            y_pred[test_index] = reg.predict(np.array(X_test_s))

    r, p = nnan_pearsonr(y, y_pred)
    rs, ps = nnan_spearmanr(y, y_pred)
    mae = np.nanmean(np.abs(y - y_pred))

    res_summary = {
        'r': r,
        'p': p,
        'rs': rs,
        'ps': ps,
        'mae': mae,
        'y_pred': y_pred,
        'y_true': y,
        'coeff': np.array(coeff),
    }
    return res_summary


def linear_reg_test(x_, y_, groupby='condition', prune_features=True,
                    k_folds=2, use_pls=True, max_n_pls_components=25):
    """
    :param x_: conditions x time x units
    :param y_: conditions x time x units
    :param groupby: crossval over conditions, or over conditions x time
    :param k_folds: 2
    :return: res
    """
    if groupby == 'condition':
        g = np.tile(range(y_.shape[0]), (y_.shape[1], 1)).T
        g = np.expand_dims(g, axis=2)
        res_g = rnn_analysis_utils.flatten_to_mat(g, y_)
        data_g = np.array(res_g['X'])
    else:
        data_g = None

    res_x = rnn_analysis_utils.flatten_to_mat(x_, y_)
    res_y = rnn_analysis_utils.flatten_to_mat(y_, y_)

    data_x = np.array(res_x['X'])
    data_y = np.array(res_y['X'])

    if prune_features:
        var_thres = 10 ** (-10)
        t_feat = np.nanvar(data_x, axis=0) > var_thres
        data_x = data_x[:, t_feat]

    res_reg = linear_regress_grouped(data_x, data_y,
                                     group=data_g,
                                     k_folds=k_folds,
                                     use_pls=use_pls,
                                     max_n_pls_components=max_n_pls_components)
    # res_reg['data'] = {'X': res_x, 'Y': res_y, 'groupby': groupby}
    return res_reg


def get_state_pairwise_distances(states_dict, masks):
    def pairwise_distance_base(x, dist_type='euclidean'):
        if dist_type == 'euclidean':
            return nan_euclidean_distances(x)
        elif dist_type == 'corr':
            return np.corrcoef(x)

    res_dist_states = {}
    for x_fk in states_dict.keys():
        X = np.array(states_dict[x_fk])
        for mask_fk in masks.keys():
            mask = np.array(masks[mask_fk])
            t = np.isfinite(mask)
            for dist_type in ['euclidean', 'corr']:
                pd_x = pairwise_distance_base(X[:, t].T, dist_type=dist_type)
                pd_x[np.triu_indices(pd_x.shape[0])] = np.nan

                res_dist_states['pdist_%s_%s_%s' % (x_fk, mask_fk, dist_type)] = pd_x.flatten()
    return res_dist_states
