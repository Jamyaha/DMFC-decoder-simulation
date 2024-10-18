import numpy as np
import phys.phys_utils as phys_utils
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class Decoder_3Ddata(object):
    def __init__(self, **kwargs):

        self.train_size = kwargs.get('train_size', 0.5)
        self.nfeatures_sample = kwargs.get('nfeatures_sample', None)  # sample within iter
        self.niter = kwargs.get('niter', 2)
        self.random_state = kwargs.get('random_state', 0)
        self.groupby = kwargs.get('groupby', 'condition')
        self.matched_timepoints = kwargs.get('matched_timepoints', True)
        self.align_to_start = kwargs.get('align_to_start', True)
        self.pool_time_conditions = kwargs.get('pool_time_conditions', True)  # otherwise pool time with neurons

        self.preprocess_func = kwargs.get('preprocess_func', 'none')
        self.ncomp = kwargs.get('ncomp', 50)
        self.prune_features = kwargs.get('prune_features', False)
        self.min_ncomp = 5
        return

    def mask_3ddata(self, x_dxcxt, mask_cxt):
        # this converts to 3d mat, after applying mask,
        # No longer need to align to start, since no rolls/clips will be done here.
        # to roll or clip, use different masks.
        def mask_align(x, m, matched_timepoints=True):
            if matched_timepoints:
                # already aligned
                return phys_utils.apply_mask(x, m)
            else:
                return phys_utils.realign_masked_data(phys_utils.apply_mask(x, m))

        return np.array([mask_align(xi, mask_cxt, self.matched_timepoints) for xi in x_dxcxt])

    @staticmethod
    def convert_to_mat_pool_conditions_and_time_dims(x_nxcxt, y_bxcxt, cond_cxtx1):
        """ turn NxCxT into CT x N. remove CT dims that are NaN.
        This is done for mapping with matched time (each timepoint is another point in state space).
        flatten_to_mat turns the first two dimensions into 1 cxtxn -> cxt x n
        """
        x_cxtxn = np.transpose(x_nxcxt, (1, 2, 0))
        y_cxtxb = np.transpose(y_bxcxt, (1, 2, 0))

        # omit times and conditions where any feature or any label is NaN
        mask_cxt = np.mean(x_nxcxt, axis=0) * np.mean(y_bxcxt, axis=0)
        mask_cxtx1 = np.expand_dims(mask_cxt, axis=2)

        res_x = phys_utils.flatten_to_mat(x_cxtxn, mask_cxtx1)
        res_y = phys_utils.flatten_to_mat(y_cxtxb, mask_cxtx1)
        res_g = phys_utils.flatten_to_mat(cond_cxtx1, mask_cxtx1)

        data = {
            'x': np.array(res_x['X']),
            'y': np.array(res_y['X']),
            'g': np.array(res_g['X']),
            'unflatten_res': res_g,
        }

        return data

    @staticmethod
    def convert_to_mat_pool_neurons_and_time_dims(x_nxcxt, y_bxcxt, cond_cxtx1):
        """ turn NxCxT into C x NT. remove NT dims that are NaN
        This is done for mapping with unmatched time (each timepoint is another dimension in feature space).
        flatten_to_mat turns the first two dimensions into 1 nxtxc -> nxt x c
        """

        def pool_neurons_and_time(data_nxcxt):
            data_nxtxc = np.transpose(data_nxcxt, (0, 2, 1))
            xs = data_nxtxc.shape
            data_ntxc = np.reshape(data_nxtxc, (xs[0] * xs[1], xs[2]))
            data_cxnt = data_ntxc.T
            t_keep = np.isfinite(np.mean(data_cxnt, axis=0))
            data_cxnt = data_cxnt[:, t_keep]
            return data_cxnt

        x_cxnt = pool_neurons_and_time(x_nxcxt)
        y_cxbt = pool_neurons_and_time(y_bxcxt)
        cond_cx1 = np.nanmean(cond_cxtx1, axis=1)

        data = {'x': x_cxnt, 'y': y_cxbt, 'g': cond_cx1, 'unflatten_res': None}
        return data

    @staticmethod
    def get_condition_indicator(x_nxcxt):
        n, c, t = x_nxcxt.shape
        g = np.tile(range(c), (t, 1)).T
        cond_cxtx1 = np.expand_dims(g, axis=2)
        return cond_cxtx1

    @staticmethod
    def fill_regression_metrics(y, y_pred, res_metrics=None, suffix=''):
        ny = y.shape[1]
        r_, rs_, mae_ = [], [], []
        for i in range(ny):
            r, p = phys_utils.nnan_pearsonr(y[:, i], y_pred[:, i])
            rs, ps = phys_utils.nnan_spearmanr(y[:, i], y_pred[:, i])
            mae = np.nanmean(np.abs(y[:, i] - y_pred[:, i]))
            r_.append(r)
            rs_.append(rs)
            mae_.append(mae)

        if res_metrics is None:
            res_metrics = {}
        res_metrics['r%s' % suffix] = r_
        res_metrics['rs%s' % suffix] = rs_
        res_metrics['mae%s' % suffix] = mae_
        res_metrics['y_pred%s' % suffix] = y_pred
        res_metrics['y_true%s' % suffix] = y

        return res_metrics

    def preprocess(self, X_train, X_test, X2=None):
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA, FactorAnalysis


        if self.prune_features:
            var_thres = 10 ** (-10)
            t_feat = np.nanvar(X_train, axis=0) > var_thres
            X_train = X_train[:, t_feat]
            X_test = X_test[:, t_feat]

        if self.preprocess_func == 'scale':
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
            X2_s = scaler.transform(X2) if (X2 is not None) else None

        elif self.preprocess_func == 'pca':
            ncomp = np.min([X_train.shape[1], self.ncomp])
            if ncomp < self.min_ncomp:
                X_train_s, X_test_s, X2_s = X_train, X_test, X2
            else:
                pca = PCA(n_components=ncomp, svd_solver='arpack')
                pca.fit(X_train)
                X_train_s, X_test_s = pca.transform(X_train), pca.transform(X_test)
                X2_s = pca.transform(X2) if (X2 is not None) else None
        elif self.preprocess_func == 'fa':
            ncomp = np.min([X_train.shape[1], self.ncomp])
            if ncomp < self.min_ncomp:
                X_train_s, X_test_s, X2_s = X_train, X_test, X2
            else:
                fa = FactorAnalysis(n_components=ncomp, random_state=0, rotation=None)
                fa.fit(X_train)
                X_train_s, X_test_s = fa.transform(X_train), fa.transform(X_test)
                X2_s = fa.transform(X2) if (X2 is not None) else None
        else:
            X_train_s, X_test_s, X2_s = X_train, X_test, X2

        return X_train_s, X_test_s, X2_s

    def regress_base(self, X, y, train_index, test_index, X2=None):
        if self.nfeatures_sample is not None:
            f_idx = np.random.choice(X.shape[1], self.nfeatures_sample, replace=False)
            X = X[:, f_idx]

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X2_test = X2[test_index] if (X2 is not None) else None

        X_train_s, X_test_s, X2_test_s = self.preprocess(X_train, X_test, X2=X2_test)

        y_pred = np.ones(y.shape) * np.nan
        y2_pred = np.ones(y.shape) * np.nan

        reg = LinearRegression()
        reg.fit(X_train_s, y_train)
        y_pred[test_index] = reg.predict(np.array(X_test_s))

        if X2 is not None:
            y2_pred[test_index] = reg.predict(np.array(X2_test_s))

        mets = {}
        mets = self.fill_regression_metrics(y, y_pred, res_metrics=mets, suffix='')
        if X2 is not None:
            mets = self.fill_regression_metrics(y, y2_pred, res_metrics=mets, suffix='_2')
        return mets

    def linear_regress_grouped_shuffle_split(self, X, y, group=None, X2=None):
        gss = GroupShuffleSplit(n_splits=self.niter, train_size=self.train_size, random_state=self.random_state)
        metrics_all = []
        for train_idx, test_idx in gss.split(X, y, group):
            metrics_all.append(self.regress_base(X, y, train_idx, test_idx, X2=X2))
        return metrics_all

    def decode_base(self, data):
        g = data['g'] if self.groupby == 'condition' else None
        res_dec = self.linear_regress_grouped_shuffle_split(data['x'], data['y'], X2=data['x2'], group=g)
        fk_all = res_dec[0].keys()
        res_summary = {
            'unflatten_res': data['unflatten_res'],
        }
        for fk in fk_all:
            x_curr = [r[fk] for r in res_dec]
            res_summary['%s_mu' % fk] = np.nanmean(x_curr, axis=0)
            res_summary['%s_sd' % fk] = np.nanstd(x_curr, axis=0)
            res_summary['%s_dist' % fk] = np.array(x_curr)
        return res_summary

    def get_data(self, x_nxcxt, y_bxcxt, x2_nxcxt=None):
        cond_cxtx1 = self.get_condition_indicator(x_nxcxt)

        if self.pool_time_conditions:
            convert_func = self.convert_to_mat_pool_conditions_and_time_dims
        else:
            convert_func = self.convert_to_mat_pool_neurons_and_time_dims

        data = convert_func(x_nxcxt, y_bxcxt, cond_cxtx1)
        if x2_nxcxt is not None:
            data2 = convert_func(x2_nxcxt, y_bxcxt, cond_cxtx1)
            data['x2'] = data2['x']
        else:
            data['x2'] = None
        return data

    def decode_one(self, data_neur_nxcxt, data_beh_bxcxt,
                   mask_neur_cxt, mask_beh_cxt, data_neur_2_nxcxt=None):

        x_nxcxtm = self.mask_3ddata(data_neur_nxcxt, mask_neur_cxt)
        y_bxcxtm = self.mask_3ddata(data_beh_bxcxt, mask_beh_cxt)

        data = self.get_data(x_nxcxtm, y_bxcxtm, x2_nxcxt=data_neur_2_nxcxt)
        return self.decode_base(data)
