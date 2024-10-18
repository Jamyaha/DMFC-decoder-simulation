from phys import phys_utils, data_utils
import sys
from scipy import stats, linalg
import argparse
import numpy as np
import pickle as pk
import pandas as pd
from phys.phys_utils import nnan_pearsonr as nnan_pearsonr

default_save_path = '/Users/hansem/Dropbox (MIT)/MPong/phys/results/old_from_rishi_om/rnn_comparison_results/'
# default_save_path = '/om/user/rishir/lib/MentalPong/phys/results/rnn_comparison_results/'

parser = argparse.ArgumentParser()
parser.add_argument('--subject_id', default='all_hand_dmfc') # perle_hand_dmfc
parser.add_argument('--condition', default='occ')
parser.add_argument('--neural_data_to_use', default='neural_responses_reliable_FactorAnalysis_50')
parser.add_argument('--timebinsize', default=50, type=int)

""" need to refactor this to use decode_utils.py for single unit encoding fits """


class RnnNeuralComparer(object):
    def __init__(self, **kwargs):
        self.subject_id = kwargs.get('subject_id', 'perle_hand_dmfc')
        self.timebinsize = kwargs.get('timebinsize', 50)
        self.condition = kwargs.get('condition', 'occ')
        self.neural_data_to_use = kwargs.get('neural_data_to_use', 'neural_responses_reliable_FactorAnalysis_50')

        self.save_path = default_save_path
        self.save_fn_suffix = '%s_%s_%dms_%s' % (self.subject_id,
                                                 self.condition,
                                                 self.timebinsize,
                                                 self.neural_data_to_use)
        self.save_fn = '%s/rnn_compare_%s.pkl' % (self.save_path, self.save_fn_suffix)

        self.masks = {}
        self.mask_full_fn = 'start_end_pad0'

        self.rnn_epoch = kwargs.get('rnn_epoch', 'output_vis-sim')
        self.rnn_state_fn = kwargs.get('rnn_state_fn', 'output_state')

        self.neural_data_to_predict = {}
        self.ball_data_as_control = {}

        self.model_fns = '/om/user/rishir/lib/PongRnn/dat/rnn_res/model_filepaths.pkl'
        self.model_fn_list = pk.load(open(self.model_fns, 'rb'))

        self.output_summary = None
        self.output_per_neuron = None

        return

    @staticmethod
    def get_noise_corrected_corr(X, Y, X1, X2, Y1, Y2):
        def sb_correction(r):
            # spearman brown correction for split halves
            return 2 * r / (1 + r)

        r_xy = nnan_pearsonr(X, Y)[0]
        r_xy_v2 = np.nanmean([nnan_pearsonr(X2, Y1)[0], nnan_pearsonr(X1, Y2)[0]])  # using halves
        r_xx = nnan_pearsonr(X1, X2)[0]
        r_yy = nnan_pearsonr(Y1, Y2)[0]

        denom = (r_xx * r_yy) ** 0.5
        denom_sb = (sb_correction(r_xx) * sb_correction(r_yy)) ** 0.5

        r_xy_n = r_xy_v2 / denom
        r_xy_n_sb = r_xy / denom_sb

        reg_metrics = {'r_xx': r_xx, 'r_yy': r_yy, 'r_xy': r_xy, 'r_xy_n_sb': r_xy_n_sb, 'r_xy_n': r_xy_n}
        return reg_metrics

    def get_noise_corrected_partial_corr(self, X, Y, X1, X2, Y1, Y2, Z):
        def residual_after_regression(x, y):
            t = np.isfinite(x) & np.isfinite(y)
            x, y = x[t], y[t]
            from sklearn.linear_model import LinearRegression
            if x.ndim == 1:
                x = np.expand_dims(x, axis=1)
            reg = LinearRegression().fit(x, y)
            y_pred = reg.predict(x)
            return y - y_pred

        X_ = residual_after_regression(X, Z)
        Y_ = residual_after_regression(Y, Z)
        X1_ = residual_after_regression(X1, Z)
        X2_ = residual_after_regression(X2, Z)
        Y1_ = residual_after_regression(Y1, Z)
        Y2_ = residual_after_regression(Y2, Z)
        return self.get_noise_corrected_corr(X_, Y_, X1_, X2_, Y1_, Y2_)

    @staticmethod
    def load_one_model_and_subsample_conditions(fn_in):
        def load_one_model(fn):
            from rnn_analysis.PongRNNSummarizer import PongRNNSummarizer
            from rnn_analysis import data_utils as behavior_data_utils
            """ load one example RNN state, with meta and masks. """
            prs = PongRNNSummarizer(filename=fn,
                                    plot_detailed_summaries=False,
                                    fig_out_path=None)
            prs.get_results_summary_base()
            res_example = prs.results_output
            res, full_meta, masks = behavior_data_utils.load_one_base(res_example, mask_early_late=False)
            rnn_data = {'res': res, 'full_meta': full_meta, 'masks': masks, 'specs': prs.specs}
            return rnn_data

        def subsample_conditions(rnn_data):
            #     res, full_meta, masks = rnn_data['res'], rnn_data['full_meta'], rnn_data['masks']
            unique_conds = phys_utils.PONG_BASIC_META_IDX
            meta = rnn_data['res']['meta_valid']
            ordered_condition_index = [np.nonzero(meta.index == uci)[0][0] for uci in unique_conds]

            res_fks = [fk for fk in rnn_data['res'].keys() if 'output' in fk] + ['state']
            for res_fk in res_fks:
                rnn_data['res'][res_fk] = rnn_data['res'][res_fk][ordered_condition_index, :, :]

            for mask_fk in rnn_data['masks'].keys():
                rnn_data['masks'][mask_fk] = rnn_data['masks'][mask_fk][ordered_condition_index, :, :]

            for meta_fk in rnn_data['full_meta'].keys():
                rnn_data['full_meta'][meta_fk] = rnn_data['full_meta'][meta_fk][ordered_condition_index, :, :]

            return rnn_data

        rnn_data_ = load_one_model(fn_in)
        return subsample_conditions(rnn_data_)

    def load_neural_data(self):
        """
        Load reference RNN model to get masks for alignment.
        Map neurons to RNN time axis using correspondence between masks.
        This mapping automatically applies the mask.
        :return: sets data_to_fit
        """

        def map_neurons_to_rnn_state_dimensions(rnn_dat, neural_dat):
            """
            Neural data is not sampled at the same rate at RNNs.
            Use the corresponding masks to stretch neural data trials.
            """
            mask_neural = neural_dat['masks'][self.condition]['start_end_pad0']
            mask_rnn = rnn_dat['masks']['output_vis-sim'][:, :, 0]

            def map_one(data_neur):
                data_neur_mapped = np.ones(mask_rnn.shape) * np.nan
                for tr_i in range(mask_rnn.shape[0]):
                    t_n = np.nonzero(np.isfinite(mask_neural[tr_i]))[0]
                    x_n = np.linspace(0, 1, len(t_n))
                    y_n = data_neur[tr_i, t_n]

                    t_r = np.nonzero(np.isfinite(mask_rnn[tr_i]))[0]
                    x_r = np.linspace(0, 1, len(t_r))

                    y_r = np.interp(x_r, x_n, y_n)
                    data_neur_mapped[tr_i, t_r] = y_r
                return data_neur_mapped

            dat_mapped = {
                'behavioral_responses': {self.condition: {}},
                '%s' % self.neural_data_to_use: {self.condition: {}},
                '%s_sh1' % self.neural_data_to_use: {self.condition: {}},
                '%s_sh2' % self.neural_data_to_use: {self.condition: {}},
            }
            for fk1 in dat_mapped.keys():
                x_neural_dat = neural_dat[fk1][self.condition]
                if isinstance(x_neural_dat, dict):
                    for fk2 in x_neural_dat.keys():
                        dat_mapped[fk1][self.condition][fk2] = map_one(x_neural_dat[fk2])
                else:
                    dat_mapped[fk1][self.condition] = []
                    for xx in x_neural_dat:
                        dat_mapped[fk1][self.condition].append(map_one(xx))
                    dat_mapped[fk1][self.condition] = np.array(dat_mapped[fk1][self.condition])
            return dat_mapped

        def set_masks_from_rnn(rnn_dat):
            # to do: fill this in.
            self.masks['start_end_pad0'] = rnn_dat['masks']['output_vis-sim'][:, :, 0]
            self.masks['start_occ_pad0'] = rnn_dat['masks']['output_vis'][:, :, 0]
            self.masks['occ_end_pad0'] = rnn_dat['masks']['output_sim'][:, :, 0]
            self.masks['f_pad0'] = rnn_dat['masks']['output_f'][:, :, 0]
            self.masks['occ_pad0'] = np.array(rnn_dat['masks']['output_sim'][:, :, 0])

            for i in range(self.masks['occ_pad0'].shape[0]):
                time_idx = np.nonzero(np.isfinite(self.masks['occ_pad0'][i]))[0][0]
                self.masks['occ_pad0'][i, time_idx + 1:] = np.nan
            return

        def set_neural_data_to_predict_raw(dat_neur_mapped):
            """ this doesn't do anything interesting. just picking out fields of a dictionary and renaming (for now)"""
            data_to_fit = {}
            for suffix in ['', '_sh1', '_sh2']:
                sfn = '%s%s' % (self.neural_data_to_use, suffix)
                x_ = dat_neur_mapped[sfn][self.condition]
                data_to_fit['X%s' % suffix] = np.array(x_)

            data_to_fit = phys_utils.get_state_pairwise_distances(data_to_fit, self.masks)
            self.neural_data_to_predict = data_to_fit
            return

        def set_control_data_for_partials(dat_neur_mapped):
            tmp = []
            tmp.append(dat_neur_mapped['behavioral_responses']['occ']['ball_pos_x_TRUE'])
            tmp.append(dat_neur_mapped['behavioral_responses']['occ']['ball_pos_y_TRUE'])
            data_to_fit = {'ball': np.array(tmp)}
            self.ball_data_as_control = phys_utils.get_state_pairwise_distances(data_to_fit, self.masks)
            return

        data_aug = data_utils.load_neural_dataset(subject_id=self.subject_id,
                                                  timebinsize=self.timebinsize,
                                                  recompute_augment=True)
        rnn_data_reference = self.load_one_model_and_subsample_conditions(self.model_fn_list[0])
        data_mapped = map_neurons_to_rnn_state_dimensions(rnn_data_reference, data_aug)
        set_masks_from_rnn(rnn_data_reference)
        set_neural_data_to_predict_raw(data_mapped)
        set_control_data_for_partials(data_mapped)

        return

    def compare_representational_geometry_rnn_to_neural_data(self, rnn_states):
        X_rnn = {'x_rnn': rnn_states}
        pdist_metrics_rnn = phys_utils.get_state_pairwise_distances(X_rnn, self.masks)

        res_sim = {}
        for mask_fk in self.masks.keys():
            for dist_type in ['euclidean']:
                X = np.array(self.neural_data_to_predict['pdist_X_%s_%s' % (mask_fk, dist_type)])
                X1 = np.array(self.neural_data_to_predict['pdist_X_sh1_%s_%s' % (mask_fk, dist_type)])
                X2 = np.array(self.neural_data_to_predict['pdist_X_sh2_%s_%s' % (mask_fk, dist_type)])
                Y = np.array(pdist_metrics_rnn['pdist_x_rnn_%s_%s' % (mask_fk, dist_type)])
                Y1, Y2 = Y, Y
                Z = np.array(self.ball_data_as_control['pdist_ball_%s_%s' % (mask_fk, dist_type)])

                res_sim_ = self.get_noise_corrected_corr(X, Y, X1, X2, Y1, Y2)
                for rfk in res_sim_.keys():
                    res_sim['pdist_similarity_%s_%s_%s' % (mask_fk, dist_type, rfk)] = res_sim_[rfk]

                res_sim_ = self.get_noise_corrected_partial_corr(X, Y, X1, X2, Y1, Y2, Z)
                for rfk in res_sim_.keys():
                    res_sim['pdist_similarity_partial_%s_%s_%s' % (mask_fk, dist_type, rfk)] = res_sim_[rfk]
        return res_sim

    def predict_neural_data_from_rnn(self):
        res_summary = []
        for fn in self.model_fn_list:
            rnn_data = self.load_one_model_and_subsample_conditions(fn)
            rnn_states_cxtxd = np.array(rnn_data['res'][self.rnn_state_fn])
            rnn_states_dxcxt = np.transpose(rnn_states_cxtxd, (2, 0, 1))  # unit x cond x time

            res_summary_curr = self.compare_representational_geometry_rnn_to_neural_data(rnn_states_dxcxt)
            res_summary_curr.update(rnn_data['specs'])
            res_summary.append(res_summary_curr)

        res_summary = pd.DataFrame(res_summary)
        self.output_summary = res_summary

        return

    def consolidate_with_behavioral_metrics(self):
        from rnn_analysis import data_utils as beh_data_utils
        df_base, df_model_base, df_primate = beh_data_utils.load__comparison_summary_df()
        for cfn in ['cons_Humanocc_residual_err_total', 'cons_Monkeyocc_residual_err_total',
                    'decode_vis-sim_to_sim_index_mae_k2', 'error_f_mae',
                    'geom_vis-sim_state_rel_speed_full', 'geom_vis-sim_state_PR',
                    'geom_vis-sim_state_raw_pos_full', 'geom_vis-sim_state_rel_acc_full']:
            self.output_summary[cfn] = df_model_base[cfn]
        return

    def save_results(self):
        data_to_save = {'summary': self.output_summary}
        with open(self.save_fn, 'wb') as f:
            f.write(pk.dumps(data_to_save))
        print('dat_to_save saved to %s' % self.save_fn)
        return

    def run_all(self):
        print('Loading data')
        self.load_neural_data()
        print('Compare data to RNNs')
        self.predict_neural_data_from_rnn()
        print('Consolidate with behavior')
        self.consolidate_with_behavioral_metrics()
        print('Save')
        self.save_results()
        return


def main(argv):
    print('RNN to DMFC comparison')
    flags_, _ = parser.parse_known_args(argv)
    flags = vars(flags_)
    pong_exp = RnnNeuralComparer(**flags)
    pong_exp.run_all()

    return


if __name__ == "__main__":
    main(sys.argv[1:])
