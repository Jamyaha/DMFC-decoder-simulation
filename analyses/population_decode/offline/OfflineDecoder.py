import numpy as np
import pandas as pd
import pickle as pk
from phys.phys_utils import nnan_pearsonr as nnan_pearsonr
from phys import data_utils, phys_utils
import sys
import argparse
from phys.decode_utils import Decoder_3Ddata

parser = argparse.ArgumentParser()
parser.add_argument('--subject_id', default='rnn') # all_hand_dmfc rnn random perle_hand_dmfc mahler_hand_dmfc
parser.add_argument('--neural_data_to_use', default='neural_responses_reliable')
parser.add_argument('--ncomp', default=50, type=int)
parser.add_argument('--train_size', default=0.5, type=float)

default_save_path = '/Users/hansem/Dropbox (MIT)/MPong/phys/results/offline_decode_results/'
# '/om/user/rishir/lib/MentalPong/phys/results/offline_decode_results/'


class OfflineDecoder(object):
    def __init__(self, **kwargs):
        self.subject_id = kwargs.get('subject_id', 'all_hand_dmfc')
        self.neural_data_to_use = kwargs.get('neural_data_to_use', 'neural_responses_reliable')
        self.ncomp = kwargs.get('ncomp', 50)
        self.train_size = kwargs.get('train_size', 0.5)

        spec_suffix = '%d_occ' % (self.ncomp)  ### '_%d_%2.2f' % (self.ncomp, self.train_size)
        # spec_suffix = '%d' % (self.ncomp) # '_%d_%2.2f' % (self.ncomp, self.train_size)

        self.beh_to_decode = ['ball_final_y']  ##### ball_occ_start_y     ball_final_y

        self.save_fn = '%s/%s/offline_%s_%s_%s.pkl' % (default_save_path,self.beh_to_decode[0], self.subject_id,
                                                    self.neural_data_to_use, spec_suffix)

        preproc_func = 'none' if self.subject_id in ['random', 'rnn'] else 'pca'

        self.decoder_specs = {
            'train_size': self.train_size,
            'niter': 100,
            'groupby': 'condition',
            'matched_timepoints': True,
            'preprocess_func': preproc_func,
            'ncomp': self.ncomp,
            'prune_features': False,
        }
        return

    def set_conditions_to_test(self):
        meta_dfn = '/Users/hansem/Dropbox (MIT)/MPong/lib/MentalPong/pong_basic/RF/occluded_pong_bounce1_pad8_4speed/valid_meta_sample_full.pkl'
        # /user/rishir/data/pong_basic/RF/occluded_pong_bounce1_pad8_4speed/valid_meta_sample_full.pkl'
        meta = pd.read_pickle(meta_dfn)

        # subsample phys conditions
        unique_conds = phys_utils.PONG_BASIC_META_IDX
        py_meta_index = np.array(meta['meta_index'])
        ordered_condition_index = [np.nonzero(py_meta_index == uci)[0][0] for uci in unique_conds]
        meta_subsampled = meta.iloc[ordered_condition_index]
        meta_subsampled = meta_subsampled.reset_index(drop=True)

        # select conditions by bounce number
        n_bounce = np.array(meta_subsampled['n_bounce_correct'])
        self.t_cond_sel = {
            'all': np.nonzero(n_bounce >= 0)[0],
            'bounce': np.nonzero(n_bounce > 0)[0],
            'bounce_occ':[3, 9, 20, 27, 33, 34, 41, 48, 51, 75, 78],
            'bounce_vis':[1, 5, 10, 11, 15, 23, 29, 30, 36, 38, 42, 43, 47, 52, 69, 73],
            'no_bounce': np.nonzero(n_bounce == 0)[0],
        }
        return

    @staticmethod
    def load_one_model_and_subsample_conditions(fn_in):
        def load_one_model(fn):
            from phys.code.rnn_analysis.PongRNNSummarizer import PongRNNSummarizer
            from phys.code.rnn_analysis import data_utils as behavior_data_utils
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

    def load_rnn_data(self, fn):

        def map_position_to_mworks(x0, rnn_display_size=32.0, mwk_screen_size=20):
            hw = rnn_display_size / 2.0
            return (mwk_screen_size / 2.0) * (x0 - hw) / hw

        res_rnn = self.load_one_model_and_subsample_conditions(fn)
        state = np.array(res_rnn['res']['state'])
        output_f = np.squeeze(res_rnn['res']['output_f'])
        ball_y_f_mat = np.tile(np.nanmean(output_f, axis=1), (output_f.shape[1], 1)).T
        ball_y_f_mat = map_position_to_mworks(ball_y_f_mat)

        data_neur_nxcxt = np.transpose(state, (2, 0, 1))
        data_beh_bxcxt = np.array([ball_y_f_mat])
        mask_neur_cxt = mask_beh_cxt = np.array(res_rnn['masks']['output_vis-sim'][:, :, 0])
        mask_plot = np.array(res_rnn['masks']['output_vis'][:, :, 0])

        data = {
            'data_neur_nxcxt': data_neur_nxcxt,
            'data_beh_bxcxt': data_beh_bxcxt,
            'data_neur_2_nxcxt': None,
            'mask_neur_cxt': mask_neur_cxt,
            'mask_beh_cxt': mask_beh_cxt,
            'mask_plot': mask_plot,
            'time_window': np.arange(6, 12),  # 250-500ms aligned to mask_plot at 41ms steps
        }
        return data

    def load_neural_data_base(self, subject_id):
        neural_data = data_utils.load_neural_dataset(subject_id=subject_id, timebinsize=50, recompute_augment=False)

        condition = 'occ'
        beh_to_decode = self.beh_to_decode # ['ball_occ_start_y'] ##### ball_occ_start_y     ball_final_y ['ball_occ_start_y'] ##### ball_occ_start_y     ball_final_y
        beh_to_decode_2 = ['ball_pos_x_TRUE', 'ball_pos_y_TRUE', 'ball_pos_dx_TRUE', 'ball_pos_dy_TRUE']

        mask_fn_neur = mask_fn_beh = 'occ_end_pad0' ### 'start_end_pad0'
        mask_neur_cxt = np.array(neural_data['masks'][condition][mask_fn_neur])
        mask_beh_cxt = np.array(neural_data['masks'][condition][mask_fn_beh])

        data_neur_nxcxt = np.array(neural_data[self.neural_data_to_use][condition])
        data_beh_bxcxt = np.array([neural_data['behavioral_responses'][condition][fk] for fk in beh_to_decode])
        data_beh_2_bxcxt = np.array([neural_data['behavioral_responses'][condition][fk] for fk in beh_to_decode_2])
        mask_plot = np.array(neural_data['masks'][condition]['occ_end_pad0']) ### start_occ_pad0

        data = {
            'data_neur_nxcxt': data_neur_nxcxt,
            'data_beh_bxcxt': data_beh_bxcxt,
            'data_beh_2_bxcxt': data_beh_2_bxcxt,
            'data_neur_2_nxcxt': None,
            'mask_neur_cxt': mask_neur_cxt,
            'mask_beh_cxt': mask_beh_cxt,
            'mask_plot': mask_plot,
            'time_window_oi': np.arange(5, 10),  # 250-500ms aligned to mask_plot at 50ms steps
        }
        return data

    def load_neural_data(self, subject_id):
        # neurons to endpoint (n_t -> y_f)
        data_neur = self.load_neural_data_base(subject_id)
        data_neur['data_neur_2_nxcxt'] = None
        data_neur['data_beh_2_bxcxt'] = None
        return data_neur

    def load_neural_prediction_of_ball_state(self, subject_id, direct_mapping=True, ball_state_ndims=4):
        """ neural data is mapped to ball state (x,y,dx,dy). this is then mapped to yf.
        direct_mapping : True means that the decoder is trained and tested on the estimate ball state.
        direct_mapping: False means that the decoder is trained on the true ball state, and tested on the estimated
        ball state. """

        data_neur = self.load_neural_data_base(subject_id)
        data_neur_nxcxt = np.array(data_neur['data_neur_nxcxt'])
        data_beh_bxcxt = np.array(data_neur['data_beh_2_bxcxt'])  # test on neur -> ball state
        mask_neur_cxt = np.array(data_neur['mask_neur_cxt'])
        mask_beh_cxt = np.array(data_neur['mask_beh_cxt'])
        dec = Decoder_3Ddata(**self.decoder_specs)
        res_neur_to_state = dec.decode_one(data_neur_nxcxt, data_beh_bxcxt,
                                           mask_neur_cxt, mask_beh_cxt, data_neur_2_nxcxt=None)
        y_pred_mu = np.nanmean(res_neur_to_state['y_pred_dist'], axis=0)
        yp_cxtxb = phys_utils.unflatten_to_3d_mat(y_pred_mu,
                                                  res_neur_to_state['unflatten_res'])
        yp_bxcxt = np.transpose(yp_cxtxb, (2, 0, 1))
        if direct_mapping:
            data_state = {
                'data_neur_nxcxt': yp_bxcxt[:ball_state_ndims,:,:],  # use neural prediction of state to train
                'data_beh_bxcxt': np.array(data_neur['data_beh_bxcxt']),
                'data_neur_2_nxcxt': None,  # use output of trained mapping to test
                'mask_neur_cxt': np.array(data_neur['mask_neur_cxt']),
                'mask_beh_cxt': np.array(data_neur['mask_beh_cxt']),
                'mask_plot': np.array(data_neur['mask_plot']),
            }
        else:
            data_state = {
                'data_neur_nxcxt': np.array(data_neur['data_beh_2_bxcxt'][:ball_state_ndims,:,:]),  # use true ball state to train
                'data_beh_bxcxt': np.array(data_neur['data_beh_bxcxt']),
                'data_neur_2_nxcxt': yp_bxcxt[:ball_state_ndims,:,:],  # test trained mapping on neural prediction of state
                'mask_neur_cxt': np.array(data_neur['mask_neur_cxt']),
                'mask_beh_cxt': np.array(data_neur['mask_beh_cxt']),
                'mask_plot': np.array(data_neur['mask_plot']),
            }
        return data_state

    def run_offline_decoder_base(self, data):

        def unflatten_and_remask(res, fk, mask):
            x_2 = []
            for x in res[fk]:
                y = phys_utils.unflatten_to_3d_mat(x, res['unflatten_res'])
                ym = phys_utils.realign_masked_data(phys_utils.apply_mask(y, mask), align_to_start=True)
                x_2.append(ym)
            return np.squeeze(x_2)

        dec = Decoder_3Ddata(**self.decoder_specs)
        res_dec = dec.decode_one(data['data_neur_nxcxt'], data['data_beh_bxcxt'],
                                 data['mask_neur_cxt'], data['mask_beh_cxt'],
                                 data_neur_2_nxcxt=data['data_neur_2_nxcxt'])
        if data['data_neur_2_nxcxt'] is None:
            yp = unflatten_and_remask(res_dec, 'y_pred_dist', data['mask_plot'])
            yt = unflatten_and_remask(res_dec, 'y_true_dist', data['mask_plot'])
        else:
            yp = unflatten_and_remask(res_dec, 'y_pred_2_dist', data['mask_plot'])
            yt = unflatten_and_remask(res_dec, 'y_true_2_dist', data['mask_plot'])
        return {'yp': yp, 'yt': yt}

    def get_decoding_metrics(self, res):
        def r_t(x_ixcxt_, y_ixcxt_, t_select):
            x_ixcxt, y_ixcxt = x_ixcxt_[:, t_select, :], y_ixcxt_[:, t_select, :]
            niter, ntime = x_ixcxt.shape[0], x_ixcxt.shape[2]
            r_all = []
            for i in range(niter):
                x_cxt, y_cxt = x_ixcxt[i], y_ixcxt[i]
                r_all.append([nnan_pearsonr(x_cxt[:, t_i], y_cxt[:, t_i])[0] for t_i in range(ntime)])
            return np.array(r_all)

        def mae_t(x_ixcxt_, y_ixcxt_, t_select):
            x_ixcxt, y_ixcxt = x_ixcxt_[:, t_select, :], y_ixcxt_[:, t_select, :]
            niter, ntime = x_ixcxt.shape[0], x_ixcxt.shape[2]
            mae_all = []
            for i in range(niter):
                x_cxt, y_cxt = x_ixcxt[i], y_ixcxt[i]
                mae_all.append(np.nanmean(np.abs(x_cxt - y_cxt), axis=0))
            return np.array(mae_all)

        t_select_choices = self.t_cond_sel
        yp, yt = res['yp'], res['yt']
        metrics = {'yp': yp, 'yt': yt}
        for align_to_start in [True]:  # , False]:
            yp_curr = np.array([phys_utils.realign_masked_data(x, align_to_start=align_to_start) for x in yp])
            yt_curr = np.array([phys_utils.realign_masked_data(x, align_to_start=align_to_start) for x in yt])
            for t_select_fn in t_select_choices.keys():
                t_sel = t_select_choices[t_select_fn]
                metrics['r_start%d_%s' % (align_to_start, t_select_fn)] = r_t(yp_curr, yt_curr, t_sel)
                metrics['mae_start%d_%s' % (align_to_start, t_select_fn)] = mae_t(yp_curr, yt_curr, t_sel)

        return metrics

    def run_for_all_neural_models(self):

        data_to_test = {
            'neural_data': self.load_neural_data(self.subject_id),
            'pos_state_pred': self.load_neural_prediction_of_ball_state(self.subject_id,
                                                                        direct_mapping=True,
                                                                        ball_state_ndims=2),
            # 'pos_state_pred_transfer': self.load_neural_prediction_of_ball_state(self.subject_id,
            #                                                                      direct_mapping=False,
            #                                                                      ball_state_ndims=2),
            'posvel_state_pred': self.load_neural_prediction_of_ball_state(self.subject_id,
                                                                        direct_mapping=True,
                                                                        ball_state_ndims=4),
            # 'posvel_state_pred_transfer': self.load_neural_prediction_of_ball_state(self.subject_id,
            #                                                                      direct_mapping=False,
            #                                                                      ball_state_ndims=4),
        }

        all_metrics = {}
        for fk in data_to_test.keys():
            res_curr = self.run_offline_decoder_base(data_to_test[fk])
            all_metrics[fk] = self.get_decoding_metrics(res_curr)

        data_to_save = {'all_metrics': all_metrics}

        with open(self.save_fn, 'wb') as f:
            f.write(pk.dumps(data_to_save))
        return

    def run_for_all_rnns(self):
        summary_fn = '/Users/hansem/Dropbox (MIT)/MPong/lib/MentalPong/dat/comparison_summary_pred.pkl'
        # summary_fn = '/om/user/rishir/lib/MentalPong/dat/comparison_summary_pred.pkl'
        df = pd.read_pickle(summary_fn)
        df = df.iloc[:-4]

        all_metrics = {}
        state = {}
        for i in range(df.shape[0]):
            fn = df['filename'][i]
            data = self.load_rnn_data(fn)
            res_curr = self.run_offline_decoder_base(data)
            all_metrics[fn] = self.get_decoding_metrics(res_curr)
            state[fn] = data

        data_to_save = {'all_metrics': all_metrics, 'state': state, 'df': df} # , 'df': df

        with open(self.save_fn, 'wb') as f:
            f.write(pk.dumps(data_to_save))
        return

    def run_all(self):
        self.set_conditions_to_test()
        if self.subject_id == 'rnn':
            self.run_for_all_rnns()
        else:
            self.run_for_all_neural_models()
        return


def main(argv):
    flags_, _ = parser.parse_known_args(argv)
    flags = vars(flags_)

    decoder = OfflineDecoder(**flags)
    decoder.run_all()
    return


if __name__ == "__main__":
    main(sys.argv[1:])
