import sys
import argparse
import pickle as pk
from phys.decode_utils import Decoder_3Ddata
from phys import data_utils
from phys import dataset_augment_utils as aug_utils
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--masks_to_test_suffix', default='start_end_pad0') # start_end_pad0 occ_end_pad0
# ['start_end_pad0', 'start_occ_pad0', 'occ_end_pad0'] ['start_pad0_roll0', 'occ_pad0_roll0', 'half_pad0_roll0', 'f_pad0_roll0']:
parser.add_argument('--neural_data_to_use', default='neural_responses_reliable_FactorAnalysis_50') # neural_responses_reliable_FactorAnalysis_50 neural_responses_reliable
parser.add_argument('--ncomp', default=50, type=int)
parser.add_argument('--timebinsize', default=50, type=int) ### 10
parser.add_argument('--subject_id', default='all_hand_dmfc') ### perle_hand
parser.add_argument('--condition', default='occ')
parser.add_argument('--train_size', default=0.5, type=float)

default_save_path = '/Users/hansem/Dropbox (MIT)/MPong/phys/results/old_from_rishi_om/decode_results/'
# default_save_path = '/om/user/rishir/lib/MentalPong/phys/results/decode_results/'

class DecodeOverTime(object):
    def __init__(self, **kwargs):
        self.data = kwargs.get('data')
        self.masks_to_test_suffix = kwargs.get('masks_to_test_suffix')
        self.timebinsize = kwargs.get('timebinsize')
        self.subject_id = kwargs.get('subject_id')
        self.condition = kwargs.get('condition', 'occ')
        self.train_size = kwargs.get('train_size', 0.5)
        self.niter = kwargs.get('niter', 100)
        self.neural_data_to_use = kwargs.get('neural_data_to_use', 'neural_responses_reliable_FactorAnalysis_50')
        self.preprocess_func = kwargs.get('preprocess_func', 'none')
        self.ncomp = kwargs.get('ncomp', 50)


        self.save_path = default_save_path
        preproc_suffix = '_%s%d' % (self.preprocess_func, self.ncomp) if self.preprocess_func != 'none' else ''
        self.save_fn_suffix = '%s_%s_%s_%dms_%2.2f_%s%s' % (self.subject_id,
                                                          self.condition,
                                                          self.masks_to_test_suffix,
                                                          self.timebinsize,
                                                          self.train_size, self.neural_data_to_use, preproc_suffix)
        self.save_fn = '%s/decode_%s.pkl' % (self.save_path, self.save_fn_suffix)

        self.beh_to_decode = ['ball_pos_x_TRUE', 'ball_pos_y_TRUE', 'ball_pos_dx_TRUE', 'ball_pos_dy_TRUE', # 0 1 2 3
                              'ball_pos_dspeed_TRUE', 'ball_pos_dtheta_TRUE', # 4 5
                              'ball_pos_ego_x_TRUE', 'ball_pos_ego_y_TRUE', # 6 7 new
                              'paddle_pos_y', 'joy', # 8 9
                              'eye_v', 'eye_h', 'eye_dv', 'eye_dh', 'eye_dtheta', 'eye_dspeed', # 10 11 12 13 14 15
                              'eye_dx_postmean', 'eye_dy_postmean', 'eye_dtheta_postmean', 'eye_dspeed_postmean',
                              'hand_dx_postmean', 'hand_dy_postmean', 'hand_dtheta_postmean', 'hand_dspeed_postmean',
                              't_from_start', 't_from_occ', 't_from_end', # 24 25 26
                              'ball_pos_y_rel', 'target_y', 'target_y_relative',
                              'ball_final_y', 'ball_initial_y', 'ball_occ_start_y']

        self.max_roll = int(250 / self.timebinsize)  # 250ms

        self.multi_timepoint_masks = ['start_end_pad0', 'start_occ_pad0', 'occ_end_pad0']
        self.single_timepoint_masks = ['start_pad0_roll', 'occ_pad0_roll', 'f_pad0_roll'] # ['start_pad0_roll0', 'occ_pad0_roll0', 'half_pad0_roll0', 'f_pad0_roll0']:
        matched_timepoints = self.masks_to_test_suffix in self.multi_timepoint_masks

        self.decoder_specs = {
            'train_size': kwargs.get('train_size', self.train_size),
            'niter': kwargs.get('niter', self.niter),
            'groupby': kwargs.get('groupby', 'condition'),
            'matched_timepoints': matched_timepoints,
            'preprocess_func': self.preprocess_func,
            'ncomp': self.ncomp,
            'prune_features': False,
        }

        self.mask_conditions_to_test = None  # what masks to use? this gets filled in

        return

    def print_specs(self):
        print('masks_to_test_suffix', self.masks_to_test_suffix)
        print('condition', self.condition)
        print('subject_id', self.subject_id)
        print('timebinsize', self.timebinsize)
        print('data', self.data[self.neural_data_to_use][self.condition].shape)

        return

    def set_masks_multi_timepoint(self):
        # roll neural only, behavior should stay fixed because beh variables (e.g. ball) are
        # only present for some time during the trial.
        all_masks = list(self.data['masks'][self.condition].keys())
        mask_conditions_to_test = []
        mask_fn_template = self.masks_to_test_suffix

        mask_fn_rolled_list = [fk for fk in all_masks if '%s_roll' % mask_fn_template in fk]
        for mask_fn2 in mask_fn_rolled_list:
            roll_length = np.abs(float(mask_fn2.split('_roll')[-1]))
            if roll_length <= self.max_roll:
                mask_conditions_to_test.append([mask_fn2, mask_fn_template])
        return mask_conditions_to_test

    def set_masks_single_timepoint(self):
        # rolls for neural mask aligned to specific event. behavior mask stays fixed.
        def ensure_mask_within_trial(mask_fn_curr, mask_fn_template):
            mask_in_template = mask_fn_curr[:len(mask_fn_template)] == mask_fn_template
            mask_curr = self.data['masks'][self.condition][mask_fn_curr]
            mask_full_trial = np.array(self.data['masks'][self.condition]['start_end_pad0'])
            nconds = np.nansum(np.isfinite(mask_full_trial[np.isfinite(mask_curr)]))
            return mask_in_template & (nconds == mask_curr.shape[0])

        all_masks = list(self.data['masks'][self.condition].keys())
        mask_conditions_to_test = []
        mask_fn_template = self.masks_to_test_suffix
        mask_fn_n_list = [fk for fk in all_masks if ensure_mask_within_trial(fk, mask_fn_template)]
        for mask_fn_n in mask_fn_n_list:
            for mask_fn_b in ['start_pad0_roll0', 'occ_pad0_roll0', 'half_pad0_roll0', 'f_pad0_roll0']:
                mask_conditions_to_test.append([mask_fn_n, mask_fn_b])
        return mask_conditions_to_test

    def set_masks(self):

        if self.masks_to_test_suffix in self.multi_timepoint_masks:
            mask_conditions_to_test = self.set_masks_multi_timepoint()
        elif self.masks_to_test_suffix in self.single_timepoint_masks:
            mask_conditions_to_test = self.set_masks_single_timepoint()
        else:
            mask_conditions_to_test = None
        self.mask_conditions_to_test = mask_conditions_to_test
        return

    def decode_base(self, mask_fn_neur, mask_fn_beh, neural_sample=None):
        if neural_sample is None:
            data_neur_nxcxt = np.array(self.data[self.neural_data_to_use][self.condition])
        else:
            data_neur_nxcxt = np.array(self.data[self.neural_data_to_use][self.condition][:, neural_sample, :])
        data_beh_bxcxt = np.array([self.data['behavioral_responses'][self.condition][fk] for fk in self.beh_to_decode])

        mask_neur_cxt = np.array(self.data['masks'][self.condition][mask_fn_neur])
        mask_beh_cxt = np.array(self.data['masks'][self.condition][mask_fn_beh])

        dec = Decoder_3Ddata(**self.decoder_specs)
        return dec.decode_one(data_neur_nxcxt, data_beh_bxcxt,
                              mask_neur_cxt, mask_beh_cxt)

    def decode_over_time_for_mask_pairs(self):
        res_all = []
        for masks_curr in self.mask_conditions_to_test:
            mask_n, mask_b = masks_curr[0], masks_curr[1]
            res_all.append(self.decode_base(mask_n, mask_b))

        res_to_save = {'res_decode': res_all,
                       'ncond': self.data['meta'].shape[0],
                       'mask_conditions': self.mask_conditions_to_test,
                       'condition': self.condition,
                       'neural_data_to_use ': self.neural_data_to_use,
                       'beh_to_decode': self.beh_to_decode,
                       'decoder_specs': self.decoder_specs,
                       }
        with open(self.save_fn, 'wb') as f:
            f.write(pk.dumps(res_to_save))
        return


def main(argv):
    flags_, _ = parser.parse_known_args(argv)
    flags = vars(flags_)

    flags['data'] = data_utils.load_neural_dataset(subject_id=flags['subject_id'],
                                                   timebinsize=flags['timebinsize'],
                                                   recompute_augment=False,
                                                   compute_egocentric=True)

    decoder = DecodeOverTime(**flags)
    decoder.set_masks()
    decoder.print_specs()
    decoder.decode_over_time_for_mask_pairs()
    return


if __name__ == "__main__":
    main(sys.argv[1:])
