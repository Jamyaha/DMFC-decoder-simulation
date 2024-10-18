import sys
import argparse
import pickle as pk
from phys.decode_utils import Decoder_3Ddata
from phys import data_utils
import numpy as np
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--masks_to_test_suffix')
parser.add_argument('--neural_data_to_use')
parser.add_argument('--timebinsize', default=10, type=int)
parser.add_argument('--subject_id', default='perle_hand')
parser.add_argument('--condition', default='occ')
parser.add_argument('--k_folds', default=5, type=int)

default_save_path = '/om/user/rishir/lib/MentalPong/phys/results/decode_results/'


class DimensionalityEstimate(object):
    def __init__(self, **kwargs):
        self.subject_id = kwargs.get('subject_id', 'perle_hand_dmfc')
        self.timebinsize = kwargs.get('timebinsize', 50)
        self.condition = kwargs.get('condition', 'occ')
        self.masks_to_test_suffix = kwargs.get('masks_to_test_suffix', 'start_end_pad0')
        self.neural_data_to_use = kwargs.get('neural_data_to_use', 'neural_responses_reliable')

        self.save_path = default_save_path
        self.save_fn_suffix = '%s_%s_%s_%dms_%s' % (self.subject_id,
                                                    self.condition,
                                                    self.masks_to_test_suffix,
                                                    self.timebinsize,
                                                    self.neural_data_to_use)
        self.save_fn = '%s/dim_%s.pkl' % (self.save_path, self.save_fn_suffix)

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
        multi_timepoint_masks = ['start_end_pad0', 'start_occ_pad0', 'occ_end_pad0']
        single_timepoint_masks = ['start_pad0_roll', 'occ_pad0_roll', 'f_pad0_roll']

        if self.masks_to_test_suffix in multi_timepoint_masks:
            mask_conditions_to_test = self.set_masks_multi_timepoint()
        elif self.masks_to_test_suffix in single_timepoint_masks:
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

    timebinsize = flags['timebinsize']
    subject_id = flags['subject_id']
    condition = flags['condition']
    flags['data'] = data_utils.load_neural_dataset(subject_id=subject_id, timebinsize=timebinsize)


    decoder = DecodeOverTime(**flags)
    decoder.set_masks()
    decoder.print_specs()
    decoder.decode_over_time_for_mask_pairs()
    return


if __name__ == "__main__":
    main(sys.argv[1:])
