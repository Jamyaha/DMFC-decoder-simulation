from utils import phys_utils
import numpy as np

def convert_allocentric_to_egocentric(dat_,condition='occ'):

    """ eye velocity and arm velocity should be computed on single trial and then averaged, but this is the version
    that estimates velocity from the trial-averaged eye and arm position."""

    eye_v = np.array(dat_['behavioral_responses'][condition]['eye_v'])
    eye_h = np.array(dat_['behavioral_responses'][condition]['eye_h'])

    # to get "true" ball trajectory, use data from trials where the paddle doesn't hit the ball
    other_analog_ch_suffix = 'TRUE'

    for fn_suffix in ['', '_%s' % other_analog_ch_suffix]:
        x = dat_['behavioral_responses'][condition]['ball_pos_x%s' % fn_suffix]
        y = dat_['behavioral_responses'][condition]['ball_pos_y%s' % fn_suffix]

        dat_['behavioral_responses'][condition]['ball_pos_ego_x%s' % fn_suffix] = x - eye_h
        dat_['behavioral_responses'][condition]['ball_pos_ego_y%s' % fn_suffix] = y - eye_v
    return dat_

def get_speed_from_position(x, y):
    dx = np.diff(x, axis=1, append=np.nan)
    dy = np.diff(y, axis=1, append=np.nan)

    t = np.isfinite(y) & np.isnan(dy)
    ni, nj = np.nonzero(t)
    for i, j in zip(ni, nj):
        dx[i, j] = dx[i, j - 1]
        dy[i, j] = dy[i, j - 1]

    dtheta = np.arctan2(dy, dx)
    dspeed = (dy ** 2 + dx ** 2) ** 0.5
    return dx, dy, dtheta, dspeed


def augment_with_ball_speed(dat_, condition='occ'):
    # to get "true" ball trajectory, use data from trials where the paddle doesn't hit the ball
    other_analog_ch_suffix = 'TRUE'

    for fn_suffix in ['', '_%s' % other_analog_ch_suffix]:
        x = dat_['behavioral_responses'][condition]['ball_pos_x%s' % fn_suffix]
        y = dat_['behavioral_responses'][condition]['ball_pos_y%s' % fn_suffix]

        dx, dy, dtheta, dspeed = get_speed_from_position(x, y)

        dat_['behavioral_responses'][condition]['ball_pos_dx%s' % fn_suffix] = dx
        dat_['behavioral_responses'][condition]['ball_pos_dy%s' % fn_suffix] = dy
        dat_['behavioral_responses'][condition]['ball_pos_dspeed%s' % fn_suffix] = dspeed
        dat_['behavioral_responses'][condition]['ball_pos_dtheta%s' % fn_suffix] = dtheta
    return dat_


def augment_with_trial_time(dat_, condition='occ'):
    ntr, ntime = dat_['masks'][condition]['start_occ_pad0'].shape
    t_from_start, t_from_occ, t_from_end = [], [], []
    base_t_ax = np.arange(ntime)

    vis_mask = dat_['masks'][condition]['start_occ_pad0']
    occ_mask = dat_['masks'][condition]['occ_end_pad0']

    for trial_i in range(ntr):
        vis_epoch = np.nonzero(np.isfinite(vis_mask[trial_i]))[0]
        occ_epoch = np.nonzero(np.isfinite(occ_mask[trial_i]))[0]

        t_from_start.append(base_t_ax - vis_epoch[0])
        t_from_occ.append(base_t_ax - occ_epoch[0])
        t_from_end.append(base_t_ax - occ_epoch[-1])

    dat_['behavioral_responses'][condition]['t_from_start'] = np.array(t_from_start).astype(float)
    dat_['behavioral_responses'][condition]['t_from_occ'] = np.array(t_from_occ).astype(float)
    dat_['behavioral_responses'][condition]['t_from_end'] = np.array(t_from_end).astype(float)

    return dat_


def augment_with_relative_dynamic_ball_position(dat_, condition='occ'):
    ball_y = dat_['behavioral_responses'][condition]['ball_pos_y']
    paddle_y = dat_['behavioral_responses'][condition]['paddle_pos_y']
    dat_['behavioral_responses'][condition]['ball_pos_y_rel'] = ball_y - paddle_y
    return dat_


def augment_with_target(dat_, condition='occ'):
    ball_y = np.array(dat_['behavioral_responses'][condition]['ball_pos_y'])
    mask = np.array(dat_['masks'][condition]['f_pad0'])
    ball_y_f = phys_utils.apply_mask(ball_y, mask)
    ball_y_f_mat = np.tile(np.nanmean(ball_y_f, axis=1), (ball_y_f.shape[1], 1)).T

    dat_['behavioral_responses'][condition]['target_y'] = ball_y_f_mat
    paddle_y = dat_['behavioral_responses'][condition]['paddle_pos_y']
    dat_['behavioral_responses'][condition]['target_y_relative'] = ball_y_f_mat - paddle_y
    return dat_


def augment_with_static_ball_position(dat_, condition='occ'):
    static_masks = {
        'ball_final_y': 'f_pad0',
        'ball_initial_y': 'start_pad0',
        'ball_occ_start_y': 'occ_pad0'
    }
    ball_y = np.array(dat_['behavioral_responses'][condition]['ball_pos_y'])
    for sfn in static_masks.keys():
        mask_fn = static_masks[sfn]
        mask = np.array(dat_['masks'][condition][mask_fn])
        ball_y_f = phys_utils.apply_mask(ball_y, mask)
        ball_y_f_mat = np.tile(np.nanmean(ball_y_f, axis=1), (ball_y_f.shape[1], 1)).T
        dat_['behavioral_responses'][condition][sfn] = ball_y_f_mat
    return dat_


def augment_with_movement_variables_postmean(dat_, condition='occ'):
    """ eye velocity and arm velocity should be computed on single trial and then averaged, but this is the version
    that estimates velocity from the trial-averaged eye and arm position."""

    eye_v = np.array(dat_['behavioral_responses'][condition]['eye_v'])
    eye_h = np.array(dat_['behavioral_responses'][condition]['eye_h'])
    eye_dx, eye_dy, eye_dtheta, eye_dspeed = get_speed_from_position(eye_h, eye_v)
    dat_['behavioral_responses'][condition]['eye_dx_postmean'] = eye_dx
    dat_['behavioral_responses'][condition]['eye_dy_postmean'] = eye_dy
    dat_['behavioral_responses'][condition]['eye_dtheta_postmean'] = eye_dtheta
    dat_['behavioral_responses'][condition]['eye_dspeed_postmean'] = eye_dspeed

    hand_x = np.zeros(dat_['behavioral_responses'][condition]['joy'].shape)
    hand_y = np.array(dat_['behavioral_responses'][condition]['joy'])
    hand_dx, hand_dy, hand_dtheta, hand_dspeed = get_speed_from_position(hand_x, hand_y)
    dat_['behavioral_responses'][condition]['hand_dx_postmean'] = hand_dx
    dat_['behavioral_responses'][condition]['hand_dy_postmean'] = hand_dy
    dat_['behavioral_responses'][condition]['hand_dtheta_postmean'] = hand_dtheta
    dat_['behavioral_responses'][condition]['hand_dspeed_postmean'] = hand_dspeed

    return dat_


def augment_with_masks(dat, condition='occ'):
    ntr, ntime = dat['masks'][condition]['start_end_pad0'].shape

    def add_point_masks(mask_fn_):
        tmp = np.ones((ntr, ntime)) * np.nan
        for i in range(ntr):
            if mask_fn_ == 'start_pad0':
                j = np.nonzero(np.isfinite(dat['masks'][condition]['start_end_pad0'][i, :]))[0][0]
            elif mask_fn_ == 'occ_pad0':
                j = np.nonzero(np.isfinite(dat['masks'][condition]['occ_end_pad0'][i, :]))[0][0]
            elif mask_fn_ == 'half_pad0':  # halfway in time.
                j1 = np.nonzero(np.isfinite(dat['masks'][condition]['occ_end_pad0'][i, :]))[0][0]
                j2 = np.nonzero(np.isfinite(dat['masks'][condition]['occ_end_pad0'][i, :]))[0][-1]
                j = int((j1 + j2) / 2)
            elif mask_fn_ == 'f_pad0':
                j = np.nonzero(np.isfinite(dat['masks'][condition]['start_end_pad0'][i, :]))[0][-1]
            tmp[i, j] = 1
        return tmp

    for mask_fn in ['start_pad0', 'occ_pad0', 'f_pad0', 'half_pad0']:
        dat['masks'][condition][mask_fn] = add_point_masks(mask_fn)
        for roll_by in np.arange(-50, 50):
            if (roll_by < 0) & ('start_' in mask_fn):
                continue
            if (roll_by > 0) & ('f_' in mask_fn):
                continue
            mask_curr = dat['masks'][condition][mask_fn]
            dat['masks'][condition]['%s_roll%d' % (mask_fn, roll_by)] = np.roll(mask_curr, roll_by, axis=1)

    for mask_fn in ['start_end_pad0', 'occ_end_pad0', 'start_occ_pad0']:
        for roll_by in np.arange(-50, 50):
            mask_curr = np.array(dat['masks'][condition][mask_fn])
            mask_curr_rolled = np.roll(mask_curr, roll_by, axis=1)
            # mask_curr_rolled[np.isnan(mask_curr)] = np.nan
            dat['masks'][condition]['%s_roll%d' % (mask_fn, roll_by)] = mask_curr_rolled
    return dat


def augment_with_control_features(dat, condition='occ'):
    def pool_variables_into_mat(cv):
        cv_mat = []
        for cfn in cv:
            cv_mat.append(dat['behavioral_responses'][condition][cfn])
        return np.array(cv_mat)

    clist1 = ['ball_pos_x', 'ball_pos_y',
              'ball_pos_dx', 'ball_pos_dy']
    clist2 = ['paddle_pos_y', 'joy', 'eye_v', 'eye_h', 'eye_dv', 'eye_dh']

    dat['control_responses_1'] = {
        condition: pool_variables_into_mat(clist1)
    }
    dat['control_responses_2'] = {
        condition: pool_variables_into_mat(clist2)
    }
    dat['control_responses_all'] = {
        condition: pool_variables_into_mat(clist1 + clist2)
    }
    return dat


def augment_with_reliable_units(dat, condition='occ',
                                neural_data_to_use='neural_responses',
                                mask_full='start_end_pad0',
                                reliability_thres_pval=0.01):
    def get_reliable_units_idx():
        mask = np.array(dat['masks'][condition][mask_full])
        sf1 = '%s_sh1' % neural_data_to_use
        sf2 = '%s_sh2' % neural_data_to_use
        x_sh1 = np.array(dat[sf1][condition])
        x_sh2 = np.array(dat[sf2][condition])
        nneur = x_sh1.shape[0]
        reliability_r, reliability_p,n_data = [], [],[]
        for i in range(nneur):
            x1_, x2_ = x_sh1[i][np.isfinite(mask)], x_sh2[i][np.isfinite(mask)]
            r, p = phys_utils.nnan_pearsonr(x1_, x2_)
            reliability_p.append(p)
            reliability_r.append(r)
            n_data.append(x1_.shape[0])
        reliability_p = np.array(reliability_p)
        reliability_r = np.array(reliability_r)
        n_data = np.array(n_data)
        return np.nonzero((reliability_p < reliability_thres_pval) & (reliability_r > 0))[0]

    neur_oi = get_reliable_units_idx()
    dat['reliable_neural_idx'] = {condition: neur_oi}

    fk = 'neural_responses_reliable'
    if fk not in dat.keys():
        dat[fk] = {}
        dat['%s_sh1' % fk] = {}
        dat['%s_sh2' % fk] = {}

    for suffix in ['', '_sh1', '_sh2']:
        fr = dat['%s%s' % (neural_data_to_use, suffix)][condition]
        dat['%s%s' % (fk, suffix)][condition] = fr[neur_oi]

    return dat


def augment_with_low_dim_factors_base(dat, condition='occ',
                                      neural_data_to_use='neural_responses_reliable',
                                      mask_full='start_end_pad0',
                                      method='FactorAnalysis', num_dims=50):
    m = dat['masks'][condition][mask_full]
    fk = 'neural_responses_reliable_%s_%d' % (method, num_dims)

    if fk not in dat.keys():
        dat[fk] = {}
        dat['%s_sh1' % fk] = {}
        dat['%s_sh2' % fk] = {}

    for suffix in ['', '_sh1', '_sh2']:
        FR = dat['%s%s' % (neural_data_to_use, suffix)][condition]
        FR_embed, transformer_ = phys_utils.get_embedding(FR, m, method=method, n_components=num_dims)
        dat['%s%s' % (fk, suffix)][condition] = FR_embed

    return dat


def augment_with_low_dim_factors(dat, condition='occ', mask_full='start_end_pad0', ):
    # factors are extracted from all trials of reliable neurons.
    # technically, this is not within the cross validation train split,
    # so some information about test set could leak out.
    # however, checked empirically that this is negligible.
    for factor_method in ['FactorAnalysis']:  # FactorAnalysis_wrotation, PCA
        for num_dims in [10, 20, 50]:
            dat = augment_with_low_dim_factors_base(dat, condition=condition,
                                                    neural_data_to_use='neural_responses_reliable',
                                                    mask_full=mask_full,
                                                    method=factor_method, num_dims=num_dims)
    return dat


def augment_data_structure(dat, condition='occ',
                           mask_full='start_end_pad0',
                           reliability_thres_pval=0.01):
    dat = augment_with_masks(dat, condition=condition)
    dat = augment_with_movement_variables_postmean(dat, condition=condition)
    dat = augment_with_ball_speed(dat, condition=condition)
    dat = augment_with_trial_time(dat, condition=condition)
    dat = augment_with_relative_dynamic_ball_position(dat, condition=condition)
    dat = augment_with_static_ball_position(dat, condition=condition)
    dat = augment_with_target(dat, condition=condition)

    # dat = augment_with_control_features(dat, condition=condition)
    dat = augment_with_reliable_units(dat, condition=condition,
                                      neural_data_to_use='neural_responses',
                                      mask_full=mask_full,
                                      reliability_thres_pval=reliability_thres_pval)
    dat = augment_with_low_dim_factors(dat, condition=condition, mask_full=mask_full)

    return dat
