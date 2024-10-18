import sys
import os
import numpy as np
from glob import glob
import pickle as pk

PATH_ = '/om/user/rishir/lib/analysis-tools/Python3/'
sys.path.insert(0, PATH_)
PATH_ = '/om/user/rishir/lib/SpikeGLX_Datafile_Tools/Python/'
sys.path.insert(0, PATH_)

import OpenEphys as OE_oe
from pathlib import Path

from DemoReadSGLXData.readSGLX import readMeta, SampRate, makeMemMapRaw, ExtractDigital


def convert_samples_to_ms(x, fs=30000):
    # default 30000 samples/sec
    return np.array(x) / (fs / 10 ** 3)


##########################################################
# Open-ephys daq utils
##########################################################
def get_oe_data_directory(data_directory):
    dir_fn = glob('%s/*/100_CH1.continuous' % data_directory)[0]
    return "/".join(dir_fn.split('/')[:-1])


def get_oe_sync_on(data_directory):
    oe_data_directory = get_oe_data_directory(data_directory)
    fn = '%s/all_channels.events' % oe_data_directory
    tmp = OE_oe.load(fn)
    t = (tmp['channel'] == 0) & (tmp['eventId'] == 1)
    t_sync_on_oe = convert_samples_to_ms(tmp['timestamps'][t])
    return t_sync_on_oe


def get_oe_neural_data_raw(data_directory, kilosort_subdirectory):
    """
    Open-ephys has annoying convention for storing events and continuous data.
    Continuous data is stored as relative samples, starting at 0.
        t_relative = [0, 1, 2, …. N]
    which corresponds to samples
        t_absolute = [offset+0, offset+1, …., offset+N]

    This offset is necessary to map to event data, which is stored as absolute timestamps.

    The offset should be a fixed number, if the recording isn’t paused.
    otherwise, it should be updated for every 1024 samples, I think.

    So to align:
        add offset to timestamps of continuous data (as we do for spikes)
    or
        subtract offset to timestamps of event data (as we do for photodiode)

    From wiki:
        The timestamps are represented in int64 format, representing the sample count from the start of acquisition,
         which can be converted to seconds by dividing into the corresponding sample rate.
        In the case of continuous data, the timestamp corresponding to the first sample of each processing block
        can be obtained with the getTimestamp() method.
        For spikes and events, the timestamp is embedded into the data structure itself.
    """
    oe_data_directory = get_oe_data_directory(data_directory)
    fn = '%s/100_CH1.continuous' % oe_data_directory
    tmp = OE_oe.load(fn)
    offset = tmp['timestamps'][0]
    # The timestamps are represented in int64 format, representing the sample count from the start of acquisition,
    #  which can be converted to seconds by dividing into the corresponding sample rate.
    # In the case of continuous data, the timestamp corresponding to the first sample of each processing block
    # can be obtained with the getTimestamp() method.
    # For spikes and events, the timestamp is embedded into the data structure itself.

    neural_data_raw = {}
    # spike_templates: original unit assignment; spike_clusters: merged unit assignment.
    for fn_ in ['spike_times', 'templates', 'spike_templates']:
        neural_data_raw[fn_] = np.load('%s/%s/%s.npy' % (data_directory, kilosort_subdirectory, fn_))

    neural_data_raw['spike_times'] = neural_data_raw['spike_times'] + offset  # in samples, not seconds
    neural_data_raw['spike_times'] = convert_samples_to_ms(neural_data_raw['spike_times'])  # in ms

    return neural_data_raw


def get_oe_photodiode(data_directory, photodiode_ad_fn):
    oe_data_directory = get_oe_data_directory(data_directory)
    fn = '%s/%s' % (oe_data_directory, photodiode_ad_fn)
    photodiode_oe = OE_oe.load(fn)
    return {'data': photodiode_oe['data'], 'offset': photodiode_oe['timestamps'][0]}


##########################################################
# PXI->spikeglx utils
##########################################################
def extract_spikeglx_sync_on(data_directory):
    """ this could be replaced by CatGT (Optionally extract tables of sync waveform
     edge times to drive TPrime.) installed on windows machine. """

    def get_ttl_rise_base(t, x):
        dx = np.diff(x[0, :])
        t_up = np.nonzero(dx == 1)[0] + 1
        t_down = np.nonzero(dx == 255)[0] + 1
        return t[t_up], t[t_down]

    bin_fn = glob('%s/*.ap.bin' % data_directory)
    binFullPath = Path(bin_fn[0])
    meta = readMeta(binFullPath)
    rawData = makeMemMapRaw(binFullPath, meta)

    nChan = int(meta['nSavedChans'])
    nFileSamp = int(int(meta['fileSizeBytes']) / (2 * nChan))
    dw, dLineList = 0, [6]

    n_batches = 50
    batch_t = [int(x) for x in np.linspace(0, nFileSamp, n_batches)]
    t_up_all, t_down_all = [], []
    for i in range(len(batch_t) - 1):
        first_samp = int(batch_t[i])
        last_samp = int(batch_t[i + 1])
        last_samp = np.min([last_samp, nFileSamp - 1])
        dig_array = ExtractDigital(rawData, first_samp, last_samp, dw, dLineList, meta)
        t_curr = np.arange(first_samp, last_samp + 1)
        tup, tdown = get_ttl_rise_base(t_curr, dig_array)
        t_up_all.extend(tup)
        t_down_all.extend(tdown)

    events = {'t_up': np.array(t_up_all), 't_down': np.array(t_down_all)}
    save_fn = '%s/events.pkl' % data_directory
    with open(save_fn, 'wb') as f:
        f.write(pk.dumps(events))
    return


def get_spikeglx_sync_on(data_directory):
    fn = '%s/events.pkl' % data_directory
    if os.path.isfile(fn) is False:
        extract_spikeglx_sync_on(data_directory)
    events = pk.load(open(fn, 'rb'))
    t = events['t_up']
    t_sync_on_oe = convert_samples_to_ms(t)
    return t_sync_on_oe


def get_spikeglx_neural_data_raw(data_directory, kilosort_subdirectory):
    offset = 0  # is there an offset for spikeglx?
    # When you are running with sync enabled on the Sync tab, whenever a file-writing trigger event occurs and a
    # new set of files is started, SpikeGLX internally uses the edges to make sure that the files all start at a common
    # wall time. Each file's metadata records the sample index number of the first sample in that file: firstSample.
    # These samples are aligned to each other. Said another way, the files share a common T0.

    neural_data_raw = {}
    # spike_templates: original unit assignment; spike_clusters: merged unit assignment.
    for fn_ in ['spike_times', 'templates', 'spike_templates']:
        neural_data_raw[fn_] = np.load('%s/%s/%s.npy' % (data_directory, kilosort_subdirectory, fn_))

    neural_data_raw['spike_times'] = neural_data_raw['spike_times'] + offset  # in samples, not seconds
    neural_data_raw['spike_times'] = convert_samples_to_ms(neural_data_raw['spike_times'])  # in ms

    return neural_data_raw


def get_spikeglx_photodiode(data_directory, photodiode_ad_fn):
    # no photodiode is saved on spikeglx without nidaq breakout board
    return {'data': None, 'offset': None}


##########################################################
# utils for linking mworks to phys_daq
##########################################################

def match_sync_signals(sync_b, sync_p):
    """ sync signals should be identical across behavior and phys, except for an offset (start clock time).
    However, if one stream was started before the other, or if for some reason (eye calib in between?),
    some syncs are dropped in between, then more work is needed to match sync signals.
    """

    n_b, n_p = sync_b.shape[0], sync_p.shape[0]
    dt_b, dt_p = np.diff(sync_b), np.diff(sync_p)
    n = np.nanmin([n_b, n_p])
    del_n = n_b - n_p

    def nnan_pearsonr(x, y):
        from scipy.stats import pearsonr
        ind = np.isfinite(x) & np.isfinite(y)
        if np.nansum(ind) < 2:
            return None, None
        x, y = x[ind], y[ind]
        return pearsonr(x, y)

    def nanmin(x):
        return np.min(x[np.isfinite(x)])

    def evaluate_match(sb, sp):

        time_err_per_time = (sb - nanmin(sb)) - (sp - nanmin(sp))
        time_err = np.nanmean(np.abs(time_err_per_time))
        dsb, dsp = np.diff(sb), np.diff(sp)
        r, p = nnan_pearsonr(dsb, dsp)
        n_trials = np.nansum(np.isfinite(sb))
        err = np.nanmean(np.abs(dsb - dsp))
        res = {'n_trials': n_trials,
               'delta_err': err,
               'time_err': time_err,
               'r': r,
               'sb': sb,
               'sp': sp}
        return res

    def test_same_start():
        # assume that both streams start at the same time, and find window of matched trials
        sync_b_2 = np.ones(sync_b.shape) * np.nan
        sync_p_2 = np.ones(sync_b.shape) * np.nan
        win = np.arange(0, n, 1)
        sync_b_2[win] = sync_b[win]
        sync_p_2[win] = sync_p[win]
        return evaluate_match(sync_b_2, sync_p_2)

    def test_same_end():
        # assume that both streams start at the same time, and find window of matched trials
        sync_b_2 = np.ones(sync_b.shape) * np.nan
        sync_p_2 = np.ones(sync_b.shape) * np.nan

        win2_p = np.arange(n_p - n, n_p, 1)
        win2_b = np.arange(n_b - n, n_b, 1)

        sync_b_2[win2_b] = sync_b[win2_b]
        sync_p_2[win2_b] = sync_p[win2_p]
        return evaluate_match(sync_b_2, sync_p_2)

    def test_break_point_in_middle(thres_offset_ms=5):
        sync_b_2 = np.ones(sync_b.shape) * np.nan
        sync_p_2 = np.ones(sync_b.shape) * np.nan

        err = np.abs(dt_b[:n - 1] - dt_p[:n - 1])

        segment_start = np.nonzero(err > thres_offset_ms)[0][0] + 1
        win1 = np.arange(0, int(segment_start), 1)
        sync_b_2[win1] = sync_b[win1]
        sync_p_2[win1] = sync_p[win1]

        if del_n < 0:
            win2_p = np.arange(int(segment_start) + np.abs(del_n), n_p, 1)
            win2_b = np.arange(int(segment_start), n_b, 1)
        elif del_n > 0:
            win2_p = np.arange(int(segment_start), n_p, 1)
            win2_b = np.arange(int(segment_start) + np.abs(del_n), n_b, 1)
        elif del_n == 0:
            win2_p, win2_b = [], []

        sync_b_2[win2_b] = sync_b[win2_b]
        sync_p_2[win2_b] = sync_p[win2_p]
        return evaluate_match(sync_b_2, sync_p_2)

    def find_best_match():
        res_dict = []
        test_funcs = [test_same_start, test_same_end, test_break_point_in_middle]
        for f_idx, f in enumerate(test_funcs):
            try:
                res_ = f()
                res_dict.append(res_)
            except:
                print('sync match attempt %d failed' % f_idx)

        r_all = np.array([r_d['r'] for r_d in res_dict])
        time_err_all = np.array([r_d['time_err'] for r_d in res_dict])

        possible_idx = np.nonzero(r_all > 0.99)[0]
        idx_2 = np.nanargmin(time_err_all[possible_idx])
        idx = possible_idx[idx_2]
        return res_dict[idx]

    if (n_b == n_p) and nnan_pearsonr(dt_b, dt_p)[0] > 0.99:
        return evaluate_match(sync_b, sync_p)
    else:
        return find_best_match()
