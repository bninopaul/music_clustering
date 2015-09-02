from __future__ import division
import numpy as np
import scipy.signal as ss
import pywt
import itertools as it
import pandas as pd
import cython_func as cf
import librosa as lb


def segment_music(df,num_secs):
    segments = []
    for i in df.index:
        song = df.ix[i]['signal']
        num_pts = df.ix[i]['sample_rate']*num_secs
        trimmed = np.trim_zeros(song)
        while np.all(trimmed[:10])==False:
            trimmed = trimmed[5:]
        while np.all(trimmed[-10:])==False:
            trimmed = trimmed[:-5]
        seg_beg = trimmed[:num_pts].astype(int)
        seg_mid = trimmed[int(trimmed.shape[0]/2)-(num_pts/2):int(trimmed.shape[0]/2)+(num_pts/2)].astype(int)
        seg_end = trimmed[-num_pts:].astype(int)

        segments.append([seg_beg,seg_mid,seg_end])
    return segments

#shared functions
def _partition(signal, window_length=661500, jump=22050):
    """
    Params:
    signal: (one-dimensional array)
    window_length: (int) size of the window/frame
    jump: (int) length of distance between windows/frames

    Return:
    partitions: (list) list of partitioned windows/frames of length window_length and
    """
    signal_length = len(signal)
    signal_index = range(signal_length - window_length)
    beg_index = np.array(filter(lambda x: x%jump==0, signal_index))
    end_index = beg_index + window_length
    zipped = zip(beg_index, end_index)

    partitions = [signal[i[0]:i[1]] for i in zipped]
    return partitions

def lpfilter(signal,alpha=0.99):
    y = range(len(signal))
    y[0] = 0
    for i in range(1,len(signal)):
        y[i] = ((1-alpha) * signal[i]) + (alpha * y[i-1])
    return np.array(y)

def acf(signal):
    n = len(signal)
    array = np.array([np.dot(signal[:n-i], signal[i:])/float(n) for i in xrange(n)])
    return array

#arjay's functions
def auto2bpm(index, ln=10337):
    return int(60.0 * ln / index)

def bpm2auto(bpm, ln=10337):
    return int(60.0 * ln / bpm)

def beat_histogram(signal,sr=22050):
    histogram = []
    partition = _partition(signal)
    hz = [200., 400., 800., 1600., 3200.]
    for sig in partition:
        subbands = []
        #lowpass filter
        B, A = ss.butter(4,200./sr,btype='low')
        subbands.append(np.array(ss.lfilter(B,A,pywt.dwt(sig,'db4')[0])))
        #bandpass filter
        for t in range(len(hz)-1):
            B, A = ss.butter(4,[hz[t]/sr,hz[t+1]/sr],btype='bandpass')
            subbands.append(np.array(ss.lfilter(B,A,pywt.dwt(sig,'db4')[0])))
        #highpass filter
        B, A = ss.butter(4,3200./sr,btype='highpass')
        subbands.append(np.array(ss.lfilter(B,A,pywt.dwt(sig,'db4')[0])))
        #envelope extraction
        for s in range(len(subbands)):
            y = pywt.dwt(subbands[s],'db4')[0]
            y = abs(y) #full wave rectification
            y = lpfilter(y,0.99) #low pass filter
            y = y[::16] #downsampling
            subbands[s] = y - np.mean(y)#mean removal

        x = np.sum(subbands,axis=0)

        ac = acf(x) #enhanced autocorrelation

        pk = peak(ac[bpm2auto(200):bpm2auto(40)]) #peak finding

        histogram.append(pk)
    return [i for row in histogram for i in row]

def peak(array):
    z=np.diff(array)
    ind = []
    for i in range(len(z)-1):
        if z[i+1] < 0 and z[i] > 0:
            ind.append((array[i+1],auto2bpm(i+1+bpm2auto(200))))
    ind_ = []
    pk = []
    for k,j in sorted(ind):
        if j not in ind_:
            ind_.append(j)
            pk.append((k,j))
    return pk[-3:]

def bh_feat(signal):
    bh = beat_histogram(signal)
    y = np.zeros((200))
    for i, j in bh:
        y[j] = y[j] + i

    r_amp = y / np.sum(y)

    a1, a0 = sorted(r_amp)[-2:]
    ra = a1 / a0
    sm = np.sum(y)
    mx = sorted(y)[-2:]
    p2, p1 = [i for i,x in enumerate(y) for j in mx if j == x]

    return a0, a1, ra, p1, p2, sm


#joseph's functions
def peaks(signal):
    z=np.diff(signal)
    ind = []
    for i in range(len(z)-1):
        if z[i+1] < 0 and z[i] > 0:
            ind.append(i+1)
    amp = [signal[k] for k in ind[:3]]
    return zip(ind[:3],amp)

def decompose(signal):
    B_low,A_low = ss.cheby1(5, 1,1000./22050, 'low')
    B_hi,A_hi = ss.cheby1(5, 1,1000./22050, 'highpass')
    low_pass = ss.lfilter(B_low,A_low,signal)
    high_pass = ss.lfilter(B_hi,A_hi,signal)
    return low_pass, high_pass

def half_wave_rectify(signal):
    hwr = signal.copy()
    hwr[hwr<0]=0
    return hwr

def envelope_sum(signal):
    low,hi = decompose(signal)
    env_low = lpfilter(half_wave_rectify(low))
    env_hi = lpfilter(half_wave_rectify(hi))
    env_sum = env_low + env_hi
    return env_sum

def SACF(signal):
    low,hi = decompose(signal)
    env_low = lpfilter(half_wave_rectify(low))
    env_hi = lpfilter(half_wave_rectify(hi))
    acf_low = acf(env_low)
    acf_high = acf(env_hi)
    return acf_low + acf_high

def pitch_histogram(peak_tupple):
    p = np.arange(1,2028)
    a = np.zeros(len(p))
    for i,j in peak_tupple:
        a[i] = a[i] + j
    return np.array(zip(p,a)).astype(int)

def unfolded_histogram(pitch_histogram_tuple):
    x = pitch_histogram_tuple.T[0]
    n = map(lambda x: 12*np.log2(x/440)+69,x)
    return np.array(n),pitch_histogram_tuple.T[1]

def folded_histogram(pitch_histogram_tuple):
    x,y = unfolded_histogram(pitch_histogram_tuple)
    c = map(lambda x: x.astype(int) % 12, x)
    h = np.arange(12)
    amplitude_ufh = np.zeros(12)
    for i in range(len(h)):
        for j in range(len(c)):
            if h[i] == c[j]:
                amplitude_ufh[i] += y[j]
    return h,amplitude_ufh

def folded_histogram_max_amplitude(folded_histogram_period, folded_histogram_amplitude):
    index_of_max = np.argmax(folded_histogram_amplitude)
    return folded_histogram_period[index_of_max], folded_histogram_amplitude[index_of_max]

def folded_histogram_amplitude_sum(folded_histogram_amplitude):
    return np.sum(folded_histogram_amplitude)

def unfolded_histogram_period_max_amplitude(unfolded_histogram_period, unfolded_histogram_amplitude):
    index_of_max = np.argmax(unfolded_histogram_amplitude)
    return unfolded_histogram_period[index_of_max]

def folded_histogram_pitch_interval(folded_histogram_period, folded_histogram_amplitude):
    index_max = np.argmax(folded_histogram_amplitude)
    remove_max = folded_histogram_amplitude.copy()
    remove_max[index_max] = 0
    index_max_2 = np.argmax(remove_max)
    return abs(index_max - index_max_2)

def pitch_histogram_features(signal_segment):

    windowed_signal = _partition(signal_segment,2028,2028)
    sacf = map(SACF, windowed_signal)
    my_peaks_list = map(peaks, sacf)

    peaks_flat = np.array(list(it.chain.from_iterable(my_peaks_list))).astype(int)
    PH = pitch_histogram(peaks_flat)

    q,r = unfolded_histogram(PH)
    u,v = folded_histogram(PH)

    ### Feature 1: Period of Max Amplitude of Folded Histogram
    ### Feature 2: Max Amplitude of Max Amplitude of Folded Histogram
    ### Feature 3: Sum of Amplitude of Folded Histogram
    ### Feature 4: Period of Max Amplitude of unfolded Histogram
    ### Feature 5: Pitch Interval of Folded Histogram

    F1, F2 = folded_histogram_max_amplitude(u,v)
    F3 = folded_histogram_amplitude_sum(v)
    F4 = unfolded_histogram_period_max_amplitude(q,r)
    F5 = folded_histogram_pitch_interval(u, v)

    return [F3, F4, F2, F1, F5]

#mark's function
def get_mfcc_mean_sd(data_array, n = 5, sr = 44100, n_mfcc = 40, verbose = False):
    mfccs = lb.feature.mfcc(y=data_array, sr=sr, n_mfcc=n_mfcc)
    means = [x.mean() for x in mfccs[:n]]
    sds = [x.std() for x in mfccs[:n]]
    if verbose == True:
        print means
    return means + sds
#feature extraction function
def features(df):
    seg_col = ["beg", "mid", "end"]
    segments = pd.DataFrame(segment_music(df, 30), columns= seg_col) #segments are 30 seconds long
    df = pd.concat([df, segments], axis = 1)
    df.drop(["signal", "sample_rate", "channel"], axis = 1, inplace = True)


    for seg in seg_col:
        #carlo and arjay's features
        feature1_6 = pd.DataFrame(map(lambda x: bh_feat(x), df[seg]),
                                  columns = [seg+"_amp1",
                                             seg+"_amp2",
                                             seg+"_ratio_amp1_amp2",
                                             seg+"_per1",
                                             seg+"_per2",
                                             seg+"_beat_strength"])
        df = pd.concat([df, feature1_6], axis = 1)

        #paul's features
        feature7_15 = pd.DataFrame(map(lambda x: cf.feature_wrapper(x), df[seg]),
                                  columns = [seg+"_centroid_mean",
                                             seg+"_centroid_std",
                                             seg+"_rolloff_mean",
                                             seg+"_rolloff_std",
                                             seg+"_flow_mean",
                                             seg+"_flow_std",
                                             seg+"_zrc_mean",
                                             seg+"_zrc_std",
                                             seg+"_lowenergy"])
        df = pd.concat([df, feature7_15], axis = 1)

        #pucca and mark's features
        feature16_25 = pd.DataFrame(map(lambda x: get_mfcc_mean_sd(x), df[seg]),
                                    columns = [seg+"_%dmfcc_"%i+metric for metric in ["mean", "std"] for i in range(1, 6)])
        df = pd.concat([df, feature16_25], axis = 1)

        #joseph's features
        feature26_30 = pd.DataFrame(map(lambda x: pitch_histogram_features(x), df[seg]),
                                   columns = [seg+"_pitch_strength",
                                              seg+"_per_max_unfolded",
                                              seg+"_amp_max_folded",
                                              seg+"_per_max_folded",
                                              seg+"_intrvl_peaks_folded"])
        df = pd.concat([df, feature26_30], axis = 1)

    df.drop(["beg", "mid", "end"],axis = 1, inplace = True)

    return df
