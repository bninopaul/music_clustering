
cimport numpy as np
import numpy as np
cimport cython

@cython.boundscheck(False)
def _partitionc(np.ndarray[dtype = np.int64_t , ndim = 1] signal, int window_length, int jump):
    """
    Params:
    signal: (one-dimensional array)
    window_length: (int) size of the window/frame
    jump: (int) length of distance between windows/frames

    Return:
    partitions: (list) list of partitioned windows/frames of length window_length and
    """

    cdef int signal_length
    #cdef np.ndarray[dtype = np.int64_t, ndim = 1] beg_index, end_index

    signal_length = len(signal)
    signal_index = np.arange(signal_length - window_length)
    beg_index = np.array(filter(lambda x: x%jump==0, signal_index))
    end_index = beg_index + window_length
    zipped = zip(beg_index, end_index)

    partitions = [signal[i[0]:i[1]] for i in zipped]
    return partitions

rate = 44100
window = 15*rate
jump = rate
#Spectral Centroid
#Source: https://en.wikipedia.org/wiki/Spectral_centroid
@cython.boundscheck(False)
def _spectral_centroidc(np.ndarray[dtype = np.int64_t, ndim =1] signal, float rate):
    """
    Params:
    signal: (one-dimensional array)
    rate: (int) sampling rate of the audio signal

    Return:
    spectral_centroid:(float)
    """
    cdef np.ndarray[dtype = np.float64_t, ndim = 1] fft_mag, freq
    cdef float spectral_centroid

    fft_mag = np.abs(np.fft.rfft(signal))
    freq = np.fft.rfftfreq(len(signal), 1/rate)
    spectral_centroid = np.dot(fft_mag, freq)/sum(fft_mag)
    return spectral_centroid
@cython.boundscheck(False)
def spectral_centroid_meanstdc(np.ndarray[dtype = np.int64_t, ndim = 1] signal, int rate = rate, int window_length = window , int jump = jump):
    """
    Params:
    signal:(one-dimensional array)
    rate: (int) sampling rate of the audio signal
    window_length: (int) see _partition()
    jump: (int) see_partition()

    Return:
    centroid_mean: (float) mean of the spectral centroids of a signal
    centroid_std: (float) standard deviation of the spectral centroids of a signal
    """
    cdef float centroid_mean, centroid_std

    partitions = _partitionc(signal, window_length, jump)
    centroids = map(lambda x: _spectral_centroidc(x, rate), partitions)
    centroid_mean = np.mean(centroids)
    centroid_std = np.std(centroids)
    return centroid_mean, centroid_std


#Spectral Roll-Off
#Source: http://webhome.csc.uvic.ca/~gtzan/output/tsap02gtzan.pdf
@cython.boundscheck(False)
def _spectral_rolloffc(np.ndarray[dtype = np.int64_t, ndim = 1] signal):
    """
    Params:
    signal: (one-dimensional array)

    Return:
    spectral_rolloff: (float) spectral roll-off of the signal
    """
    cdef np.ndarray[dtype = np.float64_t, ndim = 1] fft_mag, cumsum,
    cdef float dist85, spectral_rolloff
    fft_mag = np.abs(np.fft.rfft(signal))
    dist85 = 0.85*sum(fft_mag)
    cumsum = np.cumsum(fft_mag)
    lessdist85 = np.cumsum(cumsum<dist85)
    spectral_rolloff =lessdist85[-1]
    return spectral_rolloff

@cython.boundscheck(False)
def spectral_rolloff_meanstdc(np.ndarray[dtype = np.int64_t, ndim = 1] signal, int window_length = window, int jump = jump):
    """
    Params:
    signal: (one-dimensional array)
    window_length: (int) see _partition()
    jump: (int) see _partition()

    Return:
    roll_off_mean: (float) mean of the spectral roll-offs of the windows generated from the signal
    roll_off_std: (float) standard deviation of the spectral roll-offs of the windows generated from the signal
    """

    cdef float roll_off_mean, roll_off_std

    partitions = _partitionc(signal, window_length, jump)
    roll_offs = map(lambda x: _spectral_rolloffc(x), partitions)
    roll_off_mean = np.mean(roll_offs)
    roll_off_std = np.std(roll_offs)
    return roll_off_mean, roll_off_std


#Spectral Flow/Spectral Flux
#Source: http://webhome.csc.uvic.ca/~gtzan/output/tsap02gtzan.pdf
#spectral flow and spectral flux are the same thing.
@cython.boundscheck(False)
def spectral_flow_meanstdc(np.ndarray[dtype = np.int64_t, ndim = 1] signal,int window_length = window, int jump = jump):
    """
    Params:
    signal: (one-dimensional array)
    window_length: (int) see _partition()
    jump: (int) see _partition()

    Return:
    flow_mean: (float) mean of the spectral flow of the signal
    flow_std: (float) standard deviation of the spectral flow of the signal
    """
    cdef np.ndarray[dtype = np.int64_t, ndim = 2] partitions,
    cdef np.ndarray[dtype = np.float64_t, ndim = 2] normalized_partitions, Nt, Nt1,
    cdef np.ndarray[dtype = np.float64_t, ndim = 1] Ft
    cdef int n
    cdef float flow_mean, flow_std

    partitions = np.array(_partitionc(signal, window_length, jump))
    n = len(partitions)
    normalized_partitions = partitions/np.linalg.norm(partitions, axis =1).reshape((n, 1))

    Nt = np.array(normalized_partitions[1:])
    Nt1 = np.array(normalized_partitions[:len(normalized_partitions)-1])
    Ft = np.sum((Nt - Nt1)**2, axis = 1)
    flow_mean = Ft.mean()
    flow_std = Ft.std()
    return flow_mean, flow_std


#Spectral Crossing Rate
#Source: http://webhome.csc.uvic.ca/~gtzan/output/tsap02gtzan.pdf
@cython.boundscheck(False)
def zero_crossing_rate_meanstdc(np.ndarray[dtype = np.int64_t, ndim = 1] signal, int window_length = window, int jump = jump):
    """
    Params:
    signal: (one-dimensional array)
    window_length: (int) see _partition()
    jump: (int) see _partition()

    Return:
    zero_crossing_rate_mean: (float) mean of the zero crossing rates of the windows
    zero_crossing_rate_std: (float) standard deviation of the zero crossing rates of the windows
    """
    cdef np.ndarray[dtype = np.int64_t, ndim = 2] partitions,
    cdef float m,n, zero_crossing_rate_mean, zero_crossing_rate_std

    partitions = np.array(_partitionc(signal, window_length, jump))
    m,n = np.shape(partitions)
    sign, sign1 = partitions[:, 1:]>0, partitions[:, :n-1]>0
    zero_crossing_rate  = (np.sum(np.abs(sign - sign1), axis = 1))/n
    zero_crossing_rate_mean = zero_crossing_rate.mean()
    zero_crossing_rate_std = zero_crossing_rate.std()
    return zero_crossing_rate_mean, zero_crossing_rate_std


#Low Energy
#Source(definition of RMS energy): http://blog.prosig.com/2015/01/06/rms-from-time-history-and-fft-spectrum/
#SOurce(definition of low energy feature):  http://webhome.csc.uvic.ca/~gtzan/output/tsap02gtzan.pdf
@cython.boundscheck(False)
def rms_energy(signal):
    """
    Params:
    signal: (one-dimensional array)

    Return:
    rmsenergy: (float) root mean square energy of the signal
    """
    cdef float rmsenergy
    rmsenergy = np.sqrt(np.mean(signal**2))
    return rmsenergy

@cython.boundscheck(False)
def lowenergy_featurec(np.ndarray[dtype = np.int64_t, ndim = 1] signal, int aw_windowlength = 5*rate, int tw_windowlength = 10*rate, int jump = jump):
    """
    Params:
    signal: (one-dimensional array)
    aw_windowlength: (int) window length of the analysis windows
    tw_windowlength: (int) window length of the texture windows
    jump: (int) see _partition()

    Return:
    lowenergy_percentage: (float) between 0-1,the percentage of analysis windows that have less
                            RMS energy than the average RMS energy across the texture window
    """
    if tw_windowlength< aw_windowlength:
        raise ValueError
    cdef np.ndarray[dtype = np.int64_t, ndim = 2] partitions_analysis, partitions_texture
    cdef float lowenergy_percentage

    partitions_analysis = np.array(_partitionc(signal, aw_windowlength, jump))
    partitions_texture = np.array(_partitionc(signal, tw_windowlength, jump))
    rmsenergy_analysis = map(lambda x: rms_energy(x.astype(int)), partitions_analysis)
    rmsenergy_texture = map(lambda x: rms_energy(x.astype(int)), partitions_texture)
    ave_rmsenergy_texture = np.mean(rmsenergy_texture)
    lowenergy_percentage = np.mean(rmsenergy_analysis<ave_rmsenergy_texture)
    return lowenergy_percentage

def feature_wrapper(signal):
    centmean, centstd = spectral_centroid_meanstdc(signal)
    rollmean, rollstd = spectral_rolloff_meanstdc(signal)
    flowmean, flowstd = spectral_flow_meanstdc(signal)
    zrcmean, zrcstd = zero_crossing_rate_meanstdc(signal)
    lowenergy = lowenergy_featurec(signal)
    return [centmean, centstd, rollmean, rollstd, flowmean, flowstd, zrcmean, zrcstd, lowenergy]
