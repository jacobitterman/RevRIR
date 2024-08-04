import os
import numpy as np
from scipy import signal, stats
import matplotlib.pyplot as plt
from lmfit.models import LinearModel


def extractTimeParams(h,fs,T30flag = False,rmsFlag = False,startRT = 0.25,finishRT = 0.8,Nstart = 0.025, Nfinish = 0.1,std_thresh = 1.5,plots = False):
    rt60v = np.zeros([h.shape[1],1])
    edtv = np.zeros([h.shape[1],1])
    c80v = np.zeros([h.shape[1],1])
    d20v = np.zeros([h.shape[1],1])
    d50v = np.zeros([h.shape[1],1])
    tsv = np.zeros([h.shape[1],1])
    Npeaksv = np.zeros([h.shape[1],1],dtype = np.int8)
    kurt_tv = np.zeros_like(rt60v)
    kurt_Hv = np.zeros_like(rt60v)
    kurt_RMSv = np.zeros_like(rt60v)

    for jj in range(h.shape[1]):
        rt60, edt = RT60_est(h[:,jj],fs,start = startRT,finish = finishRT,T30flag = T30flag,plots=plots)
        c80 = C80_est(h[:,jj],fs)
        d20 = D20_est(h[:,jj],fs)
        d50 = D50_est(h[:,jj],fs)
        ts = TS_est(h[:,jj],fs)

        assert h.shape[1] == 1
        idx1 = np.where(h != 0)[0][0]
        h = np.roll(h, -idx1, axis=0)
        Npeaks, kurt_t, kurt_H, kurt_RMS = Env_peaks(h[:, jj], start=Nstart, finish=Nfinish, rmsFlag=rmsFlag,
                                                     RMS_window=5, std_thresh=std_thresh, plots=plots)
        # if np.isnan(kurt_t) or np.isnan(kurt_H) or np.isnan(kurt_RMS):
        #     from IPython import embed; embed()

        rt60v[jj] = rt60
        edtv[jj] = edt
        c80v[jj] = c80
        d20v[jj] = d20
        d50v[jj] = d50
        tsv[jj] = ts
        Npeaksv[jj] = Npeaks
        kurt_tv[jj] = kurt_t
        kurt_Hv[jj] = kurt_H
        kurt_RMSv[jj] = kurt_RMS
    dict = {'rt60v':rt60v,'edtv':edtv,'c80v':c80v,'d20v':d20v,'d50v':d50v,'tsv':tsv,'Npeaksv':Npeaksv,'kurt_tv':kurt_tv,'kurt_Hv':kurt_Hv,'kurt_RMSv':kurt_RMSv}
    return dict

def extractFreqParams(h,fs,refFlag = True, refDet = 0, f0 = 100, Kbands = 1.5, Nbands = 5,std_thresh = 1.5, plots = False):
    Ndet = h.shape[1]
    RoTFmat = np.zeros_like(h)
    RTFmat = np.zeros_like(h)
    Kurtmat = np.zeros([Nbands,Ndet])
    STDmat = np.zeros([Nbands,Ndet])
    MCmat = np.zeros([Nbands,Ndet])
    Npeaksmat = np.zeros([Nbands,Ndet])
    for jj in range(Ndet):
        RoTF = np.abs(np.fft.fftshift(np.fft.fft(h[:,jj])))
        RoTFmat[:,jj] = RoTF
    ff = fs*np.linspace((-1/2),(1/2-1/h.shape[0]),h.shape[0])
    if Ndet < 2:
        #print('Less than 2 detectors\n')
        refFlag = False
    if refFlag == False:
        #print('RoTF')
        RTFmat = RoTFmat
    else:
        #print('RTF, reference det no. '+str(refDet)+'\n')
        RoTFref = RoTFmat[:, refDet]
        RoTFref = RoTFref[:,np.newaxis]
        RTFmat = RoTFmat/RoTFref

    bands = np.arange(Nbands) + 1
    f1v = bands*f0/Kbands
    f2v = bands*f0*Kbands
    for kk in range(Nbands):
        startidx = np.argmax(ff > f1v[kk])
        stopidx = np.argmax(ff>f2v[kk]) - 1
        for jj in range(Ndet):
            data = RTFmat[startidx:stopidx,jj]
            Kurt = stats.kurtosis(data)
            STD = np.std(data)
            MC  = np.where(np.diff(np.sign(data-np.mean(data))))[0].shape[0]
            Npeaks = freqPeaks(data,std_thresh = std_thresh,plots=plots)

            Kurtmat[kk,jj] = Kurt
            STDmat[kk,jj] = STD
            Npeaksmat[kk,jj] = Npeaks
            MCmat[kk,jj] = MC
    dict = {'f1v':f1v,'f2v':f2v,'Kurtmat':Kurtmat,'Npeaksmat':Npeaksmat,'STDmat':STDmat,'MCmat':MCmat}
    return dict

def freqPeaks(h,std_thresh,plots):
    h_hilbert = signal.hilbert(h)
    h_env = np.abs(h_hilbert)
    h_std = np.std(h)
    fidx = np.arange(h.shape[0])
    f_peaks, _ = signal.find_peaks(h, height=(std_thresh*h_std * np.ones_like(h_env), 1.1 * h_env))
    Npeaks = f_peaks.shape[0]
    if plots == True:
        plt.figure()
        plt.plot(fidx,h)
        plt.plot(fidx,h_env,'--r')
        plt.plot(f_peaks,h[f_peaks],'x')
        plt.show()
    return Npeaks

def RT60_est(h,fs,T30flag = False, start = 0.25,finish = 0.8, plots = True):
    curve = 10 * np.log10(np.flipud(np.cumsum(np.flipud(h**2)))) #reverse cumulative sum
    cs = curve -(curve[0])  #normalize EDC vector (to start at 0 dB).
    # Create time vector:
    tt = np.arange(cs.shape[0])/fs

    inital_delay = np.argmin(np.diff(10**(cs/10)))
    m10_idx = np.argmax(cs < -10)
    EDT = (m10_idx - inital_delay) / fs
    EDT = np.max([0.0,EDT])

    #linear fit for RT60
    if T30flag == False:
        startidx = np.round(start * cs.shape[0])
        finishidx = np.round(finish * cs.shape[0])
    else:
        startidx = np.argmax(cs < -5)
        finishidx = np.argmax(cs < -35) - 1

    tstart = startidx / fs
    tfinish = finishidx / fs
    tfit = tt[int(startidx):int(finishidx)]
    lin = LinearModel()
    pars = lin.guess(cs[int(startidx):int(finishidx)], x=tfit)
    out = lin.fit(cs[int(startidx):int(finishidx)], pars, x=tfit)
    afit = out.best_values['slope']
    bfit = out.best_values['intercept']
    rt60 = -60/afit
    if np.isnan(rt60) or rt60 > 10:
        rt60 = 10
    if plots:
        plt.figure()
        plt.plot(tt,cs)
        plt.vlines(x = tstart,ymin = cs[-1], ymax = 10,colors = 'k')
        plt.vlines(x = tfinish,ymin = cs[-1], ymax = 10,colors = 'k')
        plt.plot(tfit,afit*tfit+bfit,'--r')
        plt.title('EDC RT60 = ' + str(round(rt60,2)))
        plt.xlabel('time [s]')
        plt.ylabel('Power [dB]')
        plt.show()
    return rt60, EDT

# Clarity index C80
def C80_est(h,fs):
    n80 = round(0.08 * fs)
    if h.shape[0] > n80:
        tail80 = np.sum(h[n80:-1]**2)
        c80 = 10 * (np.log10(np.sum(h[0:int(n80)]**2)) - np.log10(tail80))
    else: # bound
        c80 = 40;
    if np.isnan(c80):
        c80 = 40
    return c80

#Frame definition D20
def D20_est(h,fs):
    n20 = round(0.02 * fs)
    if h.shape[0] > n20:
        d20 = 100 * sum(h[0:int(n20)]**2) / sum(h**2)
    else:# bound
        d20 = 100
    return d20

# Definition D50
def D50_est(h,fs):
    n50 = round(0.05 * fs)
    if h.shape[0] > n50:
        d50 = 100 * np.sum(h[0:int(n50)]** 2) / sum(h**2)
    else: # bound
        d50 = 100;
    return d50

#Centre time Ts
def TS_est(h,fs):
    tt = np.arange(h.shape[0])/fs
    ts = np.sum(tt*(h**2)) / np.sum(h**2)
    return ts

def Env_peaks(h, start = 0.05, finish = 0.2,rmsFlag = False, RMS_window = 10, std_thresh = 1 ,plots = True):
    len = h.shape[0]
    tt = np.arange(len)
    startidx = np.round(start * len)
    tstart =  startidx
    finishidx = np.round(finish * len)
    tfinish = finishidx
    tfit = tt[int(startidx):int(finishidx)]
    hfit = h[int(startidx):int(finishidx)]
    h_hilbert = signal.hilbert(hfit)
    h_env = np.abs(h_hilbert)
    h_std = np.std(hfit)
    h_rms = window_rms(hfit,window_size = RMS_window)
    if rmsFlag == True:
        t_peaks, _ = signal.find_peaks(h_rms, height=(h_std*np.ones_like(h_env), 1.1 * h_env))
    else:
        t_peaks,_ = signal.find_peaks(hfit,height = (std_thresh*h_std*np.ones_like(h_env),1.1*h_env))


    Npeaks = t_peaks.shape[0]
    kurt_t = stats.kurtosis(hfit)
    kurt_H = stats.kurtosis(h_env)
    kurt_RMS = stats.kurtosis(h_rms)


    if plots == True:
        plt.figure()
        plt.plot(tt,h)
        plt.plot(tfit,h_env,'--r')
        plt.plot(tfit, h_rms, '.k')

        plt.plot(t_peaks + tstart, hfit[t_peaks], 'x')

        plt.title('Npeaks = ' + str(Npeaks))
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.show()
    return Npeaks,kurt_t, kurt_H, kurt_RMS

def RTF(h,fs):
    curve = 10 * np.log10(np.flipud(np.cumsum(np.flipud(h**2)))) #reverse cumulative sum
    cs = curve -(curve[0])  #normalize EDC vector (to start at 0 dB).
    # Create time vector:
    tt = np.arange(cs.shape[0])/fs

def window_rms(a, window_size):
  a2 = np.power(a,2)
  window = np.ones(window_size)/float(window_size)
  return np.sqrt(np.convolve(a2, window, 'same'))

def parse_features(timeParams, freqParams):
    for k, v in timeParams.items():
        timeParams[k] = v.reshape(-1)[0]
    for k, v in freqParams.items():
        freqParams[k] = v.reshape(-1).tolist()
    all_time_ks = sorted(list(timeParams.keys()))
    all_freq_ks = sorted(list(freqParams.keys()))
    all_data = [timeParams[k] for k in all_time_ks]
    all_keys = ['time_' + k for k in all_time_ks]
    for k in all_freq_ks:
        all_data.extend(freqParams[k])
        all_keys.extend([f'freq_{k}_band_{j}' for j in range(len(freqParams[k]))])

    assert len(all_data) == len(all_keys)
    return all_data, all_keys

def example_run(verbose=False,
                fs=8000,
                c=343,):
    import rir_generator as rir
    import pickle as pkl

    s = np.array([1, 1, 1])
    r = np.array([[2, 2, 2]])
    L = [3, 3, 3]
    betav = np.array([0.887, 0.887, 0.887, 0.887, 0.887, 0.887])
    n_samples = 4096
    h = rir.generate(c=c,  # Sound velocity (m/s)
                     fs=fs,  # Sample frequency (samples/s)
                     r=r,
                     s=s,  # Source position [x y z] (m)
                     L=L,  # Room dimensions [x y z] (m)
                     beta=betav,
                     nsample=n_samples  # Number of output samples
                     )

    timeParams = extractTimeParams(h,
                                   fs,
                                   startRT=0.45,
                                   finishRT=0.75,
                                   Nstart=0.025,
                                   Nfinish=0.05,
                                   T30flag=False,
                                   rmsFlag=False,
                                   std_thresh=1.5,
                                   plots=verbose)
    freqParams = extractFreqParams(h,
                                   fs,
                                   refFlag=False,
                                   refDet=0,
                                   std_thresh=1.5,
                                   f0=100,
                                   Kbands=2,
                                   Nbands=5,
                                   plots=verbose)

    features_vec, features_name = parse_features(timeParams, freqParams)
    example_save_path = os.path.join(os.path.dirname(__file__), 'room_params_results.pkl')
    # if not os.path.exists(example_save_path):
    #     print('saving results pickle')
    #     pkl.dump((features_vec, features_name), open(example_save_path, 'wb'))
    features_vec_res, features_name_res = pkl.load(open(example_save_path, 'rb'))
    assert all([x == y for x,y in zip(features_vec_res,features_vec)])
    assert all([x == y for x,y in zip(features_name, features_name_res)])
    print('text passed OK')


if __name__ == '__main__':
    example_run()