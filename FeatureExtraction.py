import os, sys
import pandas as pd
import numpy as np
from scipy import signal

# Open a file
path = "./Raw/"

dirs = os.listdir(path)

# This would print all the files and directories
for file in dirs:
    print file

desti = "./Feat/"

for m in range(0, len(
        dirs)):  # This for loop processes the raw data and builds up time and frequency domain statistical features

    rj = pd.read_csv(path + dirs[m])
    # listi[:]=[]
    listi = []
    for j in range(100, len(rj) - 100):
        if rj['AccelerometerX'][j] * 0 == 0:
            listi.append([rj['AccelerometerX'][j], rj['AccelerometerY'][j], rj['AccelerometerZ'][j]])
    listi = pd.DataFrame(listi, columns=['accx', 'accy', 'accz'])
    faccx = []
    faccy = []
    faccz = []
    faccx = abs(np.fft.fft(listi['accx'].values))
    faccy = abs(np.fft.fft(listi['accy'].values))
    faccz = abs(np.fft.fft(listi['accz'].values))
    st = 0
    en = 99
    feat = []
    cnt = 0
    fcnt = 0
    p = 0
    difx = []
    dify = []
    difz = []
    while (en < len(listi) and p < 2):
        macx = np.mean(listi['accx'].values[st:en])
        macy = np.mean(listi['accy'].values[st:en])
        macz = np.mean(listi['accz'].values[st:en])
        mfacx = np.mean(faccx[st:en])
        mfacy = np.mean(faccy[st:en])
        mfacz = np.mean(faccz[st:en])
        maxz = macx / macz
        mayz = macy / macz
        mfaxz = mfacx / mfacz
        mfayz = mfacy / mfacz
        maxi = np.amax(listi['accx'].values[st:en])
        may = np.amax(listi['accy'].values[st:en])
        maz = np.amax(listi['accz'].values[st:en])
        mix = np.amin(listi['accx'].values[st:en])
        miy = np.amin(listi['accy'].values[st:en])
        miz = np.amin(listi['accz'].values[st:en])
        specx = np.average(faccx[st:en], weights=listi['accx'].values[st:en])
        specy = np.average(faccy[st:en], weights=listi['accy'].values[st:en])
        specz = np.average(faccz[st:en], weights=listi['accz'].values[st:en])
        spec = np.average([specx, specy, specz])
        stdx = np.std(listi['accx'].values[st:en])
        stdy = np.std(listi['accy'].values[st:en])
        stdz = np.std(listi['accz'].values[st:en])
        stdav = np.mean([stdx, stdy, stdz])
        peakx = np.mean(signal.find_peaks_cwt(listi['accx'].values[st:en], np.arange(1, 10)))
        peaky = np.mean(signal.find_peaks_cwt(listi['accy'].values[st:en], np.arange(1, 10)))
        peakz = np.mean(signal.find_peaks_cwt(listi['accz'].values[st:en], np.arange(1, 10)))
        for i in range(st, en):
            cnt = cnt + np.sqrt(np.mean(np.square([listi['accx'][i], listi['accy'][i], listi['accz'][i]])))
            fcnt = fcnt + np.sqrt(np.mean(np.square([faccx[i], faccy[i], faccz[i]])))
            difx.append(abs(listi['accx'] - macx))
            dify.append(abs(listi['accy'] - macy))
            difz.append(abs(listi['accz'] - macz))

        cou = cnt / (en - st + 1)
        fcou = fcnt / (en - st + 1)
        dix = np.mean(difx)
        diy = np.mean(dify)
        diz = np.mean(difz)
        difx[:] = []
        dify[:] = []
        difz[:] = []
        cnt = 0
        fcnt = 0
        st = st + 50
        en = en + 50
        if en > len(listi):
            en = len(listi) - 1
            p = p + 1
        feat.append(
            [macx, macy, macz, mfacx, mfacy, mfacz, mfaxz, mfayz, cou, fcou, peakx, peaky, peakz, stdx, stdy, stdz,
             stdav, dix, diy, diz, maxi, may, maz, mix, miy, miz, specx, specy, specz, spec])
    print m

    featu = pd.DataFrame(feat,
                         columns=['macx', 'macy', 'macz', 'mfacx', 'mfacy', 'mfacz', 'mfaxz', 'mfayz', 'cou', 'fcou',
                                  'peakx', 'peaky', 'peakz', 'stdx', 'stdy', 'stdz', 'stdav', 'dix', 'diy', 'diz',
                                  'maxi', 'may', 'maz', 'mix', 'miy', 'miz', 'specx', 'specy', 'specz', 'spec'])
    name = str(dirs[m].split('_')[0] + '_' + dirs[m].split('_')[1] + '_ft_' + dirs[m].split('_')[2])
    featu.to_csv(desti + "\\" + dirs[m].split('_')[0] + '_' + dirs[m].split('_')[1] + '_ft_' + dirs[m].split('_')[2])
