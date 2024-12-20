import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import binned_statistic

MinLambda = 420 
MaxLambda = 720  

pattern = re.compile(r'^[\x00-\x7F]*([0-9]{3})-([0-9]{3})[\x00-\x7F]*\.txt$')

def load_cv(file_path, two_way=False):
    df = pd.read_csv(file_path, header=0, sep='\t')
    cols = df.columns
    smu1c = df[cols[-3]].values[2:-1].astype(float)
    smu2v = df[cols[-2]].values[2:-1].astype(float)
    if two_way:
        return smu1c, smu2v
    else:
        idx_end = np.where(smu2v == smu2v.max())[0][0] + 1
    return smu1c[:idx_end], smu2v[:idx_end]

def load_calibration_data(directory, extension, two_way=False):
    data = {}
    c_dark = None
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                if "Dark" in file:
                    c_dark, v_dark = load_cv(os.path.join(root, file), two_way)
                match = pattern.search(file)
                if match:
                    s = int(match.group(1))
                    e = int(match.group(2))
                    if s > MaxLambda -10 or s < MinLambda:
                        continue
                    curr_path = os.path.join(root, file)
                    current, voltage = load_cv(curr_path, two_way)
                    data[(s, e)] = {"current": current, "voltage": voltage}
                else:
                    continue
    if c_dark is not None:
        for s,e in data:
            data[(s, e)]["current"] -= c_dark
    return data

def get_calibration_matrix(calib_path, spectra_path, two_way=False, boxcar_width=11):
    data = load_calibration_data(calib_path, '.txt', two_way)
    spectra = {(s, e): pd.read_csv(f'{spectra_path}/{s}-{e}.txt', sep='\t', header=0) for s, e in data}
    calib_matrix = []
    wlens = []
    for s,e in data:
        wlens.append((s+e)//2)
        c, v = data[(s, e)]["current"], data[(s, e)]["voltage"]
        wl, pc = spectra[(s, e)]["Wavelength (nm)"].values, spectra[(s, e)]["Counts"].values
        pc = np.convolve(pc, np.ones(boxcar_width)/boxcar_width, mode='valid')
        pc -= pc[100:-100].min()
        shift = boxcar_width // 2
        wl = wl[shift:len(wl)-shift]
        m = (wl >= s) & (wl <= e)
        wl, pc = wl[m], pc[m]
        bins = np.arange(s, e + 1, 1)
        pc_mean, bin_edges, bin_number = binned_statistic(wl, pc, statistic='mean', bins=bins)
        # calib_matrix.append(c.reshape(-1,1) / pc_mean)
        calib_matrix.append(c / pc.mean())
    calib_matrix = np.array(calib_matrix)
    v_idx = np.arange(0, calib_matrix.shape[1], 9)
    plt.imshow(calib_matrix, aspect='auto', cmap='afmhot')
    plt.colorbar(label='photoresponsivity [A W]')
    plt.yticks(np.arange(len(wlens))[::2], wlens[::2])
    plt.xticks(v_idx, np.round(v[v_idx]).astype(int))
    plt.xlabel("Voltage [V]")
    plt.ylabel("Wavelength [nm]")
    plt.show()
    return np.array(calib_matrix)
    # return np.transpose(np.array(calib_matrix), (0, 2, 1)).reshape(-1, len(v))

calib_path = '20nm/20nmInAsNW_QD_A_05/calibration_10nm'
spectra_path = 'spectrometer_data/calibration_data_10nm'


pcs = []
calib_matrix = get_calibration_matrix(calib_path, spectra_path, False)

np.savetxt("CalibrationData.txt", calib_matrix)
npa = lambda x: np.array(x)
