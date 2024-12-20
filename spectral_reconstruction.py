import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep
from scipy.optimize import minimize
from scipy.linalg import svd
from scipy.optimize import lsq_linear
import os

plt.rcParams.update({'image.cmap': 'inferno'})
print(os.getcwd())
plot = True

# **************Parameter settings*************
NumWavelengths = 27  # Number of wavelengths for reconstruction (calibration measurements)
NumWavelengthsInterp = NumWavelengths * 10  # Number of interpolated wavelengths for plotting
NumGaussianBasis = NumWavelengths   # Number of Gaussian Basis
MinLambda = 450  # The smallest wavelength in nm
MaxLambda = 720  # The largest wavelength in nm

s, e = 550, 600   # Tested wavelengths (name of the file)
FWHMset = (e - s) * 1.5  # Predicted FWHM
boxcar_width = 1  # Averaging of the spectral data using boxcar width

# **************Loading data*************
ResponseCurves = np.loadtxt('CalibrationData.txt').T  # Responsivity as function of wavelength
ResponseCurves /= np.max(ResponseCurves)  # Normalize the input response curve
print(ResponseCurves.shape)
dc_path = '20nm/20nmInAsNW_QD_A_05/' \
         'calibration_10nm/241210_20nm_QD05_100mVsd_Dark_ID1-1.txt'

meas_path = '20nm/20nmInAsNW_QD_A_05/reconstruction_data'

def load_cv(file_path, two_way=False):
    df = pd.read_csv(file_path, header=0, sep='\t')
    # df.dropna(inplace=True)
    cols = df.columns
    smu1c = df[cols[-3]].values[2:-1].astype(float)
    smu2v = df[cols[-2]].values[2:-1].astype(float)
    if two_way:
        return smu1c, smu2v
    else:
        idx_end = np.where(smu2v == smu2v.max())[0][0] + 1
    return smu1c[:idx_end], smu2v[:idx_end]


def load_range(s, e, meas_path, dc_path, two_way=False):
    if s < MinLambda or e > MaxLambda:
        print(f"Invalid range")
        return None
    fpath = f'{meas_path}/241210_20nm_QD05_100mVsd_{s}-{e}_ID1-1.txt'
    dc_current, dc_voltage = load_cv(dc_path, two_way)
    current, voltage = load_cv(fpath, two_way)
    if len(dc_current) != len(current):
        print(f"Length mismatch: {len(dc_current)=}, {len(current)=}")
        valid_len = min(len(dc_current), len(current))
        return current[:valid_len] - dc_current[:valid_len], voltage[:valid_len]
    return current - dc_current, voltage

c, v = load_range(s, e, meas_path, dc_path, two_way=False)

if plot:
    fig = plt.figure(figsize=(12,4))
    fig.add_subplot(121)
    plt.imshow(ResponseCurves, aspect='auto')
    plt.title("Response curves")
    plt.colorbar()
    fig.add_subplot(122)
    plt.plot(c)
    plt.grid()
    plt.title("Measured signals")
    plt.show()


# MeasuredSignals = np.loadtxt('MeausuredSignals.txt')  # Photocurrent as function of Vg
MeasuredSignals = np.array([c]).T
# **************Initialization of variables*************
nrows = ResponseCurves.shape[0]  # Number of Vg's (Detectors)
# ncols = MeasuredSignals.shape[1]  # Multiple data processing from matrix of MeasuredSignals
ncols = 1
print(f"{nrows=}, {ncols=}")

SimulatedSignals = np.zeros((nrows, ncols))
HiResReconstructedSpectrum = np.zeros((NumWavelengthsInterp, ncols))

wlens = np.linspace(MinLambda, MaxLambda, NumWavelengths) / MaxLambda
wlensPlot = np.linspace(MinLambda, MaxLambda, NumWavelengthsInterp) / MaxLambda
hr_response_curves = np.zeros((nrows, NumWavelengthsInterp))
for r in range(nrows):
    spline_rep = splrep(wlens, ResponseCurves[r, :])
    hr_response_curves[r, :] = splev(wlensPlot, spline_rep)

MeasuResidual = np.zeros(ncols)
ReguTerm = np.zeros(ncols)

plt.imshow(hr_response_curves, aspect='auto')
plt.colorbar()
plt.title("High resolution response curves")
plt.show()

# **************Reconstruction*************
# def find_opt_gamma_gcv(c, A):
#     U, sigma, _ = svd(A)
#     print(f"{A.shape=}, {U.shape=}, {sigma.shape=}, {c.shape=}")
#     def compute_gcv(gamma):
#         fi = sigma ** 2 / (sigma ** 2 + gamma ** 2)
#         rho = np.sum(((1 - fi) * (U.T @ c)) ** 2)
#         return rho / (len(c) - np.sum(fi)) ** 2
#     result = minimize(compute_gcv, x0=1e-6, bounds=[(0, None)])
#     return result.x[0]

def find_opt_gamma_gcv(c, A):
    # Perform SVD: A is (m x n)
    # U: (m x m), sigma: (min(m,n),), Vt: (n x n)
    U, sigma, Vt = np.linalg.svd(A, full_matrices=True)
    m = A.shape[0]  # number of rows in A

    # Initial guess for Gamma (lambda)
    initial_guess = 1e-6

    # Set up the optimization problem using scipy's minimize
    # We want Gamma >= 0, so we impose a bound.
    bounds = [(0, None)]

    # Minimization of compute_gcv with respect to Gamma
    res = minimize(lambda Gamma: compute_gcv(Gamma, U, sigma, c, m),
                   initial_guess, bounds=bounds, options={'disp': False})

    Gamma_opt = res.x[0]
    # Sanity check - should be scalar
    assert np.isscalar(Gamma_opt)
    return Gamma_opt


def compute_gcv(Gamma, U, sigma, b, m):
    # sigma is a vector of singular values (length = numOfcol)
    # Compute the filter factors fi (same length as sigma)
    # fi_i = sigma_i^2 / (sigma_i^2 + Gamma^2)
    fi = (sigma ** 2) / (sigma ** 2 + Gamma ** 2)
    rho = 0.0
    numOfcol = len(sigma)
    for i in range(numOfcol):
        beta = np.dot(U[:, i], b)
        rho += ((1 - fi[i]) * beta) ** 2
    # GCV metric: G = rho / (m - sum(fi))^2
    G = rho / (m - np.sum(fi)) ** 2
    return G


print(MeasuredSignals.shape)
for i in range(ncols):
    MeasuredSignals[:, i] /= np.max(MeasuredSignals[:, i])

    GaussianCenter = np.linspace(MinLambda, MaxLambda, NumGaussianBasis) / MaxLambda
    GaussianSigma = FWHMset / 1000 / (2 * np.sqrt(2 * np.log(2)))

    HiResGaussianBasis = np.zeros((NumWavelengthsInterp, NumGaussianBasis))
    for j in range(NumGaussianBasis):
        HiResGaussianBasis[:, j] = np.exp(-0.5 * ((wlensPlot - GaussianCenter[j]) / GaussianSigma) ** 2)
        HiResGaussianBasis[:, j] /= (GaussianSigma * np.sqrt(2 * np.pi))

    LaplacianMatrix = np.eye(NumGaussianBasis)
    WeightMatrix = hr_response_curves @ HiResGaussianBasis

    print(MeasuredSignals[:, i].shape, WeightMatrix.shape)
    OptimalGamma = find_opt_gamma_gcv(MeasuredSignals[:, i], WeightMatrix)

    AugWeightMatrix = np.vstack((WeightMatrix, OptimalGamma ** 2 * LaplacianMatrix))
    AugMeasuredSignals = np.concatenate((MeasuredSignals[:, i], np.zeros(NumGaussianBasis)))

    lsq_result = lsq_linear(AugWeightMatrix, AugMeasuredSignals, bounds=(0, np.inf))
    GaussianCoefficients = lsq_result.x

    HiResReconstructedSpectrum[:, i] = HiResGaussianBasis @ GaussianCoefficients
    HiResReconstructedSpectrum[:, i] /= np.max(HiResReconstructedSpectrum[:, i])
    SimulatedSignals[:, i] = hr_response_curves @ HiResGaussianBasis @ GaussianCoefficients

    MeasuResidual[i] = np.linalg.norm(SimulatedSignals[:, i] - MeasuredSignals[:, i]) ** 2
    ReguTerm[i] = np.linalg.norm(LaplacianMatrix @ GaussianCoefficients) ** 2


n = lambda x: (x - x.min()) / (x-x.min()).max()
# Data visualization
'''
plt.figure(figsize=(12, 9))
plt.subplot(1, 2, 1)
plt.plot(SimulatedSignals, '*')
plt.xlabel('Vg index')
plt.ylabel('a.u.')
plt.title('Simulated signals')
'''


reference_data = pd.read_csv(f'spectrometer_data/reconstruction_data/{s}-{e}.txt', sep='\t', header=0)
#reference_data = pd.read_csv(f'spectrometer_data/calibration_data_10nm/{s}-{e}.txt', sep='\t', header=0)
wlens_ref = reference_data['Wavelength (nm)'].values
counts_ref = reference_data['Counts'].values
counts_ref = np.convolve(counts_ref, np.ones(boxcar_width)/boxcar_width, mode='valid')
shift = boxcar_width // 2
wlens_ref = wlens_ref[shift:len(wlens_ref)-shift]
m = (wlens_ref >= MinLambda) & (wlens_ref <= MaxLambda)
wlens_ref, counts_ref = wlens_ref[m], counts_ref[m]
counts_ref = n(counts_ref)
plt.plot(wlens_ref, counts_ref, label = 'original')

'''
plt.grid()
plt.title('Reference spectrum') # Reference spectrum
plt.subplot(1, 2, 2)
'''

plt.plot(wlensPlot * MaxLambda, HiResReconstructedSpectrum, 'ro', markersize=1, label = 'reconstructed')
#plt.title('Reconstructed spectrum')
plt.legend()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized intensity [a.u.]')
plt.xlim([MinLambda, MaxLambda])
plt.show()