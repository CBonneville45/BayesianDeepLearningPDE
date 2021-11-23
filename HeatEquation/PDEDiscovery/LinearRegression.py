import numpy as np
import os
from sklearn.linear_model import BayesianRidge, LinearRegression

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.random.seed(1)

def uncertainty_normalization(X_std, y_std = None):

    sample_weight = X_std ** 2
    sample_weight /= sample_weight.max(0)
    sample_weight = sample_weight.sum(1)
    if y_std is not None:
        sample_weight += y_std / y_std.max()
    sample_weight = 1 / sample_weight

    return sample_weight


def threshold_regression(derivatives, n_train, X_bnn, y_bnn, X_std, y_std, threshold, adaptive):

    X_bnn_i = X_bnn[:n_train, derivatives]
    X_std_i = X_std[:n_train, derivatives]
    y_bnn = y_bnn[:n_train]

    sample_weight_i = uncertainty_normalization(X_std_i, y_std)

    coef_i = BayesianRidge(fit_intercept = False).fit(X_bnn_i, y_bnn, sample_weight_i).coef_

    delete_indices = np.where(np.abs(coef_i) < threshold)[0]

    while delete_indices.shape[0] > 0:

        if adaptive:
            threshold *= 2

        X_bnn_i = np.delete(X_bnn_i, delete_indices, 1)
        X_std_i = np.delete(X_std_i, delete_indices, 1)

        sample_weight_i = uncertainty_normalization(X_std_i)

        coef_i = BayesianRidge(fit_intercept = False).fit(X_bnn_i, y_bnn, sample_weight_i).coef_

        delete_indices = np.where(np.abs(coef_i) < threshold)[0]

    n_kept_derivatives = X_bnn_i.shape[1]
    kept_derivatives = []

    for i in range(n_kept_derivatives):
        kept_derivatives.append(np.where(X_bnn[0, :] == X_bnn_i[0, i])[0][0])

    sigma_i = BayesianRidge(fit_intercept = False).fit(X_bnn_i, y_bnn, sample_weight_i).sigma_

    return coef_i, sigma_i, kept_derivatives


def threshold_regression_no_weighting(derivatives, n_train, X_bnn, y_bnn, threshold, adaptive):

    X_bnn_i = X_bnn[:n_train, derivatives]
    y_bnn = y_bnn[:n_train]

    coef_i = LinearRegression(fit_intercept = False).fit(X_bnn_i, y_bnn).coef_

    delete_indices = np.where(np.abs(coef_i) < threshold)[0]

    while delete_indices.shape[0] > 0:

        if adaptive:
            threshold *= 2

        X_bnn_i = np.delete(X_bnn_i, delete_indices, 1)

        coef_i = LinearRegression(fit_intercept = False).fit(X_bnn_i, y_bnn).coef_

        delete_indices = np.where(np.abs(coef_i) < threshold)[0]

    n_kept_derivatives = X_bnn_i.shape[1]
    kept_derivatives = []

    for i in range(n_kept_derivatives):
        kept_derivatives.append(np.where(X_bnn[0, :] == X_bnn_i[0, i])[0][0])

    return coef_i, kept_derivatives



True_coef = np.array([0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0])
NoiseCases = ['0.00', '0.01', '0.05']
derivatives = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
n_train = 10000
threshold_bnn = 0.02
threshold_dnn = 0.02
adaptive_bnn = True
adaptive_dnn = False

coef_bnn_000 = np.zeros(True_coef.shape[0])
coef_bnn_001 = np.zeros(True_coef.shape[0])
coef_bnn_005 = np.zeros(True_coef.shape[0])

coef_dnn_000 = np.zeros(True_coef.shape[0])
coef_dnn_001 = np.zeros(True_coef.shape[0])
coef_dnn_005 = np.zeros(True_coef.shape[0])

for noise in NoiseCases:

    data_bnn = np.load('/BayesianDeepLearningPDE/HeatEquation/Derivatives/DerivativesData/BNN_Noise' + noise + '/Derivatives_BNN_5800s_10000pt.npy', allow_pickle = True).item()
    data_dnn = np.load('/BayesianDeepLearningPDE/HeatEquation/Derivatives/DerivativesData/DNN_Noise' + noise + '/Derivatives_DNN_10000pt.npy', allow_pickle = True).item()

    X_bnn, y_bnn, X_std = data_bnn['X'], data_bnn['y'], data_bnn['X_std']
    X_dnn, y_dnn = data_dnn['X'], data_dnn['y']

    X_bnn = X_bnn[:n_train, derivatives]
    X_std = X_std[:n_train, derivatives]
    y_bnn = y_bnn[:n_train]
    y_std = None

    coef_bnn, _, index_bnn = threshold_regression(derivatives, n_train, X_bnn, y_bnn, X_std, y_std, threshold_bnn, adaptive_bnn)
    coef_dnn, index_dnn = threshold_regression_no_weighting(derivatives, n_train, X_dnn, y_dnn, threshold_dnn, adaptive_dnn)

    if noise == '0.00':
        coef_bnn_000[index_bnn] = coef_bnn
        coef_dnn_000[index_dnn] = coef_dnn
    elif noise == '0.01':
        coef_bnn_001[index_bnn] = coef_bnn
        coef_dnn_001[index_dnn] = coef_dnn
    else:
        coef_bnn_005[index_bnn] = coef_bnn
        coef_dnn_005[index_dnn] = coef_dnn

candidates = [' $u$ ', ' $u_{x}$ ', ' $u_{xx}$ ', ' $u_{xxx}$ ', ' $u_{xxxx}$ ', ' $uu_{x}$ ', ' $uu_{xx}$ ', ' $uu_{xxx}$ ', ' $uu_{xxxx}$ ', ' $u^2$ ', ' $u^2_{x}$ ']

print(r'Candidates & Ground True & Noiseless  & $\epsilon\sim\mathcal{N}(0,0.01^2)$ & $\epsilon\sim\mathcal{N}(0,0.05^2)$ \\\hline')
for i in range(derivatives.shape[0] - 1):
    print(candidates[i] + '& $' + str(True_coef[i]) + '$ & $' + str(round(coef_bnn_000[i], 4)) + '$ & $' + str(round(coef_bnn_001[i], 4)) + '$ & $' + str(round(coef_bnn_005[i], 4)) + r'$ \\[-5pt] ')
print(candidates[-1] + '& $' + str(True_coef[-1]) + '$ & $' + str(round(coef_bnn_000[-1], 4)) + '$ & $' + str(round(coef_bnn_001[-1], 4)) + '$ & $' + str(round(coef_bnn_005[-1], 4)) + r'$ \\\hline')
print('$e$ ($\ell^2$ norm) & $0.0$ & $' + str(round(np.linalg.norm(coef_bnn_000 - True_coef), 4)) + '$ & $' + str(round(np.linalg.norm(coef_bnn_001 - True_coef), 4)) + '$ & $' + str(round(np.linalg.norm(coef_bnn_005 - True_coef), 4)) + '$')

print('\n\n')
print(r'Candidates & Ground True & Noiseless  & $\epsilon\sim\mathcal{N}(0,0.01^2)$ & $\epsilon\sim\mathcal{N}(0,0.05^2)$ \\\hline')
for i in range(derivatives.shape[0] - 1):
    print(candidates[i] + '& $' + str(True_coef[i]) + '$ & $' + str(round(coef_dnn_000[i], 4)) + '$ & $' + str(round(coef_dnn_001[i], 4)) + '$ & $' + str(round(coef_dnn_005[i], 4)) + r'$ \\[-5pt] ')
print(candidates[-1] + '& $' + str(True_coef[-1]) + '$ & $' + str(round(coef_dnn_000[-1], 4)) + '$ & $' + str(round(coef_dnn_001[-1], 4)) + '$ & $' + str(round(coef_dnn_005[-1], 4)) + r'$ \\\hline')
print('$e$ ($\ell^2$ norm) & $0.0$ & $' + str(round(np.linalg.norm(coef_dnn_000 - True_coef), 4)) + '$ & $' + str(round(np.linalg.norm(coef_dnn_001 - True_coef), 4)) + '$ & $' + str(round(np.linalg.norm(coef_dnn_005 - True_coef), 4)) + '$')