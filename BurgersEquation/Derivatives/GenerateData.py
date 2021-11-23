from BayesianDeepLearningPDE.GenerateDerivatives.GenerateDerivatives import DerivativeData
import torch
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import warnings

warnings.filterwarnings("ignore")

torch.manual_seed(1)
np.random.seed(1)

x_min, x_max = -8, 8
t_max = 10

class DeepNeuralNetwork(torch.nn.Module):
    def __init__(self, n_hidden_units):
        super(DeepNeuralNetwork, self).__init__()

        self.fc1 = torch.nn.Linear(2, n_hidden_units)
        self.fc2 = torch.nn.Linear(n_hidden_units, n_hidden_units)
        self.fc3 = torch.nn.Linear(n_hidden_units, n_hidden_units)
        self.fc4 = torch.nn.Linear(n_hidden_units, n_hidden_units)
        self.fc5 = torch.nn.Linear(n_hidden_units, n_hidden_units)
        self.fc6 = torch.nn.Linear(n_hidden_units, 1)

    def forward(self, x):

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = self.fc6(x)

        return x

sigma_prior = 1
sigma_noise = 0.01
n_hidden_units = 50
lr = '0.0002'
n_iter = '30000'
model_loss = 'regression'
n_derivatives = 10000


def generate(noise):

    bnn = DeepNeuralNetwork(n_hidden_units)
    bnn_state_dict = torch.load('/BayesianDeepLearningPDE/BurgersEquation/SavedModels/BNN_Noise' + noise + '/bnn_sigma_prior' + str(sigma_prior) + '.pt')
    bnn.load_state_dict(bnn_state_dict)

    params_hmc = torch.load('/BayesianDeepLearningPDE/BurgersEquation/SavedModels/BNN_Noise' + noise + '/params_hmc.pt')
    params_hmc = params_hmc[200:6000]

    dnn = DeepNeuralNetwork(n_hidden_units)
    dnn_state_dict = torch.load('/BayesianDeepLearningPDE/BurgersEquation/SavedModels/DNN_Noise' + noise + '/dnn_lr_' + lr + '_iter_' + n_iter + '.pt')

    mx = np.load('/BayesianDeepLearningPDE/HMCInference/BurgersEquation/SavedModels/BNN_Noise' + noise + '/mx.npy')
    sx = np.load('/BayesianDeepLearningPDE/HMCInference/BurgersEquation/SavedModels/BNN_Noise' + noise + '/sx.npy')

    generator = DerivativeData(x_min, x_max, t_max, mx, sx, bnn, dnn, bnn_state_dict = bnn_state_dict, params_hmc = params_hmc, dnn_state_dict = dnn_state_dict,
                     model_loss = model_loss, sigma_prior = sigma_prior, sigma_noise = sigma_noise, random_seed = 1)

    coord = np.hstack((t_max * np.random.rand(n_derivatives, 1), x_min + (x_max - x_min) * np.random.rand(n_derivatives, 1)))

    Derivatives_bnn = generator.get_derivatives_bnn(n_derivatives, coord = coord)
    np.save('/BayesianDeepLearningPDE/BurgersEquation/Derivatives/DerivativesData/BNN_Noise' + noise + '/Derivatives_BNN_' + str(generator.n_hmc_samples) + 's_' + str(n_derivatives) + 'pt.npy', Derivatives_bnn)
    np.save('/BayesianDeepLearningPDE/BurgersEquation/Derivatives/DerivativesData/BNN_Noise' + noise + '/Derivatives_Coordinates_' + str(generator.n_hmc_samples) + 's_' + str(n_derivatives) + 'pt.npy', coord)

    print('Noise Case: ' + noise + ', BNN Derivatives: DONE')

    Derivatives_dnn = generator.get_derivatives_dnn(n_derivatives, coord = coord)
    np.save('/BayesianDeepLearningPDE/BurgersEquation/Derivatives/DerivativesData/DNN_Noise' + noise + '/Derivatives_DNN_' + str(n_derivatives) + 'pt.npy', Derivatives_dnn)
    np.save('/BayesianDeepLearningPDE/BurgersEquation/Derivatives/DerivativesData/DNN_Noise' + noise + '/Derivatives_Coordinates_' + str(n_derivatives) + 'pt.npy', coord)

    print('Noise Case: ' + noise + ', DNN Derivatives: DONE')

NoiseCase = ['0.00', '0.01', '0.05']

for noise in NoiseCase:
    generate(noise)