import torch
import numpy as np
from BayesianDeepLearningPDE.TrainingBNN.TrainingBNN import LearningFrameworkBNN
from BayesianDeepLearningPDE.TrainingDNN.TrainingDNN import LearningFrameworkDNN
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import warnings

warnings.filterwarnings("ignore")

torch.manual_seed(1)
np.random.seed(1)


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
lr = 2e-4
n_iter = 30000

def train_bnns(noise):

    domain_data = np.load('/BayesianDeepLearningPDE/BurgersEquation/Data/RandomSensorsDataPointsNoise' + noise + '.npy', allow_pickle = True).item()
    model_save_path_bnn = '/BayesianDeepLearningPDE/BurgersEquation/SavedModels/BNN_Noise' + noise + '/'

    params_init = torch.load('/BayesianDeepLearningPDE/BurgersEquation/SavedModels/DNN_Noise' + noise + '/dnn_lr_0.0002_iter_30000.pt')

    train_observed_x = domain_data['X']
    train_observed_x = torch.Tensor(train_observed_x)
    train_observed_y_bnn = domain_data['y'].reshape(-1, 1)
    train_observed_y_bnn = torch.Tensor(train_observed_y_bnn)

    mx = train_observed_x.mean(0)
    sx = train_observed_x.std(0)

    train_observed_x = (train_observed_x - mx) / sx

    np.save('/BayesianDeepLearningPDE/BurgersEquation/SavedModels/BNN_Noise' + noise + '/mx.npy', mx)
    np.save('/BayesianDeepLearningPDE/BurgersEquation/SavedModels/BNN_Noise' + noise + '/sx.npy', sx)

    bnn = DeepNeuralNetwork(n_hidden_units = n_hidden_units)
    learningBNN = LearningFrameworkBNN(bnn, params_init, train_observed_x, train_observed_y_bnn, model_save_path_bnn)
    learningBNN.trainBNN(sigma_prior = sigma_prior, sigma_noise = sigma_noise, step_size = 0.0005, num_samples = 6000)

    print('Noise Case: ' + noise + ', Training BNN: DONE')



def train_dnns(noise):

    domain_data = np.load('/BayesianDeepLearningPDE/BurgersEquation/Data/RandomSensorsDataPointsNoise' + noise + '.npy').item()
    model_save_path_dnn = '/BayesianDeepLearningPDE/BurgersEquation/SavedModels/DNN_Noise' + noise + '/'
    checkpoint_path = '/BayesianDeepLearningPDE/BurgersEquation/Checkpoint/'

    train_observed_x = domain_data['X']
    train_observed_x = torch.Tensor(train_observed_x)
    train_observed_y_dnn = domain_data['y']
    train_observed_y_dnn = torch.Tensor(train_observed_y_dnn)

    mx = train_observed_x.mean(0)
    sx = train_observed_x.std(0)

    train_observed_x = (train_observed_x - mx) / sx

    np.save('/BayesianDeepLearningPDE/BurgersEquation/SavedModels/DNN_Noise' + noise + '/mx.npy', mx)
    np.save('/BayesianDeepLearningPDE/BurgersEquation/SavedModels/DNN_Noise' + noise + '/sx.npy', sx)

    dnn = DeepNeuralNetwork(n_hidden_units = n_hidden_units)
    learningDNN = LearningFrameworkDNN(dnn, train_observed_x, train_observed_y_dnn, checkpoint_path, model_save_path_dnn)
    learningDNN.trainDNN(lr = lr, n_iter = n_iter)

    print('Noise Case: ' + noise + ', Training DNN: DONE')


NoiseCase = ['0.00', '0.01', '0.05']

for noise in NoiseCase:
    train_dnns(noise)
    train_bnns(noise)