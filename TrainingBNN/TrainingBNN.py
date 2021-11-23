import torch
import hamiltorch

class LearningFrameworkBNN:

    def __init__(self, bnn, params_init, train_observed_x, train_observed_y, model_save_path):

        self.bnn = bnn
        self.params_init = params_init

        self.train_observed_x = train_observed_x
        self.train_observed_y = train_observed_y

        self.model_save_path = model_save_path


    def trainBNN(self, sigma_prior = 1, sigma_noise = 0.01, step_size = 0.0005, num_samples = 1000, L = 30, burn = -1, model_loss = 'regression',  mass = 1.0, random_seed = 1):

        hamiltorch.set_random_seed(random_seed)

        tau = 1 / sigma_prior
        tau_out = 1 / sigma_noise

        tau_list = []
        for _ in self.bnn.parameters():
            tau_list.append(tau)
        tau_list = torch.tensor(tau_list)

        self.bnn.load_state_dict(self.params_init)

        params_init = hamiltorch.util.flatten(self.bnn).clone()
        inv_mass = torch.ones(params_init.shape) / mass

        sampler = hamiltorch.Sampler.HMC
        integrator = hamiltorch.Integrator.IMPLICIT

        params_hmc = hamiltorch.sample_model(self.bnn, self.train_observed_x, self.train_observed_y, params_init = params_init,
                    model_loss = model_loss, num_samples = num_samples, burn = burn, inv_mass = inv_mass, step_size = step_size,
                    num_steps_per_sample = L, tau_out = tau_out, tau_list = tau_list, sampler = sampler, integrator = integrator)

        torch.save(self.bnn.state_dict(),  self.model_save_path + '/bnn_sigma_prior' + str(sigma_prior) + '.pt')
        torch.save(params_hmc, self.model_save_path + '/params_hmc.pt')