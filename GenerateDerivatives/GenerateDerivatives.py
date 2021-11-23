import hamiltorch
import torch
import numpy as np

dtype = torch.double

class DerivativeData:

    def __init__(self, x_min, x_max, t_max, mx = 0, sx = 1, bnn = None, dnn = None, bnn_state_dict = None, params_hmc = None, dnn_state_dict = None,
                 model_loss = None, sigma_prior = None, sigma_noise = None, random_seed = None):

        self.x_min = x_min
        self.x_max = x_max
        self.t_max = t_max

        self.mx = torch.Tensor(mx)
        self.sx = torch.Tensor(sx)

        self.bnn = bnn
        self.dnn = dnn

        self.bnn.load_state_dict(bnn_state_dict)
        self.dnn.load_state_dict(dnn_state_dict)

        self.bnn_state_dict = bnn_state_dict
        self.dnn_state_dict = dnn_state_dict
        self.params_hmc = params_hmc
        self.n_hmc_samples = len(params_hmc)
        self.params_keys = list(bnn.state_dict().keys())

        self.model_loss = model_loss

        tau_prior = 1 / sigma_prior
        tau_noise = 1 / sigma_noise

        tau_list = []
        for _ in bnn.parameters():
            tau_list.append(tau_prior)

        tau_list = torch.tensor(tau_list)

        self.tau_noise = tau_noise
        self.tau_list = tau_list

        self.bounds = (torch.tensor([[0, self.x_min], [self.t_max, self.x_max]], dtype = dtype) - mx) / sx

        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            hamiltorch.set_random_seed(random_seed)



    def get_derivatives_bnn(self, n_derivatives, coord):

        bnn_input = torch.Tensor(coord)
        bnn_input.requires_grad = True

        dudt_mat = np.zeros([n_derivatives, self.n_hmc_samples])
        dudx_mat = np.zeros([n_derivatives, self.n_hmc_samples])
        dudx2_mat = np.zeros([n_derivatives, self.n_hmc_samples])
        dudx3_mat = np.zeros([n_derivatives, self.n_hmc_samples])
        dudx4_mat = np.zeros([n_derivatives, self.n_hmc_samples])
        ududx_mat = np.zeros([n_derivatives, self.n_hmc_samples])
        ududx2_mat = np.zeros([n_derivatives, self.n_hmc_samples])
        ududx3_mat = np.zeros([n_derivatives, self.n_hmc_samples])
        ududx4_mat = np.zeros([n_derivatives, self.n_hmc_samples])
        u_sq_mat = np.zeros([n_derivatives, self.n_hmc_samples])
        dudx_sq_mat = np.zeros([n_derivatives, self.n_hmc_samples])
        u_mat = np.zeros([n_derivatives, self.n_hmc_samples])

        for i in range(self.n_hmc_samples):

            params_hmc_unflat = hamiltorch.util.unflatten(self.bnn, self.params_hmc[i])
            state_dict = self.bnn.state_dict()

            w = 0
            for key in self.params_keys:
                state_dict[key] = torch.nn.Parameter(params_hmc_unflat[w])
                w += 1

            self.bnn.load_state_dict(state_dict)
            u = self.bnn((bnn_input - self.mx) / self.sx)

            grad = torch.autograd.grad(u, bnn_input, grad_outputs = torch.ones_like(u), create_graph = True, retain_graph = True, allow_unused = True)[0]
            dudt = grad[:, 0]
            dudx = grad[:, 1]
            dudx2 = torch.autograd.grad(dudx, bnn_input, grad_outputs = torch.ones_like(dudx), create_graph = True, retain_graph = True, allow_unused = True)[0][:, 1]
            dudx3 = torch.autograd.grad(dudx2, bnn_input, grad_outputs = torch.ones_like(dudx2), create_graph = True, retain_graph = True, allow_unused = True)[0][:, 1]
            dudx4 = torch.autograd.grad(dudx3, bnn_input, grad_outputs = torch.ones_like(dudx3), create_graph = True, retain_graph = True, allow_unused = True)[0][:, 1]

            dudt_mat[:, i] = dudt.detach().numpy()
            dudx_mat[:, i] = dudx.detach().numpy()
            dudx2_mat[:, i] = dudx2.detach().numpy()
            dudx3_mat[:, i] = dudx3.detach().numpy()
            dudx4_mat[:, i] = dudx4.detach().numpy()
            ududx_mat[:, i] = (u[:, 0] * dudx).detach().numpy()
            ududx2_mat[:, i] = (u[:, 0] * dudx2).detach().numpy()
            ududx3_mat[:, i] = (u[:, 0] * dudx3).detach().numpy()
            ududx4_mat[:, i] = (u[:, 0] * dudx4).detach().numpy()
            u_mat[:, i] = u[:, 0].detach().numpy()
            u_sq_mat[:, i] = u[:, 0].detach().numpy() ** 2
            dudx_sq_mat[:, i] = dudx.detach().numpy() ** 2

        dudx_mean = dudx_mat.mean(1).reshape(-1, 1)
        dudx2_mean = dudx2_mat.mean(1).reshape(-1, 1)
        dudx3_mean = dudx3_mat.mean(1).reshape(-1, 1)
        dudx4_mean = dudx4_mat.mean(1).reshape(-1, 1)
        ududx_mean = ududx_mat.mean(1).reshape(-1, 1)
        ududx2_mean = ududx2_mat.mean(1).reshape(-1, 1)
        ududx3_mean = ududx3_mat.mean(1).reshape(-1, 1)
        ududx4_mean = ududx4_mat.mean(1).reshape(-1, 1)
        u_mean = u_mat.mean(1).reshape(-1, 1)
        u_sq_mean = u_sq_mat.mean(1).reshape(-1, 1)
        dudx_sq_mean = dudx_sq_mat.mean(1).reshape(-1, 1)

        X = np.hstack((u_mean, dudx_mean, dudx2_mean, dudx3_mean, dudx4_mean, ududx_mean, ududx2_mean, ududx3_mean, ududx4_mean, u_sq_mean, dudx_sq_mean))
        y = dudt_mat.mean(1)

        dudx_std = dudx_mat.std(1).reshape(-1, 1)
        dudx2_std = dudx2_mat.std(1).reshape(-1, 1)
        dudx3_std = dudx3_mat.std(1).reshape(-1, 1)
        dudx4_std = dudx4_mat.std(1).reshape(-1, 1)
        ududx_std = ududx_mat.std(1).reshape(-1, 1)
        ududx2_std = ududx2_mat.std(1).reshape(-1, 1)
        ududx3_std = ududx3_mat.std(1).reshape(-1, 1)
        ududx4_std = ududx4_mat.std(1).reshape(-1, 1)
        u_std = u_mat.std(1).reshape(-1, 1)
        u_sq_std = u_sq_mat.std(1).reshape(-1, 1)
        dudx_sq_std = dudx_sq_mat.std(1).reshape(-1, 1)

        X_std = np.hstack((u_std, dudx_std, dudx2_std, dudx3_std, dudx4_std, ududx_std, ududx2_std, ududx3_std,
                           ududx4_std, u_sq_std, dudx_sq_std))
        y_std = dudt_mat.std(1)

        Derivatives = {'X': X, 'y': y, 'X_std': X_std, 'y_std': y_std}

        self.Derivatives_bnn = Derivatives

        return Derivatives



    def get_derivatives_dnn(self, n_derivatives, coord):

        dnn_input = torch.Tensor(coord)
        dnn_input.requires_grad = True

        dudt_mat = np.zeros([n_derivatives, 1])
        dudx_mat = np.zeros([n_derivatives, 1])
        dudx2_mat = np.zeros([n_derivatives, 1])
        dudx3_mat = np.zeros([n_derivatives, 1])
        dudx4_mat = np.zeros([n_derivatives, 1])
        ududx_mat = np.zeros([n_derivatives, 1])
        ududx2_mat = np.zeros([n_derivatives, 1])
        ududx3_mat = np.zeros([n_derivatives, 1])
        ududx4_mat = np.zeros([n_derivatives, 1])
        u_sq_mat = np.zeros([n_derivatives, 1])
        dudx_sq_mat = np.zeros([n_derivatives, 1])
        u_mat = np.zeros([n_derivatives, 1])

        u = self.dnn((dnn_input - self.mx) / self.sx)

        grad = torch.autograd.grad(u, dnn_input, grad_outputs = torch.ones_like(u), create_graph = True, retain_graph = True, allow_unused = True)[0]
        dudt = grad[:, 0]
        dudx = grad[:, 1]
        dudx2 = torch.autograd.grad(dudx, dnn_input, grad_outputs = torch.ones_like(dudx), create_graph = True, retain_graph = True, allow_unused = True)[0][:, 1]
        dudx3 = torch.autograd.grad(dudx2, dnn_input, grad_outputs = torch.ones_like(dudx2), create_graph = True, retain_graph = True, allow_unused = True)[0][:, 1]
        dudx4 = torch.autograd.grad(dudx3, dnn_input, grad_outputs = torch.ones_like(dudx3), create_graph = True, retain_graph = True, allow_unused = True)[0][:, 1]

        dudt_mat[:, 0] = dudt.detach().numpy()
        dudx_mat[:, 0] = dudx.detach().numpy()
        dudx2_mat[:, 0] = dudx2.detach().numpy()
        dudx3_mat[:, 0] = dudx3.detach().numpy()
        dudx4_mat[:, 0] = dudx4.detach().numpy()
        ududx_mat[:, 0] = (u[:, 0] * dudx).detach().numpy()
        ududx2_mat[:, 0] = (u[:, 0] * dudx2).detach().numpy()
        ududx3_mat[:, 0] = (u[:, 0] * dudx3).detach().numpy()
        ududx4_mat[:, 0] = (u[:, 0] * dudx4).detach().numpy()
        u_mat[:, 0] = u[:, 0].detach().numpy()
        u_sq_mat[:, 0] = u[:, 0].detach().numpy() ** 2
        dudx_sq_mat[:, 0] = dudx.detach().numpy() ** 2

        X = np.hstack((u_mat, dudx_mat, dudx2_mat, dudx3_mat, dudx4_mat, ududx_mat, ududx2_mat, ududx3_mat, ududx4_mat, u_sq_mat, dudx_sq_mat))
        Derivatives = {'X': X, 'y': dudt_mat}

        self.Derivatives_dnn = Derivatives

        return Derivatives