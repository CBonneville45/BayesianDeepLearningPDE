import torch
import matplotlib.pyplot as plt
import numpy as np

class LearningFrameworkDNN:

    def __init__(self, dnn, train_observed_x, train_observed_y, checkpoint_path, model_save_path):

        self.dnn = dnn

        self.train_observed_x = train_observed_x
        self.train_observed_y = train_observed_y

        self.checkpoint_path = checkpoint_path
        self.model_save_path = model_save_path


    def _mse(self, nn_output_observed):

        loss_f = torch.nn.MSELoss()
        loss = loss_f(nn_output_observed, self.train_observed_y)

        return loss


    def _plotLoss(self, iter, n_iter, Loss = [], fig = None):

        if fig is None:

            fig = plt.figure()
            plt.xlim(1, n_iter)
            plt.ylabel('Loss')
            plt.xlabel('Iteration')

            return fig

        else:

            plt.clf()
            plt.plot(np.arange(1, iter + 2), Loss)
            plt.yscale('log')
            plt.xlim(1, n_iter)
            plt.ylabel('Loss')
            plt.xlabel('Iteration')
            plt.show(block = False)
            fig.canvas.draw()
            plt.pause(1e-5)

            return fig


    def trainDNN(self, lr = 2e-4, n_iter = 30000, print_loss = False, plot_loss = False):

        self.dnn.train()

        optimizer = torch.optim.Adam(self.dnn.parameters(), lr = lr)

        Loss = []
        fig = None
        loss_mse_best = 0

        for iter in range(n_iter):

            optimizer.zero_grad()

            nn_output_observed = self.dnn(self.train_observed_x).reshape(self.train_observed_y.shape)
            loss = LearningFrameworkDNN._mse(self, nn_output_observed)

            loss.backward()
            optimizer.step()

            if print_loss:
                print('Iteration: %d/%d, Loss: %3.5f' % (iter, n_iter, loss.item()))

            Loss.append(loss.item())

            if iter % 2000 == 0 and plot_loss is True:
                fig = LearningFrameworkDNN._plotLoss(self, iter, n_iter, Loss, fig)

            if iter == 0:
                loss_mse_best = loss.item()
                torch.save(self.dnn.state_dict(), self.checkpoint_path + 'model.pt')

            if iter > 0 and loss.item() < loss_mse_best:
                torch.save(self.dnn.state_dict(), self.checkpoint_path + 'model.pt')
                loss_mse_best = loss.item()

        self.dnn.load_state_dict(torch.load(self.checkpoint_path + 'model.pt'))
        torch.save(self.dnn.state_dict(),  self.model_save_path + '/dnn_lr_' + str(lr) + '_iter_' + str(n_iter) + '.pt')