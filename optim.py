from collections.abc import ValuesView
import torch


class Adam:
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.moment_1 = {
            key: torch.zeros_like(value) for key, value in parameters.items()
        }
        self.moment_2 = {
            key: torch.zeros_like(value) for key, value in parameters.items()
        }
        self.lr = lr
        self.betas = betas
        self.eps = eps

    def calc_grad(self, grad):
        grad_dict = {}
        for key, value in grad.items():
            self.moment_1[key] = (
                self.betas[0] * self.moment_1[key] + (1 - self.betas[0]) * value
            )
            self.moment_2[key] = (
                self.betas[1] * self.moment_2[key] + (1 - self.betas[1]) * value**2
            )
            m_1_hat = self.moment_1[key] / (1 - self.betas[0])
            m_2_hat = self.moment_2[key] / (1 - self.betas[1])
            grad_dict.update(
                {key: self.lr * m_1_hat / (torch.sqrt(m_2_hat) + self.eps)}
            )
        return grad_dict
