import torch
import torch.nn as nn
import torch.nn.functional as F


class RBM(nn.Module):
    def __init__(self, visible_size, hidden_size, batch_size, k_cd=1, device="cpu"):
        super().__init__()
        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.param = {
            "weight": torch.normal(
                0, 0.01, size=[hidden_size, visible_size], device=device
            ),
            "bias_v": torch.zeros([1, visible_size], device=device),
            "bias_h": torch.zeros([1, hidden_size], device=device),
        }
        self.h_state = torch.zeros([batch_size, hidden_size], device=device)
        self.k_cd = k_cd

    def _pv_given_h(self, h):
        return torch.sigmoid(
            F.linear(h, self.param["weight"].t(), self.param["bias_v"])
        )

    def _ph_given_v(self, v):
        return torch.sigmoid(F.linear(v, self.param["weight"], self.param["bias_h"]))

    def _sample_pcd(self, v):
        # persistent constractive divergence(k)法によるサンプリング
        h_pos = self._ph_given_v(v).bernoulli()
        h_neg = self.h_state
        for _ in range(self.k_cd):
            v_neg = self._pv_given_h(h_neg).bernoulli()
            h_neg = self._ph_given_v(v_neg).bernoulli()
        self.h_state = h_neg

        return h_pos, v_neg, h_neg

    def _sample_cd(self, v):
        # constractive divergence(k)法によるサンプリング
        h_pos = self._ph_given_v(v).bernoulli()
        h_neg = h_pos
        for _ in range(self.k_cd):
            v_neg = self._pv_given_h(h_neg).bernoulli()
            h_neg = self._ph_given_v(v_neg).bernoulli()
        self.h_state = h_neg

        return h_pos, v_neg, h_neg

    def _sample_pcd_cont(self, v):
        # persistent constractive divergence(k)法によるサンプリング
        ph_pos = self._ph_given_v(v)

        h_neg = self.h_state
        for _ in range(self.k_cd):
            v_neg = self._pv_given_h(h_neg).bernoulli()
            ph_neg = self._ph_given_v(v_neg)
            h_neg = ph_neg.bernoulli()
        self.h_state = h_neg

        return ph_pos, v_neg, ph_neg

    def grad(self, v, sample_type="pcd_cont"):
        if sample_type == "pcd":
            h_pos, v_neg, h_neg = self._sample_pcd(v)
        elif sample_type == "cd":
            h_pos, v_neg, h_neg = self._sample_cd(v)
        elif sample_type == "pcd_cont":
            h_pos, v_neg, h_neg = self._sample_pcd_cont(v)
        else:
            raise NotImplementedError("choose pcd, cd, or pcd_cont")
        B, _ = v.shape
        w_grad = torch.matmul(h_pos.t(), v) - torch.matmul(h_neg.t(), v_neg)
        w_grad = w_grad / B
        bh_grad = (h_pos - h_neg).mean(dim=0)
        bv_grad = (v - v_neg).mean(dim=0)
        return {"weight": w_grad, "bias_h": bh_grad, "bias_v": bv_grad}

    # def update(self, grad):
    #     param = self.state_dict()
    #     for key, value in grad.items():
    #         param[key] += value

    def update(self, v, lr, sample_type="pcd_cont"):
        grad = self.grad(v, sample_type)
        for key, value in grad.items():
            self.param[key] += lr * value

    def free_energy(self, v):
        # 訓練データによる平均自由エネルギーを計算する
        term1 = -torch.matmul(v, self.param["bias_v"].squeeze(0))
        bias = self.bias_h + torch.matmul(v, self.param["weight"].t())
        term2 = -torch.log(1 + torch.exp(bias) + 1e-10).sum(dim=1)
        return term1 + term2

    def energy(self, v):
        v_term = torch.matmul(v, self.param["bias_v"].t())
        w_x_h = torch.matmul(v, self.param["weight"].t()) + self.param["bias_h"]
        h_term = torch.sum(F.softplus(w_x_h), dim=1)
        return -h_term - v_term

    def pseudo_likelihood(self, v):
        flip = torch.randint(0, v.size(1), (1,))
        v_fliped = v.clone()
        v_fliped[:, flip] = 1 - v_fliped[:, flip]
        energy = self.energy(v)
        energy_fliped = self.energy(v_fliped)
        return v.size(1) * F.softplus(energy_fliped - energy)

    def sample_by_v(self, v, num_gib=300):
        # batchを受け取り、データを再構成する
        h = self._ph_given_v(v).bernoulli()
        for _ in range(num_gib):
            pv_gibb = self._pv_given_h(h)
            v_gibb = pv_gibb.bernoulli()
            h = self._ph_given_v(v_gibb).bernoulli()
        return v_gibb, pv_gibb
