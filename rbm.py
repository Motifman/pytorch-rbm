import torch
import torch.nn as nn
import torch.nn.functional as F


class RBM(nn.Module):
    def __init__(self, visible_size, hidden_size, k_cd=1, device="cpu"):
        super().__init__()
        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(
            torch.normal(0, 0.01, size=[hidden_size, visible_size])
        )
        self.bias_v = nn.Parameter(torch.zeros([1, visible_size]))
        self.bias_h = nn.Parameter(torch.zeros([1, hidden_size]))
        self.h_state = torch.zeros([1, hidden_size]).to(device)
        self.k_cd = k_cd

    def _pv_given_h(self, h):
        return torch.sigmoid(F.linear(h, self.weight.t(), self.bias_v))

    def _ph_given_v(self, v):
        return torch.sigmoid(F.linear(v, self.weight, self.bias_h))

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

    def grad(self, v, sample_type="pcd"):
        if sample_type == "pcd":
            h_pos, v_neg, h_neg = self._sample_pcd(v)
        elif sample_type == "cd":
            h_pos, v_neg, h_neg = self._sample_cd(v)
        else:
            raise NotImplementedError("choose pcd pr cd")
        B, _ = v.shape
        hv_pos = torch.einsum("BH,BV->HV", h_pos, v)
        hv_neg = torch.einsum("BH,BV->HV", h_neg, v_neg)
        w_grad = (hv_pos - hv_neg) / B
        bh_grad = (h_pos - h_neg).mean(dim=0)
        bv_grad = (v - v_neg).mean(dim=0)
        return {"weight": w_grad, "bias_h": bh_grad, "bias_v": bv_grad}

    def update(self, grad):
        param = self.state_dict()
        for key, value in grad.items():
            param[key] += value

    def free_energy(self, v):
        # 訓練データによる平均自由エネルギーを計算する
        term1 = -torch.matmul(v, self.bias_v.squeeze(0))
        bias = self.bias_h + torch.matmul(v, self.weight.t())
        term2 = -torch.log(1 + torch.exp(bias)).sum(dim=1)
        return term1 + term2

    def sample_by_v(self, v, num_gib=1000):
        # batchを受け取り、データを再構成する
        h = self._ph_given_v(v).bernoulli()
        for _ in range(num_gib):
            pv_gibb = self._pv_given_h(h)
            v_gibb = pv_gibb.bernoulli()
            h = self._ph_given_v(v_gibb).bernoulli()
        return v_gibb, pv_gibb
