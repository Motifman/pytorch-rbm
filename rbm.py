from matplotlib import _label_from_arg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


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
        """
        persistent constractive divergence(k)法によるサンプリング
        """
        h_pos = self._ph_given_v(v).bernoulli()
        h_neg = self.h_state
        for _ in range(self.k_cd):
            v_neg = self._pv_given_h(h_neg).bernoulli()
            h_neg = self._ph_given_v(v_neg).bernoulli()
        self.h_state = h_neg

        return h_pos, v_neg, h_neg

    def _sample_cd(self, v):
        """
        constractive divergence(k)法によるサンプリング
        """
        h_pos = self._ph_given_v(v).bernoulli()
        h_neg = h_pos
        for _ in range(self.k_cd):
            v_neg = self._pv_given_h(h_neg).bernoulli()
            h_neg = self._ph_given_v(v_neg).bernoulli()

        return h_pos, v_neg, h_neg

    def _sample_pcd_cont(self, v):
        """
        persistent constractive divergence(k)法によるサンプリング
        """
        ph_pos = self._ph_given_v(v)

        h_neg = self.h_state
        for _ in range(self.k_cd):
            v_neg = self._pv_given_h(h_neg).bernoulli()
            ph_neg = self._ph_given_v(v_neg)
            h_neg = ph_neg.bernoulli()
        self.h_state = h_neg

        return ph_pos, v_neg, ph_neg

    def grad(self, v, sample_type="pcd_cont"):
        """
        生成モデルの勾配計算
        """
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

    def grad_regressor(self, v, y):
        """
        v:(B, I), y:(B)
        自由エネルギーによる回帰を行う場合の勾配計算
        """
        h = self._ph_given_v(v)
        fe = self.free_energy(v)  # (B)
        error = fe - y
        w_grad = torch.einsum("B,BH,BI->HI", error, h, v)
        bh_grad = torch.einsum("B,BH->H", error, h)
        bv_grad = torch.einsum("B,BI->I", error, v)
        return {"weight": w_grad, "bias_h": bh_grad, "bias_v": bv_grad}

    def update(self, grad):
        """
        optimizerで計算した勾配でパラメータを更新
        """
        for key, value in grad.items():
            self.param[key] += value

    # def update(self, v, lr, sample_type="pcd_cont"):
    #     grad = self.grad(v, sample_type)
    #     for key, value in grad.items():
    #         self.param[key] += lr * value

    def free_energy(self, v):
        """
        データから自由エネルギーを計算
        """
        term1 = -torch.matmul(v, self.param["bias_v"].squeeze(0))
        bias = self.param["bias_h"] + torch.matmul(v, self.param["weight"].t())
        term2 = -torch.log(1 + torch.exp(bias) + 1e-10).sum(dim=1)
        return term1 + term2

    def _energy(self, v):
        """
        ボルツマンマシンのエネルギーを計算
        """
        v_term = torch.matmul(v, self.param["bias_v"].t())
        w_x_h = torch.matmul(v, self.param["weight"].t()) + self.param["bias_h"]
        h_term = torch.sum(F.softplus(w_x_h), dim=1)
        return -h_term - v_term

    def pseudo_likelihood(self, v):
        """
        学習の進行を確認するための疑似対数尤度を計算
        """
        flip = torch.randint(0, v.size(1), (1,))
        v_fliped = v.clone()
        v_fliped[:, flip] = 1 - v_fliped[:, flip]
        energy = self._energy(v)
        energy_fliped = self._energy(v_fliped)
        return v.size(1) * F.softplus(energy_fliped - energy)

    def sample_by_v(self, v, num_gib=300):
        """
        データから隠れ変数をサンプルして可視変数をサンプルする
        """
        h = self._ph_given_v(v).bernoulli()
        for _ in range(num_gib):
            pv_gibb = self._pv_given_h(h)
            v_gibb = pv_gibb.bernoulli()
            h = self._ph_given_v(v_gibb).bernoulli()
        return v_gibb, pv_gibb


class RBMClassification(nn.Module):
    """
    class labelとdataの同時分布をモデル化するボルツマンマシン
    """

    def __init__(
        self, visible_size, label_size, hidden_size, batch_size, k_cd=1, device="cpu"
    ):
        super().__init__()
        self.visible_size = visible_size
        self.label_size = label_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.device = device
        self.param = {
            "weight1": torch.normal(
                0, 0.01, size=[hidden_size, visible_size], device=device
            ),
            "weight2": torch.normal(
                0, 0.01, size=[hidden_size, label_size], device=device
            ),
            "bias_v1": torch.zeros([1, visible_size], device=device),
            "bias_v2": torch.zeros([1, label_size], device=device),
            "bias_h": torch.zeros([1, hidden_size], device=device),
        }
        self.h_state = torch.zeros([batch_size, hidden_size], device=device)
        self.k_cd = k_cd

    def _pv_data_given_h(self, h):
        return torch.sigmoid(
            F.linear(h, self.param["weight1"].t(), self.param["bias_v1"])
        )

    def _pv_class_given_h(self, h):
        return F.softmax(
            F.linear(h, self.param["weight2"].t(), self.param["bias_v2"]), dim=-1
        )

    def _ph_given_v(self, v, label):
        v = torch.cat([v, label], dim=-1)
        weight = torch.cat([self.param["weight1"], self.param["weight2"]], dim=1)
        return torch.sigmoid(F.linear(v, weight, self.param["bias_h"]))

    def _sample_pcd(self, v, label):
        """
        persistent constractive divergence(k)法によるサンプリング
        """
        h_pos = self._ph_given_v(v, label).bernoulli()
        h_neg = self.h_state
        for _ in range(self.k_cd):
            v_data_neg = self._pv_data_given_h(h_neg).bernoulli()
            v_class_neg = Categorical(self._pv_class_given_h(h_neg)).sample()
            h_neg = self._ph_given_v(v_data_neg, v_class_neg).bernoulli()
        self.h_state = h_neg

        return h_pos, v_data_neg, v_class_neg, h_neg

    def _sample_cd(self, v, label):
        """
        constractive divergence(k)法によるサンプリング
        """
        h_pos = self._ph_given_v(v, label).bernoulli()
        h_neg = h_pos
        for _ in range(self.k_cd):
            v_data_neg = self._pv_data_given_h(h_neg).bernoulli()
            v_class_neg = Categorical(self._pv_class_given_h(h_neg)).sample()
            h_neg = self._ph_given_v(v_data_neg, v_class_neg).bernoulli()

        return h_pos, v_data_neg, v_class_neg, h_neg

    def _sample_pcd_cont(self, v, label):
        """
        persistent constractive divergence(k)法によるサンプリング
        """
        ph_pos = self._ph_given_v(v, label)

        h_neg = self.h_state
        for _ in range(self.k_cd):
            v_data_neg = self._pv_data_given_h(h_neg).bernoulli()
            v_class_neg = Categorical(self._pv_class_given_h(h_neg)).sample()
            ph_neg = self._ph_given_v(v_data_neg, v_class_neg)
            h_neg = ph_neg.bernoulli()
        self.h_state = h_neg

        return ph_pos, v_data_neg, v_class_neg, ph_neg

    def grad(self, v, label, sample_type="pcd_cont"):
        """
        生成モデルの勾配計算
        """
        if sample_type == "pcd":
            h_pos, v_data_neg, v_class_neg, h_neg = self._sample_pcd(v, label)
        elif sample_type == "cd":
            h_pos, v_data_neg, v_class_neg, h_neg = self._sample_cd(v, label)
        elif sample_type == "pcd_cont":
            h_pos, v_data_neg, v_class_neg, h_neg = self._sample_pcd_cont(v, label)
        else:
            raise NotImplementedError("choose pcd, cd, or pcd_cont")
        B, _ = v.shape
        w1_grad = torch.matmul(h_pos.t(), v) - torch.matmul(h_neg.t(), v_data_neg)
        w1_grad = w1_grad / B
        w2_grad = torch.matmul(h_pos.t(), label) - torch.matmul(h_neg.t(), v_class_neg)
        w2_grad = w2_grad / B
        bh_grad = (h_pos - h_neg).mean(dim=0)
        bv1_grad = (v - v_data_neg).mean(dim=0)
        bv2_grad = (label - v_class_neg).mean(dim=0)
        return {
            "weight1": w1_grad,
            "weight2": w2_grad,
            "bias_h": bh_grad,
            "bias_v1": bv1_grad,
            "bias_v2": bv2_grad,
        }

    def update(self, grad):
        """
        optimizerで計算した勾配でパラメータを更新
        """
        for key, value in grad.items():
            self.param[key] += value

    def free_energy(self, v, label):
        """
        データから自由エネルギーを計算
        """
        term1 = -torch.matmul(v, self.param["bias_v1"].squeeze(0))
        term2 = -torch.matmul(label, self.param["bias_v2"].squeeze(0))
        bias = (
            self.param["bias_h"]
            + torch.matmul(v, self.param["weight1"].t())
            + torch.matmul(label, self.param["weight2"].t())
        )
        term2 = -torch.log(1 + torch.exp(bias) + 1e-10).sum(dim=1)
        return term1 + term2

    def _energy(self, v, label):
        """
        ボルツマンマシンのエネルギーを計算
        """
        v_term = torch.matmul(v, self.param["bias_v1"].t())
        label_term = torch.matmul(label, self.param["bias_v2"].t())
        w_v_h = torch.matmul(v, self.param["weight1"].t()) + self.param["bias_h"]
        w_label_h = (
            torch.matmul(label, self.param["weight2"].t()) + self.param["bias_h"]
        )
        h_v_term = torch.sum(F.softplus(w_v_h), dim=1)
        h_label_term = torch.sum(F.softplus(w_label_h), dim=1)
        return -h_v_term - h_label_term - v_term - label_term

    def pseudo_likelihood(self, v, label):
        """
        学習の進行を確認するための疑似対数尤度を計算
        """
        flip = torch.randint(0, v.size(1) + label.size(1), (1,))
        v_fliped = v.clone()
        v_fliped[:, flip] = 1 - v_fliped[:, flip]
        label_fliped = v.clone()
        label_fliped[:, flip] = 1 - label_fliped[:, flip]
        energy = self._energy(v, label)
        energy_fliped = self._energy(v_fliped, label_fliped)
        return (v.size(1) + label.size(1)) * F.softplus(energy_fliped - energy)

    def sample_by_v(self, v, label, num_gib=300):
        """
        データが与えられた条件で可視変数をサンプルする
        """
        h = self._ph_given_v(v, label).bernoulli()
        for _ in range(num_gib):
            pv_data_gibb = self._pv_data_given_h(h)
            pv_class_gibb = self._pv_class_given_h(h)
            v_data_gibb = pv_data_gibb.bernoulli()
            v_class_gibb = Categorical(pv_class_gibb).sample()
            h = self._ph_given_v(v_data_gibb, v_class_gibb).bernoulli()
        return v_data_gibb, v_class_gibb, pv_data_gibb, pv_class_gibb

    def classification(self, v):
        """
        自由エネルギーによるクラス分類
        """
        fe_list = []
        for i in range(self.label_size):
            labels = torch.zeros([self.batch_size, self.label_size], device=self.device)
            labels[:, i] = 1  # one-hotベクトルを作成
            fe_list.append(self.free_energy(v, labels))
        fe_tensor = torch.stack(fe_list, dim=-1)  # (B, label_size)
        pred_label = torch.argmax(fe_tensor, dim=-1)  # (B)
        return pred_label
