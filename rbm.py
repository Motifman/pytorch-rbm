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


class RBMNew(nn.Module):
    def __init__(self, vis_dim, hid_dim, initial_std=0.01, device="cpu"):
        super(RBM, self).__init__()
        self.device = device
        self.b = torch.zeros(1, vis_dim, device=device)
        self.c = torch.zeros(1, hid_dim, device=device)
        self.w = torch.empty((hid_dim, vis_dim), device=device).normal_(
            mean=0, std=initial_std
        )

    def _visible_to_hidden(self, v):
        """可視ユニットから隠れユニットをサンプル"""
        p = torch.sigmoid(F.linear(v, self.w, self.c))
        return p.bernoulli()

    def _hidden_to_visible(self, h):
        """隠れユニットから可視ユニットをサンプル"""
        p = torch.sigmoid(F.linear(h, self.w.t(), self.b))
        return p.bernoulli()

    def _visible_to_ph(self, v):
        """P(h=1|v)を計算"""
        return torch.sigmoid(F.linear(v, self.w, self.c))

    def sample(self, v, gib_num=1):
        """データをサンプリング"""
        v = v.view(-1, self.w.size(1)).to(self.device)
        h = self._visible_to_hidden(v)
        for _ in range(gib_num):
            v_gibb = self._hidden_to_visible(h)
            h = self._visible_to_hidden(v_gibb)
        return v_gibb

    def sample_ph(self, v, gib_num=15):
        """phをサンプリング"""
        v = v.view(-1, self.w.size(1)).to(self.device)
        ph = self._visible_to_ph(v)
        h = ph.bernoulli()
        # Gibbs Sampling 1 ~ k
        for _ in range(gib_num):
            v_gibb = self._hidden_to_visible(h)
            ph_gibb = self._visible_to_ph(v_gibb)
            h = ph_gibb.bernoulli()
        return ph_gibb

    def energy(self, v):
        """エネルギーを計算"""
        v_term = torch.matmul(v, self.b.t())
        w_x_h = torch.matmul(v, self.w.t()) + self.c
        h_term = torch.sum(F.softplus(w_x_h), dim=1)
        return -h_term - v_term

    def pseudo_likelihood(self, v):
        """疑似対数尤度を計算"""
        flip = torch.randint(0, v.size(1), (1,))
        v_fliped = v.clone()
        v_fliped[:, flip] = 1 - v_fliped[:, flip]
        energy = self.energy(v)
        energy_fliped = self.energy(v_fliped)
        return v.size(1) * F.softplus(energy_fliped - energy)

    def _update(self, v_pos, lr=0.1):
        """ミニバッチあたりの学習更新"""
        # positive part
        ph_pos = self._visible_to_ph(v_pos)
        # negative part
        v_neg = self._hidden_to_visible(self.h_states)
        ph_neg = self._visible_to_ph(v_neg)

        lr = lr / v_pos.size(0)
        # Update W
        update = torch.matmul(ph_pos.t(), v_pos) - torch.matmul(ph_neg.t(), v_neg)
        self.w += lr * update
        self.b += lr * torch.sum(v_pos - v_neg, dim=0)
        self.c += lr * torch.sum(ph_pos - ph_neg, dim=0)

        # PCDのために隠れユニットの値を保持
        self.h_states = ph_neg.bernoulli()

    def fit(self, data, n_epoch=10, lr=1e-1, batch_size=128):
        train = MyDataset(data[: int(len(data) * 0.7)])
        test = MyDataset(data[int(len(data) * 0.7) :])
        train_loader = torch.utils.data.DataLoader(
            dataset=train, batch_size=batch_size, shuffle=True, num_workers=0
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=train, batch_size=batch_size, shuffle=True, num_workers=0
        )

        train_loss_avg, val_loss_avg = [], []

        # pcd memory
        self.h_states = torch.zeros(batch_size, self.w.size(0), device=device)

        for epoch in tqdm(range(n_epoch)):
            train_loss_avg.append(0)
            val_loss_avg.append(0)

            self.train()
            for i, data in enumerate(train_loader):
                data = data.to(self.device)
                self._update(data)
                train_loss_avg[-1] += -self.pseudo_likelihood(data).mean().item()
            train_loss_avg[-1] /= data.size(1)

            self.eval()
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    data = data.view(-1, self.w.size(1)).to(self.device)
                    val_loss_avg[-1] += -self.pseudo_likelihood(data).mean().item()
                val_loss_avg[-1] /= data.size(1)

            print(
                f"[EPOCH]: {epoch+1}, [LOSS]: {train_loss_avg[-1]:.4f}, [VAL]: {val_loss_avg[-1]:.4f}"
            )
        return train_loss_avg, val_loss_avg
