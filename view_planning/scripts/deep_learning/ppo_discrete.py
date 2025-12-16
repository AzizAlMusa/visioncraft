# ppo_discrete.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PolicyValueNet(nn.Module):
    """
    Simple MLP for discrete PPO.

    Input: obs_dim
    Outputs:
      - logits over actions (act_dim)
      - scalar value V(s)
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden_sizes=(256, 256)):
        super().__init__()
        h1, h2 = hidden_sizes

        self.fc1 = nn.Linear(obs_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc_pi = nn.Linear(h2, act_dim)
        self.fc_v = nn.Linear(h2, 1)

    def forward(self, obs: torch.Tensor):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        logits = self.fc_pi(x)
        value = self.fc_v(x).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor):
        """
        obs: (obs_dim,) or (B,obs_dim)
        Returns: action (int or tensor), log_prob, value
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action.squeeze(0), logp.squeeze(0), value.squeeze(0)


class PPOAgent:
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 vf_coef: float = 0.5,
                 ent_coef: float = 0.01,
                 device: str = "cpu"):
        self.device = torch.device(device)
        self.net = PolicyValueNet(obs_dim, act_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

    def collect_rollouts(self, env, min_steps: int):
        """
        Collect rollouts until we have at least 'min_steps' transitions.
        Returns a dict of numpy arrays (obs, act, logp, val, rew, done, ep_returns, ep_lengths).
        """
        obs_buf = []
        act_buf = []
        logp_buf = []
        val_buf = []
        rew_buf = []
        done_buf = []

        ep_returns = []
        ep_lengths = []

        steps_collected = 0

        while steps_collected < min_steps:
            obs = env.reset()
            done = False
            ep_rew = 0.0
            ep_len = 0

            while not done:
                obs_t = torch.from_numpy(obs).float().to(self.device)
                with torch.no_grad():
                    a, logp, v = self.net.act(obs_t)

                a_int = int(a.item())
                next_obs, r, done, info = env.step(a_int)

                obs_buf.append(obs)
                act_buf.append(a_int)
                logp_buf.append(logp.item())
                val_buf.append(v.item())
                rew_buf.append(r)
                done_buf.append(float(done))

                ep_rew += r
                ep_len += 1
                steps_collected += 1
                obs = next_obs

                if steps_collected >= min_steps:
                    break

            ep_returns.append(ep_rew)
            ep_lengths.append(ep_len)

        data = {
            "obs": np.array(obs_buf, dtype=np.float32),
            "act": np.array(act_buf, dtype=np.int64),
            "logp": np.array(logp_buf, dtype=np.float32),
            "val": np.array(val_buf, dtype=np.float32),
            "rew": np.array(rew_buf, dtype=np.float32),
            "done": np.array(done_buf, dtype=np.float32),
            "ep_returns": np.array(ep_returns, dtype=np.float32),
            "ep_lengths": np.array(ep_lengths, dtype=np.float32),
        }
        return data

    def _compute_advantages(self, rew, val, done, last_val):
        """
        GAE advantage computation.

        rew, val, done: numpy arrays of shape (T,)
        last_val: float value of V(s_T) for bootstrap at end.
        """
        T = len(rew)
        adv = np.zeros_like(rew)
        last_gae = 0.0

        for t in reversed(range(T)):
            nonterminal = 1.0 - done[t]
            delta = rew[t] + self.gamma * last_val * nonterminal - val[t]
            last_gae = delta + self.gamma * self.gae_lambda * nonterminal * last_gae
            adv[t] = last_gae
            last_val = val[t]
        ret = adv + val
        return adv, ret

    def update(self, data, train_epochs: int = 10, batch_size: int = 64):
        obs = torch.from_numpy(data["obs"]).float().to(self.device)
        act = torch.from_numpy(data["act"]).long().to(self.device)
        logp_old = torch.from_numpy(data["logp"]).float().to(self.device)
        val_old = torch.from_numpy(data["val"]).float().to(self.device)
        rew = data["rew"]
        done = data["done"]

        with torch.no_grad():
            # bootstrap with last value = 0 (episodes ended)
            adv, ret = self._compute_advantages(
                rew=rew,
                val=val_old.cpu().numpy(),
                done=done,
                last_val=0.0,
            )
        adv = torch.from_numpy(adv).float().to(self.device)
        ret = torch.from_numpy(ret).float().to(self.device)

        # normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        N = obs.shape[0]
        for epoch in range(train_epochs):
            idx = np.arange(N)
            np.random.shuffle(idx)

            for start in range(0, N, batch_size):
                end = start + batch_size
                batch_idx = idx[start:end]

                batch_obs = obs[batch_idx]
                batch_act = act[batch_idx]
                batch_logp_old = logp_old[batch_idx]
                batch_adv = adv[batch_idx]
                batch_ret = ret[batch_idx]

                logits, value = self.net(batch_obs)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(batch_act)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - batch_logp_old)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio,
                                    1.0 - self.clip_ratio,
                                    1.0 + self.clip_ratio) * batch_adv
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                value_loss = F.mse_loss(value, batch_ret)

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }
