"""
model.py  –  PPO Agent  (vectorised · NPZ checkpoints · fast)
==============================================================
Pure NumPy — no PyTorch required.

Observation : 22 floats  (20 ray fracs + speed_norm + steer_norm)
Actions     : 9  (COAST, THROTTLE, BRAKE, LEFT, RIGHT,
                  THROTTLE_LEFT, THROTTLE_RIGHT, BRAKE_LEFT, BRAKE_RIGHT)
"""

import os, json, math, time
import numpy as np

# ── Paths ──────────────────────────────────────────────────────
_HERE         = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR    = os.path.join(_HERE, "models")
MANIFEST_PATH = os.path.join(MODELS_DIR, "manifest.json")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Hyper-parameters ───────────────────────────────────────────
OBS_DIM      = 22
ACT_DIM      = 9
HIDDEN       = 128
GAMMA        = 0.995  # decisions are ~60 Hz (frame-skip), so ~3 s horizon
LAM          = 0.95
CLIP_EPS     = 0.2
LR_ACTOR     = 3e-4
LR_CRITIC    = 1e-3
ENTROPY_COEF = 0.05
ENTROPY_MIN  = 0.01
EPOCHS       = 4
UPDATE_EVERY = 4096      # decisions per PPO update (notebook: 32768 = 256 cars x 128 steps)
MINIBATCHES  = 8         # minibatches per epoch (same ratio as the notebook: 32768 / 4096)
# Notebook decays the entropy bonus by 0.9985 per 32768 decisions; scale so the schedule
# per DECISION is identical even though updates here are smaller.
ENTROPY_DECAY= 0.9985 ** (UPDATE_EVERY / 32768)
TARGET_KL    = 0.03      # stop an update early if the policy moved too far
MAX_GRAD     = 0.5       # global gradient-norm clip (actor + critic together)
MAX_EP_STEPS = 1800      # 30 s episode cap, treated as truncation (bootstrapped), not failure
CRASH_PENALTY= -15.0

# ── Action map ─────────────────────────────────────────────────
COAST=0; THROTTLE=1; BRAKE=2; LEFT=3; RIGHT=4
THROTTLE_LEFT=5; THROTTLE_RIGHT=6; BRAKE_LEFT=7; BRAKE_RIGHT=8

ACTION_MAP = {
    COAST:          (0.0, False,  0.0),
    THROTTLE:       (1.0, False,  0.0),
    BRAKE:          (0.0, True,   0.0),
    LEFT:           (0.3, False, -1.0),
    RIGHT:          (0.3, False, +1.0),
    THROTTLE_LEFT:  (0.9, False, -1.0),
    THROTTLE_RIGHT: (0.9, False, +1.0),
    BRAKE_LEFT:     (0.0, True,  -1.0),
    BRAKE_RIGHT:    (0.0, True,  +1.0),
}

# ── Math helpers ───────────────────────────────────────────────

def softmax_2d(X):
    E = np.exp(X - X.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)

def orthogonal(fi, fo, gain):
    """Orthogonal init for a (fi, fo) weight matrix, like torch.nn.init.orthogonal_."""
    A = np.random.randn(max(fi, fo), min(fi, fo))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    return gain * (Q if fi >= fo else Q.T)

def clip_grads(gs, mx):
    norm = math.sqrt(sum(float(np.sum(g**2)) for g in gs))
    if norm > mx:
        s = mx / (norm + 1e-8)
        return [g * s for g in gs]
    return gs


class RunningNorm:
    def __init__(self, eps=1e-4):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = eps

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        batch_mean  = x.mean()
        batch_var   = x.var()
        batch_count = x.size
        delta = batch_mean - self.mean
        tot   = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2  = m_a + m_b + delta**2 * self.count * batch_count / tot
        self.mean  = new_mean
        self.var   = M2 / tot
        self.count = tot

    @property
    def std(self):
        return math.sqrt(max(self.var, 1e-8))

    def norm(self, x):
        return (x - self.mean) / self.std

    def denorm(self, x):
        return x * self.std + self.mean

    def to_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    @classmethod
    def from_dict(cls, d):
        rn = cls()
        rn.mean, rn.var, rn.count = d["mean"], d["var"], d["count"]
        return rn


# ── MLP ────────────────────────────────────────────────────────

class MLP:
    def __init__(self, in_dim, hid, out_dim, out_gain=1.0):
        g = math.sqrt(2.0)
        self.W1 = orthogonal(in_dim, hid, g);        self.b1 = np.zeros(hid)
        self.W2 = orthogonal(hid,    hid, g);        self.b2 = np.zeros(hid)
        self.W3 = orthogonal(hid, out_dim, out_gain); self.b3 = np.zeros(out_dim)
        self._c = {}

    def forward(self, x):
        return self.forward_batch(x[None])[0]

    def forward_batch(self, X):
        Z1 = X @ self.W1 + self.b1;  H1 = np.tanh(Z1)
        Z2 = H1 @ self.W2 + self.b2; H2 = np.tanh(Z2)
        out = H2 @ self.W3 + self.b3
        self._c = dict(X=X, Z1=Z1, H1=H1, Z2=Z2, H2=H2)
        return out

    def backward_batch(self, d_out):
        c = self._c; N = len(c['X'])
        dW3 = c['H2'].T @ d_out / N; db3 = d_out.mean(0)
        dH2 = d_out @ self.W3.T
        dZ2 = dH2 * (1 - np.tanh(c['Z2'])**2)
        dW2 = c['H1'].T @ dZ2 / N;  db2 = dZ2.mean(0)
        dH1 = dZ2 @ self.W2.T
        dZ1 = dH1 * (1 - np.tanh(c['Z1'])**2)
        dW1 = c['X'].T @ dZ1 / N;   db1 = dZ1.mean(0)
        return [dW1, db1, dW2, db2, dW3, db3]

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def to_dict(self):
        return {k: getattr(self, k).tolist()
                for k in ('W1','b1','W2','b2','W3','b3')}

    @classmethod
    def from_dict(cls, d, in_dim, hid, out_dim):
        m = cls(in_dim, hid, out_dim)
        for k in ('W1','b1','W2','b2','W3','b3'):
            setattr(m, k, np.array(d[k], dtype=np.float64))
        return m


# ── Adam ───────────────────────────────────────────────────────

class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-5):
        self.lr=lr; self.b1=b1; self.b2=b2; self.eps=eps
        self.m=[np.zeros_like(p) for p in params]
        self.v=[np.zeros_like(p) for p in params]
        self.t=0

    def step(self, params, grads):
        self.t += 1
        bc1 = 1 - self.b1**self.t; bc2 = 1 - self.b2**self.t
        for i,(p,g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1*self.m[i] + (1-self.b1)*g
            self.v[i] = self.b2*self.v[i] + (1-self.b2)*g*g
            p -= self.lr * (self.m[i]/bc1) / (np.sqrt(self.v[i]/bc2) + self.eps)

    def state_dict(self):
        return {
            "t": self.t,
            "m": [x.tolist() for x in self.m],
            "v": [x.tolist() for x in self.v],
        }

    def load_state_dict(self, d):
        self.t = d["t"]
        self.m = [np.array(x, dtype=np.float64) for x in d["m"]]
        self.v = [np.array(x, dtype=np.float64) for x in d["v"]]


# ── PPO Agent ──────────────────────────────────────────────────

class PPOAgent:

    def __init__(self, max_speed=180.0, max_steer_deg=28.0):
        self.max_speed     = max_speed
        self.max_steer_deg = max_steer_deg
        self.actor  = MLP(OBS_DIM, HIDDEN, ACT_DIM, out_gain=0.01)
        self.critic = MLP(OBS_DIM, HIDDEN, 1,       out_gain=1.0)
        self.actor_opt  = Adam(self.actor.params(),  lr=LR_ACTOR)
        self.critic_opt = Adam(self.critic.params(), lr=LR_CRITIC)
        self.entropy_coef   = ENTROPY_COEF
        self.episode        = 0
        self.total_steps    = 0
        self.best_reward    = -1e9
        self.reward_history = []
        self.last_actor_loss  = 0.0
        self.last_critic_loss = 0.0
        self.ret_norm = RunningNorm()

    # ── observation ────────────────────────────────────────────

    def build_obs(self, rays, speed, steer):
        fracs = [r['frac'] if isinstance(r, dict) else float(r)
                 for r in (rays or [])]
        fracs = (fracs + [1.0]*20)[:20]
        sn = float(np.clip(speed / self.max_speed, -1.0, 1.0))
        an = float(np.clip(math.degrees(steer) / self.max_steer_deg, -1.0, 1.0))
        return np.array(fracs + [sn, an], dtype=np.float64)

    # ── action selection ────────────────────────────────────────

    def select_action(self, obs):
        # ε-greedy: occasionally take a fully-random action so the policy
        # keeps seeing new states even after it has mostly converged.
        # Pure on-policy sampling (no ε-greedy: it made the data off-policy and
        # biased the gradient). Exploration comes from the entropy bonus.
        logits   = self.actor.forward(obs)
        probs    = softmax_2d(logits[None])[0]
        probs    = np.clip(probs, 1e-8, 1.0); probs /= probs.sum()
        action   = int(np.random.choice(len(probs), p=probs))
        log_prob = float(np.log(probs[action]))
        raw_value = float(self.critic.forward(obs)[0])
        value     = self.ret_norm.denorm(raw_value)
        return action, log_prob, value

    @staticmethod
    def action_to_controls(idx):
        assert idx in ACTION_MAP, f"Invalid action index: {idx}"
        return ACTION_MAP[idx]

    # ── reward ─────────────────────────────────────────────────

    def compute_reward(self, speed, lat_norm, fwd, crashed):
        """Same reward as the notebook.
        lat_norm = distance from centre line / half lane width,
        fwd      = cos(angle between heading and the road direction of travel)."""
        if crashed:
            return CRASH_PENALTY
        return float(3.0 * speed * fwd / self.max_speed - 2.0 * lat_norm ** 2)

    # ── GAE ────────────────────────────────────────────────────

    @staticmethod
    def compute_gae(rewards, values, dones, last_val=0.0):
        """
        rewards, values: length T lists.
        last_val: V(s_T) — bootstrap value for the state AFTER the buffer ends.
                  Pass 0.0 for terminal episodes; pass critic(final_obs) for
                  non-terminal rollout truncation.
        """
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float64); last = 0.0
        for t in reversed(range(T)):
            nv   = values[t+1] if t+1 < T else last_val
            mask = 0.0 if dones[t] else 1.0
            d    = rewards[t] + GAMMA * nv * mask - values[t]
            last = d + GAMMA * LAM * mask * last
            adv[t] = last
        return adv, adv + np.array(values[:T], dtype=np.float64)

    # ── PPO update ─────────────────────────────────────────────

    def update(self, buffer, last_val=0.0):
        if len(buffer) < 16:
            return 0.0, 0.0, 0.0

        obs  = np.array([b[0] for b in buffer], dtype=np.float64)
        acts = np.array([b[1] for b in buffer], dtype=np.int32)
        lpo  = np.array([b[2] for b in buffer], dtype=np.float64)
        adv, ret = self.compute_gae(
            [b[3] for b in buffer],
            [b[4] for b in buffer],
            [b[5] for b in buffer],
            last_val=last_val)

        self.ret_norm.update(ret)
        ret_n = self.ret_norm.norm(ret)
        adv   = (adv - adv.mean()) / (adv.std() + 1e-8)      # normalised over the whole batch

        N  = len(buffer)
        mb = max(N // MINIBATCHES, 16)
        ta = tc = te = nu = 0.0

        for _ in range(EPOCHS):
            kls = []
            perm = np.random.permutation(N)
            for s in range(0, N, mb):
                bi = perm[s:s+mb]
                if len(bi) < 4: continue
                n = len(bi)
                bo, ba, bl = obs[bi], acts[bi], lpo[bi]
                bA, bR = adv[bi], ret_n[bi]

                # ── Actor (per-sample gradients; backward_batch averages them) ──
                logits = self.actor.forward_batch(bo)
                probs  = softmax_2d(logits)
                probs  = np.clip(probs, 1e-8, 1.0)
                probs /= probs.sum(1, keepdims=True)

                lp_new = np.log(probs[np.arange(n), ba])
                logr   = lp_new - bl
                ratio  = np.exp(logr)
                rc     = np.clip(ratio, 1-CLIP_EPS, 1+CLIP_EPS)
                kls.append(float(np.mean((ratio - 1.0) - logr)))

                surr1  = ratio * bA
                surr2  = rc * bA
                active = (surr1 <= surr2).astype(np.float64)   # 0 where the clip is binding
                d_lp   = ratio * (-bA * active)

                d_logits = -probs.copy()
                d_logits[np.arange(n), ba] += 1.0
                d_logits *= d_lp[:, None]

                ent    = -np.sum(probs * np.log(probs+1e-8), 1)
                d_ent  = probs * (np.log(probs+1e-8) + 1.0)
                d_ent -= d_ent.sum(1, keepdims=True) * probs    # = -dH/dz
                d_logits += self.entropy_coef * d_ent           # loss = pg - coef*H

                aloss = float(-(np.minimum(surr1, surr2).mean()) - self.entropy_coef * ent.mean())
                ag = self.actor.backward_batch(d_logits)

                # ── Critic: 0.5 * MSE on normalised returns ──
                vp    = self.critic.forward_batch(bo)[:, 0]
                closs = float(0.5 * np.mean((vp - bR)**2))
                cg    = self.critic.backward_batch((vp - bR)[:, None])

                # ── One global gradient-norm clip over actor + critic, then step ──
                clipped = clip_grads(ag + cg, MAX_GRAD)
                self.actor_opt.step(self.actor.params(),  clipped[:len(ag)])
                self.critic_opt.step(self.critic.params(), clipped[len(ag):])

                ta += aloss; tc += closs; te += float(ent.mean()); nu += 1

            if kls and np.mean(kls) > 1.5 * TARGET_KL:          # policy moved too far: stop epochs
                break

        self.entropy_coef = max(ENTROPY_MIN, self.entropy_coef * ENTROPY_DECAY)
        n = max(nu, 1)
        return ta/n, tc/n, te/n

    # ── Save (.npz) ────────────────────────────────────────────

    def save_npz(self, label=None):
        ts   = time.strftime("%Y%m%d_%H%M%S")
        name = f"ppo_{label or ts}.npz"
        path = os.path.join(MODELS_DIR, name)

        actor_opt_state  = self.actor_opt.state_dict()
        critic_opt_state = self.critic_opt.state_dict()

        np.savez_compressed(path,
            a_W1=self.actor.W1,  a_b1=self.actor.b1,
            a_W2=self.actor.W2,  a_b2=self.actor.b2,
            a_W3=self.actor.W3,  a_b3=self.actor.b3,
            c_W1=self.critic.W1, c_b1=self.critic.b1,
            c_W2=self.critic.W2, c_b2=self.critic.b2,
            c_W3=self.critic.W3, c_b3=self.critic.b3,
            episode          = np.array(self.episode),
            total_steps      = np.array(self.total_steps),
            best_reward      = np.array(self.best_reward),
            entropy_coef     = np.array(self.entropy_coef),
            reward_history   = np.array(self.reward_history[-200:], dtype=np.float32),
            last_actor_loss  = np.array(self.last_actor_loss,  dtype=np.float64),
            last_critic_loss = np.array(self.last_critic_loss, dtype=np.float64),
            ret_norm_mean    = np.array(self.ret_norm.mean,  dtype=np.float64),
            ret_norm_var     = np.array(self.ret_norm.var,   dtype=np.float64),
            ret_norm_count   = np.array(self.ret_norm.count, dtype=np.float64),
            max_speed        = np.array(self.max_speed,       dtype=np.float64),
            max_steer_deg    = np.array(self.max_steer_deg,   dtype=np.float64),
            # Adam state stored as JSON string in a 0-d object array
            actor_opt_state  = np.array(json.dumps(actor_opt_state),  dtype=object),
            critic_opt_state = np.array(json.dumps(critic_opt_state), dtype=object),
        )
        manifest = _load_manifest()
        manifest.append({"label": label or ts, "timestamp": ts,
                         "episode": self.episode,
                         "best_reward": round(self.best_reward, 3),
                         "file": name})
        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2)
        return path

    def save_version(self, label=None):
        return self.save_npz(label)

    # ── Load ───────────────────────────────────────────────────

    @classmethod
    def load_npz(cls, path):
        d = np.load(path, allow_pickle=True)
        max_speed     = float(d['max_speed'])     if 'max_speed'     in d else 180.0
        max_steer_deg = float(d['max_steer_deg']) if 'max_steer_deg' in d else 28.0
        ag = cls(max_speed=max_speed, max_steer_deg=max_steer_deg)

        for net, pfx in [(ag.actor,'a_'), (ag.critic,'c_')]:
            for k in ('W1','b1','W2','b2','W3','b3'):
                setattr(net, k, d[pfx+k].copy())

        ag.episode        = int(d['episode'])
        ag.total_steps    = int(d['total_steps'])
        ag.best_reward    = float(d['best_reward'])
        ag.entropy_coef   = float(d['entropy_coef'])
        ag.reward_history = d['reward_history'].tolist()
        ag.last_actor_loss  = float(d['last_actor_loss'])  if 'last_actor_loss'  in d else 0.0
        ag.last_critic_loss = float(d['last_critic_loss']) if 'last_critic_loss' in d else 0.0

        if 'ret_norm_mean' in d:
            ag.ret_norm.mean  = float(d['ret_norm_mean'])
            ag.ret_norm.var   = float(d['ret_norm_var'])
            ag.ret_norm.count = float(d['ret_norm_count'])

        # Restore Adam state if present (preserves optimizer momentum across saves)
        ag.actor_opt  = Adam(ag.actor.params(),  lr=LR_ACTOR)
        ag.critic_opt = Adam(ag.critic.params(), lr=LR_CRITIC)
        if 'actor_opt_state' in d:
            try:
                ag.actor_opt.load_state_dict(json.loads(str(d['actor_opt_state'])))
                ag.critic_opt.load_state_dict(json.loads(str(d['critic_opt_state'])))
            except Exception:
                pass   # old checkpoint without optimizer state — fresh Adam is fine

        return ag

    @classmethod
    def load_version(cls, path):
        if path.endswith('.npz'):
            return cls.load_npz(path)
        with open(path) as f:
            data = json.load(f)
        ag = cls()
        ag.actor  = MLP.from_dict(data['actor'],  OBS_DIM, HIDDEN, ACT_DIM)
        ag.critic = MLP.from_dict(data['critic'], OBS_DIM, HIDDEN, 1)
        ag.episode        = data.get('episode', 0)
        ag.best_reward    = data.get('best_reward', -1e9)
        ag.total_steps    = data.get('total_steps', 0)
        ag.entropy_coef   = data.get('entropy_coef', ENTROPY_COEF)
        ag.reward_history = data.get('reward_history', [])
        if 'ret_norm' in data:
            ag.ret_norm = RunningNorm.from_dict(data['ret_norm'])
        ag.actor_opt  = Adam(ag.actor.params(),  lr=LR_ACTOR)
        ag.critic_opt = Adam(ag.critic.params(), lr=LR_CRITIC)
        return ag


# ── Manifest helpers ───────────────────────────────────────────

def _load_manifest():
    try:
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    except Exception:
        return []

load_manifest = _load_manifest


def get_latest_model_path():
    m = _load_manifest()
    if not m:
        return None
    latest = sorted(m, key=lambda x: x["timestamp"])[-1]
    path   = os.path.join(MODELS_DIR, latest["file"])
    return path if os.path.exists(path) else None


# ── PPO Trainer ────────────────────────────────────────────────

class PPOTrainer:
    """
    Call .step(rays, speed, steer, crashed, dt) every frame.
    Returns (throttle, brake, steer_dir).
    """
    def __init__(self, agent: PPOAgent):
        self.agent      = agent
        self.buffer     = []
        self.ep_reward  = 0.0
        self.ep_steps   = 0
        self.last_obs   = None
        self.last_act   = None
        self.last_lp    = None
        self.last_val   = None
        self.last_steer = 0.0
        self.reset_requested = False   # set when an episode hit MAX_EP_STEPS: caller must respawn the car
        self.stats = {
            "episode":        agent.episode,
            "ep_reward":      0.0,
            "best_reward":    agent.best_reward,
            "total_steps":    agent.total_steps,
            "actor_loss":     agent.last_actor_loss,
            "critic_loss":    agent.last_critic_loss,
            "entropy_coef":   agent.entropy_coef,
            "reward_history": agent.reward_history[-50:],
            "running":        True,
        }

    def step(self, rays, speed, steer, crashed, dt, track=(0.0, 1.0),
             training_state_path=None):
        """Call once per decision (frame).  track = (lateral offset / half lane width,
        cos(heading vs road direction)) - used ONLY for the reward, never fed to the policy.
        If self.reset_requested is True afterwards, respawn the car and call step() again."""
        obs = self.agent.build_obs(rays, speed, steer)
        self.reset_requested = False
        episode_end = truncated = False

        # Store previous transition (reward is for the state the last action led to)
        if self.last_obs is not None:
            r = self.agent.compute_reward(speed, track[0], track[1], crashed)
            self.ep_reward += r
            self.ep_steps  += 1
            truncated = (not crashed) and self.ep_steps >= MAX_EP_STEPS
            if truncated:   # time limit is not a failure: bootstrap the value of where we stopped
                r += GAMMA * self.agent.ret_norm.denorm(float(self.agent.critic.forward(obs)[0]))
            episode_end = crashed or truncated
            self.buffer.append((self.last_obs, self.last_act,
                                self.last_lp, r, self.last_val, episode_end))

        if episode_end:
            self.agent.episode     += 1
            self.agent.total_steps += self.ep_steps
            self.agent.reward_history.append(round(self.ep_reward, 3))
            if self.ep_reward > self.agent.best_reward:
                self.agent.best_reward = self.ep_reward

            # the last transition is marked done, so last_val is irrelevant here
            if len(self.buffer) >= UPDATE_EVERY:
                al, cl, _ = self.agent.update(self.buffer, last_val=0.0)
                self.buffer.clear()
                self.stats["actor_loss"]  = round(al, 6)
                self.stats["critic_loss"] = round(cl, 6)
                self.agent.last_actor_loss  = al
                self.agent.last_critic_loss = cl

            self.stats.update({
                "episode":        self.agent.episode,
                "best_reward":    round(self.agent.best_reward, 3),
                "total_steps":    self.agent.total_steps,
                "entropy_coef":   round(self.agent.entropy_coef, 5),
                "reward_history": self.agent.reward_history[-50:],
            })
            self.ep_reward = 0.0
            self.ep_steps  = 0

            if truncated:   # the car is still on the old episode: ask the caller to respawn
                self.last_obs = None
                self.reset_requested = True
                return PPOAgent.action_to_controls(COAST)

        # Flush buffer mid-episode if it hits UPDATE_EVERY (non-terminal truncation)
        elif len(self.buffer) >= UPDATE_EVERY:
            raw_last_val = float(self.agent.critic.forward(obs)[0])
            last_val     = self.agent.ret_norm.denorm(raw_last_val)
            al, cl, _ = self.agent.update(self.buffer, last_val=last_val)
            self.buffer.clear()
            self.stats["actor_loss"]  = round(al, 6)
            self.stats["critic_loss"] = round(cl, 6)
            self.agent.last_actor_loss  = al
            self.agent.last_critic_loss = cl

        self.stats["ep_reward"] = round(self.ep_reward, 2)

        if training_state_path:
            try:
                try:
                    with open(training_state_path) as _f:
                        existing = json.load(_f)
                except Exception:
                    existing = {}
                existing.update(self.stats)
                with open(training_state_path, "w") as f:
                    json.dump(existing, f)
            except OSError:
                pass

        action, lp, val = self.agent.select_action(obs)
        self.last_obs   = obs
        self.last_act   = action
        self.last_lp    = lp
        self.last_val   = val
        self.last_steer = steer
        return PPOAgent.action_to_controls(action)
