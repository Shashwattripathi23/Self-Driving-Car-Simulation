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
GAMMA        = 0.99
LAM          = 0.95
CLIP_EPS     = 0.2
LR_ACTOR     = 5e-4
LR_CRITIC    = 1e-3
ENTROPY_COEF = 0.02   # was 0.05 — too high for 9 discrete actions, was fighting convergence
ENTROPY_DECAY= 0.999
ENTROPY_MIN  = 0.005
ACTOR_CLIP   = 1.0    # was 0.5 (shared) — actor needs more room now EPOCHS is lower
CRITIC_CLIP  = 0.5    # critic stays tight; normalized targets keep this safe
EPOCHS       = 5      # was 10 — fewer passes per rollout, less overfit to one batch of experience
BATCH_SIZE   = 256
UPDATE_EVERY = 1024   # collect this many steps then update

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
    """Row-wise softmax for (N, K) array."""
    E = np.exp(X - X.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)

def he(fi, fo):
    return np.random.randn(fi, fo) * math.sqrt(2.0 / fi)

def clip_grads(gs, mx):
    norm = math.sqrt(sum(float(np.sum(g**2)) for g in gs))
    if norm > mx:
        s = mx / (norm + 1e-8)
        return [g * s for g in gs]
    return gs


class RunningNorm:
    """
    Tracks running mean/std of a scalar stream (Welford's algorithm) so the
    critic can be trained on normalized targets instead of raw returns,
    which can blow up to the hundreds/thousands under high GAMMA + long
    episodes. norm()/denorm() convert between raw and normalized scale.
    """
    def __init__(self, eps=1e-4):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = eps   # avoid div-by-zero before any data seen

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


# ── MLP with batched forward / backward ────────────────────────

class MLP:
    def __init__(self, in_dim, hid, out_dim):
        self.W1 = he(in_dim, hid);  self.b1 = np.zeros(hid)
        self.W2 = he(hid,    hid);  self.b2 = np.zeros(hid)
        self.W3 = he(hid, out_dim); self.b3 = np.zeros(out_dim)
        self._c = {}

    # single-sample shim
    def forward(self, x):
        return self.forward_batch(x[None])[0]

    def forward_batch(self, X):
        """X: (N, in) → (N, out). Caches for backward_batch."""
        Z1 = X @ self.W1 + self.b1;  H1 = np.tanh(Z1)
        Z2 = H1 @ self.W2 + self.b2; H2 = np.tanh(Z2)
        out = H2 @ self.W3 + self.b3
        self._c = dict(X=X, Z1=Z1, H1=H1, Z2=Z2, H2=H2)
        return out

    def backward_batch(self, d_out):
        """d_out: (N, out) → averaged param grads."""
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
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
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


# ── PPO Agent ─────────────────────────────────────────────────

class PPOAgent:

    def __init__(self):
        self.actor  = MLP(OBS_DIM, HIDDEN, ACT_DIM)
        self.critic = MLP(OBS_DIM, HIDDEN, 1)
        self.actor_opt  = Adam(self.actor.params(),  lr=LR_ACTOR)
        self.critic_opt = Adam(self.critic.params(), lr=LR_CRITIC)
        self.entropy_coef   = ENTROPY_COEF
        self.episode        = 0
        self.total_steps    = 0
        self.best_reward    = -1e9
        self.reward_history = []
        self.last_actor_loss  = 0.0   # persisted so dashboard shows real values on resume
        self.last_critic_loss = 0.0
        self.ret_norm = RunningNorm()  # tracks return scale so critic trains on normalized targets

    # ── observation ────────────────────────────────────────

    @staticmethod
    def build_obs(rays, speed, steer, max_speed=180.0, max_steer_deg=28.0):
        fracs = [r['frac'] if isinstance(r, dict) else float(r)
                 for r in (rays or [])]
        fracs = (fracs + [1.0]*20)[:20]
        sn = float(np.clip(speed / max_speed, -1.0, 1.0))
        an = float(np.clip(math.degrees(steer) / max_steer_deg, -1.0, 1.0))
        return np.array(fracs + [sn, an], dtype=np.float64)

    # ── action selection ────────────────────────────────────

    def select_action(self, obs):
        logits = self.actor.forward(obs)
        probs  = softmax_2d(logits[None])[0]
        probs  = np.clip(probs, 1e-8, 1.0); probs /= probs.sum()
        action   = int(np.random.choice(len(probs), p=probs))   # len() not ACT_DIM
        log_prob = float(np.log(probs[action]))
        # critic is trained on normalized targets (see update()); denormalize
        # back to raw reward scale here since GAE needs values in that scale.
        raw_value = float(self.critic.forward(obs)[0])
        value     = self.ret_norm.denorm(raw_value)
        return action, log_prob, value

    @staticmethod
    def action_to_controls(idx):
        return ACTION_MAP.get(idx, ACTION_MAP[COAST])   # .get() = safe fallback

    # ── reward ─────────────────────────────────────────────

    @staticmethod
    def compute_reward(speed, rays, crashed, dt, steer=0.0):
        if crashed:
            return -15.0
        fracs    = [r['frac'] if isinstance(r, dict) else float(r)
                    for r in (rays or [])]
        min_f    = min(fracs) if fracs else 1.0
        sn       = abs(speed) / 180.0
        wall_pen = max(0.0, (0.25 - min_f) * 10.0) if min_f < 0.25 else 0.0

        # ── Steering bonus: match the best front ray, whatever it is ──
        # Previously this was two separate, conflicting terms:
        #   - turn_bon: rewarded steering magnitude UNCONDITIONALLY, even on
        #     open straight road. That meant once the agent learned to turn
        #     into corners, it had no incentive to ever straighten back out
        #     -- holding the wheel turned was free reward as long as it was
        #     still moving fast.
        #   - align_bon: rewarded matching steering to the most-open front
        #     ray, but only scaled by "urgency" (closeness to a wall). On a
        #     straight stretch right after a corner, urgency is near zero,
        #     so the "go straight" signal was too weak to counteract turn_bon.
        # Folded into one term: always reward matching the steering angle to
        # the best front ray's direction, whether that's a turn or straight
        # ahead. No urgency gating -- being centered is correct on a straight
        # exactly as much as turning is correct in a corner, so both should
        # be rewarded with the same weight, not have one dominate by default.
        steer_bon = 0.0
        if rays:
            front = rays[:len(rays)//2] if isinstance(rays[0], dict) else []
            if front:
                # Weighted average direction (weighted by frac), not argmax.
                # argmax picks a single ray, so on a uniformly-open straight
                # (all fracs ~equal) it arbitrarily locks onto whichever ray
                # happens to be first/highest by a hair -- e.g. the leftmost
                # ray -- and wrongly tells the agent "turn hard left" when
                # really every direction is equally fine. Averaging washes
                # out ties/noise and naturally centers on straights, while
                # still pointing toward the open side when one genuinely
                # exists (a real wall-avoid or curve case has an asymmetric
                # frac distribution, so the weighted average shifts toward it).
                total_w = sum(r.get('frac', 1.0) for r in front)
                if total_w > 1e-6:
                    best_angle_deg = sum(r.get('frac', 1.0) * math.degrees(r.get('rel_angle', 0.0))
                                          for r in front) / total_w
                else:
                    best_angle_deg = 0.0
                steer_deg = math.degrees(steer)
                if abs(best_angle_deg) < 5.0:
                    # best direction is straight ahead -- reward NOT steering
                    match = 1.0 - min(1.0, abs(steer_deg) / 28.0)
                else:
                    # reward steering toward the same side as the best ray,
                    # scaled by both how hard it steered and how far off
                    #-center the target direction actually is
                    target_strength = min(1.0, abs(best_angle_deg) / 28.0)
                    same_sign = (steer_deg * best_angle_deg) > 0
                    match = (min(1.0, abs(steer_deg) / 28.0) * target_strength
                             if same_sign else 0.0)
                steer_bon = 1.2 * sn * match

        return float(sn * 3.0
                     - max(0.0, (0.20 - sn) * 5.0)
                     - wall_pen + steer_bon
                     + 0.02 * dt * (1.0 + sn))

    # ── GAE ────────────────────────────────────────────────

    @staticmethod
    def compute_gae(rewards, values, dones):
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float64); last = 0.0
        for t in reversed(range(T)):
            nv   = values[t+1] if t+1 < len(values) else 0.0
            mask = 0.0 if dones[t] else 1.0
            d    = rewards[t] + GAMMA*nv*mask - values[t]
            last = d + GAMMA*LAM*mask*last
            adv[t] = last
        return adv, adv + np.array(values[:T], dtype=np.float64)

    # ── PPO update (vectorised, fast) ──────────────────────

    def update(self, buffer):
        if len(buffer) < 16:
            return 0.0, 0.0, 0.0

        obs  = np.array([b[0] for b in buffer], dtype=np.float64)
        acts = np.array([b[1] for b in buffer], dtype=np.int32)
        lpo  = np.array([b[2] for b in buffer], dtype=np.float64)
        adv, ret = self.compute_gae(
            [b[3] for b in buffer],
            [b[4] for b in buffer],
            [b[5] for b in buffer])

        # Update running return stats ONCE per rollout (not per minibatch)
        # so the normalization target stays stable across all epochs below.
        self.ret_norm.update(ret)
        ret_n = self.ret_norm.norm(ret)   # normalized returns, critic's actual target

        ta = tc = te = nu = 0.0

        for _ in range(EPOCHS):
            for bi in [np.random.permutation(len(buffer))[s:s+BATCH_SIZE]
                       for s in range(0, len(buffer), BATCH_SIZE)]:
                if len(bi) < 4: continue
                N = len(bi)
                bo, ba, bl = obs[bi], acts[bi], lpo[bi]
                bA = adv[bi]; bA = (bA - bA.mean()) / (bA.std() + 1e-8)
                bR = ret_n[bi]   # normalized target — keeps critic loss/grads in a sane range

                # ── Actor ──
                logits = self.actor.forward_batch(bo)          # (N,A)
                probs  = softmax_2d(logits)
                probs  = np.clip(probs, 1e-8, 1.0)
                probs /= probs.sum(1, keepdims=True)

                lp_new = np.log(probs[np.arange(N), ba])      # (N,)
                ratio  = np.exp(lp_new - bl)
                rc     = np.clip(ratio, 1-CLIP_EPS, 1+CLIP_EPS)
                surr   = np.minimum(ratio*bA, rc*bA)
                ent    = -np.sum(probs * np.log(probs+1e-8), 1)
                aloss  = -surr.mean() - self.entropy_coef*ent.mean()

                in_clip = ((ratio>=1-CLIP_EPS)&(ratio<=1+CLIP_EPS)).astype(np.float64)
                d_ratio = np.where(ratio*bA <= rc*bA, -bA, -bA*in_clip)
                d_lp    = ratio * d_ratio
                d_logits = -probs.copy()
                d_logits[np.arange(N), ba] += 1.0
                d_logits *= d_lp[:, None]; d_logits /= N
                d_ent = probs*(np.log(probs+1e-8)+1.0)
                d_ent -= d_ent.sum(1, keepdims=True)*probs; d_ent /= N
                d_logits -= self.entropy_coef * d_ent

                ag = clip_grads(self.actor.backward_batch(d_logits), ACTOR_CLIP)
                self.actor_opt.step(self.actor.params(), ag)

                # ── Critic ──
                vp     = self.critic.forward_batch(bo)[:,0]   # (N,)
                closs  = float(np.mean((vp - bR)**2))
                d_vp   = 2.0*(vp - bR)[:,None] / N
                cg = clip_grads(self.critic.backward_batch(d_vp), CRITIC_CLIP)
                self.critic_opt.step(self.critic.params(), cg)

                ta+=float(aloss); tc+=closs; te+=float(ent.mean()); nu+=1

        self.entropy_coef = max(ENTROPY_MIN, self.entropy_coef*ENTROPY_DECAY)
        n = max(nu, 1)
        return ta/n, tc/n, te/n

    # ── Save  (.npz) ───────────────────────────────────────

    def save_npz(self, label=None):
        ts   = time.strftime("%Y%m%d_%H%M%S")
        name = f"ppo_{label or ts}.npz"
        path = os.path.join(MODELS_DIR, name)
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
            reward_history   = np.array(self.reward_history[-100:], dtype=np.float32),
            last_actor_loss  = np.array(self.last_actor_loss,  dtype=np.float64),
            last_critic_loss = np.array(self.last_critic_loss, dtype=np.float64),
            ret_norm_mean    = np.array(self.ret_norm.mean,  dtype=np.float64),
            ret_norm_var     = np.array(self.ret_norm.var,   dtype=np.float64),
            ret_norm_count   = np.array(self.ret_norm.count, dtype=np.float64),
        )
        manifest = _load_manifest()
        manifest.append({"label": label or ts, "timestamp": ts,
                         "episode": self.episode,
                         "best_reward": round(self.best_reward, 3),
                         "file": name})
        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2)
        return path

    # Alias so old code calling save_version() still works
    def save_version(self, label=None):
        return self.save_npz(label)

    # ── Load ───────────────────────────────────────────────

    @classmethod
    def load_npz(cls, path):
        d = np.load(path, allow_pickle=False)
        ag = cls()
        for net, pfx in [(ag.actor,'a_'), (ag.critic,'c_')]:
            for k in ('W1','b1','W2','b2','W3','b3'):
                setattr(net, k, d[pfx+k].copy())
        ag.episode        = int(d['episode'])
        ag.total_steps    = int(d['total_steps'])
        ag.best_reward    = float(d['best_reward'])
        ag.entropy_coef   = float(d['entropy_coef'])
        ag.reward_history = d['reward_history'].tolist()
        # Restore last known losses so dashboard shows real values immediately
        ag.last_actor_loss  = float(d['last_actor_loss'])  if 'last_actor_loss'  in d else 0.0
        ag.last_critic_loss = float(d['last_critic_loss']) if 'last_critic_loss' in d else 0.0
        # Restore return-normalization stats (old checkpoints fall back to fresh stats)
        if 'ret_norm_mean' in d:
            ag.ret_norm.mean  = float(d['ret_norm_mean'])
            ag.ret_norm.var   = float(d['ret_norm_var'])
            ag.ret_norm.count = float(d['ret_norm_count'])
        ag.actor_opt  = Adam(ag.actor.params(),  lr=LR_ACTOR)
        ag.critic_opt = Adam(ag.critic.params(), lr=LR_CRITIC)
        return ag

    @classmethod
    def load_version(cls, path):
        if path.endswith('.npz'):
            return cls.load_npz(path)
        # Legacy JSON fallback
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

load_manifest = _load_manifest   # public alias


def get_latest_model_path():
    m = _load_manifest()
    if not m:
        return None
    latest = sorted(m, key=lambda x: x["timestamp"])[-1]
    path   = os.path.join(MODELS_DIR, latest["file"])
    return path if os.path.exists(path) else None


# ── PPO Trainer (game-loop integration) ────────────────────────

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
        self.stats = {
            "episode":        agent.episode,
            "ep_reward":      0.0,
            "best_reward":    agent.best_reward,
            "total_steps":    agent.total_steps,
            "actor_loss":     agent.last_actor_loss,   # restored from checkpoint
            "critic_loss":    agent.last_critic_loss,  # restored from checkpoint
            "entropy_coef":   agent.entropy_coef,
            "reward_history": agent.reward_history[-50:],
            "running":        True,
        }

    def step(self, rays, speed, steer, crashed, dt,
             training_state_path=None):
        obs = PPOAgent.build_obs(rays, speed, steer)

        # store previous transition
        if self.last_obs is not None:
            r = PPOAgent.compute_reward(speed, rays, crashed, dt, self.last_steer)
            self.buffer.append((self.last_obs, self.last_act,
                                self.last_lp, r, self.last_val, crashed))
            self.ep_reward += r
            self.ep_steps  += 1

        # episode end
        if crashed and self.last_obs is not None:
            self.agent.episode     += 1
            self.agent.total_steps += self.ep_steps
            self.agent.reward_history.append(round(self.ep_reward, 3))
            if self.ep_reward > self.agent.best_reward:
                self.agent.best_reward = self.ep_reward

            if len(self.buffer) >= UPDATE_EVERY:
                al, cl, _ = self.agent.update(self.buffer)
                self.buffer.clear()
                self.stats["actor_loss"]  = round(al, 6)
                self.stats["critic_loss"] = round(cl, 6)
                # Keep agent fields in sync so next save captures latest losses
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

        self.stats["ep_reward"] = round(self.ep_reward, 2)

        # write stats for dashboard
        if training_state_path:
            try:
                # Merge over existing file so save_request injected by the
                # dashboard is not wiped out before map_reader.py can read it.
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

        # next action
        action, lp, val = self.agent.select_action(obs)
        self.last_obs, self.last_act = obs, action
        self.last_lp,  self.last_val = lp, val
        self.last_steer = steer
        return PPOAgent.action_to_controls(action)