# Faisla — PPO Car Simulator: Architecture

## 1. Folder structure

```mermaid
flowchart LR
    ROOT["Faisla/"]

    subgraph SRC["Python source"]
        ENVPY["environment.py<br/>track editor: draw, smooth, save JSON"]
        MRPY["map_reader.py<br/>simulator + RL environment loop<br/>visual and --headless modes"]
        MODELPY["model.py<br/>NumPy PPO: MLP, Adam, RunningNorm,<br/>PPOAgent, PPOTrainer, checkpoints"]
        TESTPY["test_model.py<br/>greedy inference viewer"]
        DASHPY["dashboard.py<br/>live training dashboard"]
        CARPY["car.py<br/>F1 close-up view + telemetry"]
    end

    subgraph NB["GPU notebooks - PyTorch"]
        NB1["train_gpu.ipynb<br/>128 cars, single map"]
        NB2["train_gpu_multimap.ipynb<br/>256 cars, 400 generated tracks,<br/>held-out test maps"]
    end

    subgraph TRACKS["Track files - JSON"]
        T1["first.json, circle.json, circuit.json"]
        T2["second.json, training.json, diff.json"]
        T3["weird.json, f1.json, sunflower.json"]
    end

    subgraph MODELS["models/"]
        MAN["manifest.json<br/>label, timestamp, episode, best_reward, file"]
        NPZ["ppo_*.npz checkpoints<br/>final_circle.npz, multi_final.npz"]
    end

    subgraph MISC["Other"]
        ROOTNPZ["final.npz, multi_final.npz<br/>notebook exports"]
        REQ["requirements.txt"]
        GI[".gitignore, .git/"]
    end

    ROOT --> SRC
    ROOT --> NB
    ROOT --> TRACKS
    ROOT --> MODELS
    ROOT --> MISC

    ENVPY -->|"writes"| TRACKS
    TRACKS -->|"read by"| MRPY
    TRACKS -->|"read by"| TESTPY
    TRACKS -->|"read by"| NB
    MRPY -->|"imports"| MODELPY
    TESTPY -->|"imports sim helpers"| MRPY
    MRPY -->|"launches subprocess"| DASHPY
    MRPY -->|"launches subprocess"| CARPY
    MODELPY -->|"save_npz / load_npz"| MODELS
    NB -->|"export_npz"| ROOTNPZ
```

## 2. How the algorithm works

Three zoom levels. Each box names the real function in the code.

### 2a. The main loop (map_reader.py)

One frame = one decision. Physics runs 6 times per frame; the action is held in between.

```mermaid
flowchart TD
    A["load_map(path)"] --> B["spawn_car_on_road()"]
    B --> C{"--checkpoint given?"}
    C -->|yes| D["PPOAgent.load_npz()"]
    C -->|no| E["PPOAgent()"]
    D --> F["PPOTrainer(agent)"]
    E --> F
    F --> G["New frame: substep 1 of 6"]
    G --> H["compute_rays()<br/>20 distance rays"]
    H --> I{"First substep,<br/>or just crashed?"}
    I -->|yes| J["track_state()<br/>for the reward"]
    J --> K["PPOTrainer.step()<br/>returns throttle, brake, steer"]
    K --> L["MapCar.update()<br/>move the car"]
    I -->|no: keep last action| L
    L --> M["enforce_road_boundary()"]
    M --> N{"Hit a wall, or<br/>stuck for 5 s?"}
    N -->|yes| O["signal_crash = True<br/>random_ai_car(): respawn"]
    N -->|no| P{"6 substeps done?"}
    O --> P
    P -->|no| H
    P -->|yes| G
```

### 2b. One decision (model.py, PPOTrainer.step)

```mermaid
flowchart TD
    A["build_obs()<br/>20 rays + speed + steer = 22 numbers"] --> B{"Is there a previous action?"}
    B -->|no| H
    B -->|yes| C["compute_reward()<br/>speed along road - off-centre penalty<br/>or -15 on crash"]
    C --> D{"30 s limit reached?"}
    D -->|yes| E["reward += gamma x V(state)<br/>time-out is not a failure"]
    D -->|no| F
    E --> F["buffer.append()<br/>obs, action, log-prob, reward, value, done"]
    F --> G{"Buffer has 4096 steps?"}
    G -->|yes| U["PPOAgent.update()<br/>then buffer.clear()"]
    G -->|no| H
    U --> H["select_action()<br/>softmax, sample an action,<br/>store log-prob and V"]
    H --> I["action_to_controls()<br/>action -> throttle, brake, steer"]
```

### 2c. Learning from the buffer (model.py, PPOAgent.update)

```mermaid
flowchart TD
    A["compute_gae()<br/>advantages + target returns"] --> B["ret_norm.update() / norm()<br/>scale the returns"]
    B --> C["Normalise advantages<br/>mean 0, std 1"]
    C --> D["Epoch: shuffle, split into 8 minibatches"]
    D --> E["actor.forward_batch() + softmax_2d()<br/>new log-probs"]
    E --> F["ratio = new prob / old prob<br/>clip to 0.8 - 1.2"]
    F --> G["Gradient of clipped loss<br/>+ entropy bonus"]
    G --> H["actor.backward_batch()"]
    H --> I["critic.forward_batch() + backward_batch()<br/>0.5 x MSE vs returns"]
    I --> J["clip_grads()<br/>total norm max 0.5"]
    J --> K["Adam.step()<br/>actor and critic"]
    K --> L{"More minibatches?"}
    L -->|yes| E
    L -->|no| M{"KL > 0.045 or 4 epochs done?"}
    M -->|no| D
    M -->|yes| N["Shrink entropy_coef<br/>min 0.01"]
```

Not shown, to keep the diagrams readable:
- After 30 s, `PPOTrainer.step()` sets `reset_requested`. The main loop then calls `random_ai_car()` and calls `step()` again for the new episode.
- Every 150 episodes the main loop calls `PPOAgent.save_npz()` and moves the car to a new random spawn.
- `PPOAgent.update()` also runs mid-episode if the buffer fills up. It then uses V(current state) as `last_val` in `compute_gae()`.
