# Reinforcement Learning-Based Fast Battery Charging Control

## Overview

This project implements a reinforcement learning (RL) controller for a DC–DC converter–battery charging system.  
The design combines a two-loop PID control structure (voltage and current loops) with a residual PPO (Proximal Policy Optimization) agent that fine-tunes the current reference for optimal charging performance.

The objective is to charge the battery to full state of charge (SOC → 1.0) as efficiently and safely as possible while minimizing voltage ripple, losses, and temperature rise.

---

## Control Architecture

The system uses a hierarchical control approach consisting of two classical PID loops and a learning-based residual agent:

```
          PPO RL Agent (Residual)
                 │
                 ▼
       PID Current Controller (inner loop)
                 │
                 ▼
       PID Voltage Controller (outer loop)
                 │
                 ▼
       DC–DC Converter  →  Battery Model
```

| Variable | Objective | Control Loop |
|-----------|------------|--------------|
| Battery Voltage (Vbat) | Track reference (≈60 V) | Outer PID loop |
| Inductor Current (IL)  | Regulate current flow | Inner PID loop |
| State of Charge (SOC)  | Reach 100% safely | RL residual control |
| Temperature (T)        | Maintain safe levels | Reward penalty |
| Efficiency (η)         | Maximize | Reward term |

---

## Reward Function

The reward encourages fast charging progress while maintaining converter efficiency and temperature within limits:

\[
R = +50 \cdot ΔSOC - (5·ripple + 3·temp\_penalty + 3·eff\_penalty)
\]

where:

- `ΔSOC` = incremental increase in battery SOC per step  
- `ripple` = normalized voltage deviation, |Vbat − Vref| / Vref  
- `temp_penalty` = scaled penalty above 40 °C  
- `eff_penalty` = loss-based efficiency term  

This formulation directly rewards SOC progress and penalizes instability, heat, and inefficiency.

---

## Simulation Setup

| Parameter | Value |
|------------|--------|
| Simulation timestep (`dt`) | 1e-3 s |
| Episode length | 200 s |
| Battery capacity | 50 Ah |
| Input voltage | 400 V |
| Converter type | Buck (averaged model) |
| Controllers | PID (voltage & current) + PPO residual |
| Training steps | 200 000 |
| Evaluation | Until SOC ≥ 0.99 |

---

## Results

The figure below shows the final evaluation of the trained controller.

![Charging Results](./results.png)

### Observations

- Stable voltage regulation near 55–60 V  
- Steady charging current of approximately 20 A (≈ 0.4 C rate)  
- SOC increases gradually from 50 % to 52 % over the simulation window  
- Battery temperature rises from 25 °C to approximately 40 °C  
- Converter efficiency remains near unity  
- Reward stabilizes, indicating consistent policy behavior

This demonstrates stable, efficient, and safe battery charging performance using the hybrid PID + RL control strategy.

---

## Project Structure

```
src/
 ├── converter_model.py       # Averaged buck converter model
 ├── pid_controller.py        # Inner and outer PID controllers
 ├── battery_model.py         # Electrothermal battery model
 ├── converter_env.py         # Gym-compatible environment
notebooks/
 ├── battery_test.ipynb       # Battery model validation
 ├── converter_pid_test.ipynb # PID control validation
 ├── rl_training.ipynb        # PPO training and evaluation
logs/
 └── tensorboard/             # PPO training logs
results.png                   # Example result plot
```

---

## How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train PPO agent**
   ```python
   from stable_baselines3 import PPO
   from src.converter_env import ConverterEnv

   env = ConverterEnv()
   model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
   model.learn(total_timesteps=200_000)
   model.save("converter_rl")
   ```

3. **Evaluate the trained model**
   ```python
   obs = env.reset()
   done = False
   while not done:
       action, _ = model.predict(obs, deterministic=True)
       obs, reward, done, info = env.step(action)
   ```

---

## Summary

- Demonstrated a hybrid control architecture combining PID and RL for DC–DC converter-based battery charging.  
- Validated converter and battery dynamics through standalone simulations.  
- Achieved stable voltage control, efficient charging, and safe thermal behavior.  
- Framework is extensible for CC–CV transitions, adaptive charge profiles, and real-time optimization.
