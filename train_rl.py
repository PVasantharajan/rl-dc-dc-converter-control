from stable_baselines3 import PPO
from src.converter_env import ConverterEnv
import matplotlib.pyplot as plt

# --- Environment & model setup ---
env = ConverterEnv()
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
model.learn(total_timesteps=200_000)
model.save("converter_rl")

# --- Evaluation storage ---
Vhist, Ihist, RewardHist, RippleHist, EffHist, TempHist, SOChist = [], [], [], [], [], [], []

obs = env.reset()

# --- Run until SOC >= 0.99 or max steps reached ---
for _ in range(300_000):  # enough steps for a full charge
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    Vhist.append(env.batt.Vbat)
    Ihist.append(env.conv.IL)
    RewardHist.append(reward)
    RippleHist.append(info["ripple"])
    EffHist.append(1 - info["eff_penalty"])  # convert back to efficiency
    TempHist.append(info["Temp"])
    SOChist.append(info["SOC"])

    # Stop once battery is ~fully charged
    if env.batt.SOC >= 0.99:
        print(f"Charging complete at t={env.t:.2f}s, SOC={env.batt.SOC:.3f}")
        break


# --- Plot results ---
plt.figure(figsize=(10,8))

plt.subplot(6,1,1)
plt.plot(Vhist)
plt.title("Battery Voltage (Vbat)")
plt.ylabel("V [V]")

plt.subplot(6,1,2)
plt.plot(SOChist, color='green')
plt.ylabel("SOC")
plt.title("State of Charge (SOC)")

plt.subplot(6,1,3)
plt.plot(RippleHist, label="Voltage ripple term")
plt.plot(EffHist, label="Efficiency (1 - eff_penalty)")
plt.legend()
plt.ylabel("Efficiency / Ripple")

plt.subplot(6,1,4)
plt.plot(TempHist, label="Battery temperature [°C]")
plt.legend()
plt.ylabel("Temp [°C]")

plt.subplot(6,1,5)
plt.plot(RewardHist, label="Total reward")
plt.legend()
plt.ylabel("Reward")
plt.xlabel("Steps")

plt.subplot(6,1,6)
plt.plot(Ihist, label="Charging Current")
plt.legend()
plt.ylabel("Icharg [A]")
plt.xlabel("Steps")

plt.tight_layout()
plt.show()
