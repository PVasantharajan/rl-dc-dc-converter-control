import gym
import numpy as np
from gym import spaces
from src.converter_model import DCDCConverter
from src.pid_controller import PID
from src.battery_model import Battery

class ConverterEnv(gym.Env):
    def __init__(self, dt=1e-4, episode_len=10):
        super().__init__()
        self.dt = dt
        self.steps = int(episode_len / dt)

        # --- Components ---
        self.conv = DCDCConverter(Vin=400.0, Rload=None, dt=dt)  # Rload handled by battery now
        self.batt = Battery(dt=1e-3)
        self.pid_v = PID(Kp=29.999, Ki=220, Kd=0.00602, dt=dt, out_min=0, out_max=20)
        self.pid_i = PID(Kp=0.009, Ki=26, dt=dt) #, out_min=0, out_max=1)

        # --- Gym spaces ---
        # Observation: [Vout (=Vbat), IL, Vin, Temp, SOC]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 200, 0, 0], dtype=np.float32),
            high=np.array([100, 50, 450, 100, 1], dtype=np.float32),
            dtype=np.float32
        )

        # Action: residual current delta [-2, +2] A
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)

        self.reset()

    # ---------------------- #
    # RESET AND STATE ACCESS #
    # ---------------------- #
    def reset(self):
        self.conv.reset()
        self.batt.reset()
        self.t = 0.0
        return self._get_state()

    def _get_state(self):
        # battery voltage replaces converter output voltage
        return np.array([
            self.batt.Vbat,        # output voltage (battery terminal)
            self.conv.IL,          # inductor / charging current
            self.conv.Vin,         # input (PV) voltage
            self.batt.T,           # battery temperature
            self.batt.SOC          # state of charge
        ], dtype=np.float32)

    # ---------------------- #
    # MAIN STEP FUNCTION     #
    # ---------------------- #
    def step(self, action):
        Vref = self.batt.V_max
        headroom = 5.0
        limited_mode = self.conv.Vin < (Vref + headroom)
        target_SOC = 1.0

        if limited_mode:
            # Input voltage too low for desired output
            duty = 1.0
            self.pid_v.reset()
            self.pid_i.reset()
            Iref_pid = 0.0
            Iref_total = 0.0
        else:
            # Outer voltage loop (battery voltage)
            Iref_pid = self.pid_v.update(Vref - self.batt.Vbat)
            # RL residual current correction
            Iref_total = Iref_pid + 20.0 * float(np.clip(action, -1, 1)) #float(np.clip(action, -2, 2))
            # Inner current loop
            duty = self.pid_i.update(Iref_total - self.conv.IL)

        # --- Update converter with current battery voltage as load ---
        self.conv.update(duty, Vload=self.batt.Vbat)

        self.last_Iref_pid = Iref_pid
        self.last_Iref_total = Iref_total
        self.last_action = float(action)

        # --- Update battery state based on converter current ---
        Vbat = self.batt.update(I_charge=self.conv.IL)

        # --- Compute reward ---
        ripple = abs(Vbat - Vref) / Vref               # normalized ripple 0–0.1
        eff_penalty = 1.0 - np.clip(self.conv.efficiency, 0.0, 1.0)
        temp_penalty = np.clip((self.batt.T - 40) / 100, 0, 1)   # normalized temp term
        soc_error = (self.batt.SOC - target_SOC) ** 2

        reward = - (5 * ripple + 2 * eff_penalty + temp_penalty + 5 * soc_error)


        if limited_mode:
            reward -= 0.5  # small penalty when input voltage too low

        # --- Advance time ---
        self.t += self.dt
        done = self.t >= (self.steps * self.dt) or self.batt.SOC >= 0.999

        # Optional info dict for logging
        info = {
            "Vbat": Vbat,
            "SOC": self.batt.SOC,
            "Temp": self.batt.T,
            "Efficiency": self.conv.efficiency,
            "LimitedMode": limited_mode,
            "ripple": ripple,
            "eff_penalty": eff_penalty,
            "temp_penalty": temp_penalty
        }

        return self._get_state(), reward, done, info
