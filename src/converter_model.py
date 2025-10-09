"""
converter_model.py

Buck (DC-DC) converter model for EV/PV charging simulation.

Implements an averaged continuous-time model:

    L * dIL/dt = D * (Vin - Vload)
    C * dVout/dt = IL - Iload

For simplicity, here Vout ≈ Vload (battery terminal voltage).
This model provides current and efficiency dynamics
for control and RL experiments.
"""

import numpy as np

class DCDCConverter:
    def __init__(self, Vin=400.0, L=1e-3, C=470e-6, dt=1e-5, Rload=None):
        """
        Initialize converter parameters and state.

        Args:
            Vin (float): Input voltage [V]
            L (float): Inductance [H]
            C (float): Capacitance [F]
            dt (float): Simulation timestep [s]
            Rload (float): Optional fixed load (ignored when battery coupled)
        """
        self.Vin = Vin
        self.L = L
        self.C = C
        self.dt = dt
        self.Rload = Rload  # only used if no battery model is present

        self.reset()

    # ------------------- #
    # CORE UPDATE METHOD  #
    # ------------------- #
    def update(self, duty: float, Vload: float):
        """
        Advance converter state one step.

        Args:
            duty (float): PWM duty ratio (0–1)
            Vload (float): Output/load voltage (e.g., battery terminal)
        """
        duty = np.clip(duty, 0.0, 1.0)

        # --- Inductor current dynamics ---
        dIL = (self.Vin * duty - Vload) / self.L
        self.IL += dIL * self.dt

        # --- Output voltage follows load voltage (battery-coupled) ---
        self.Vout = Vload

        # --- Efficiency estimation (crude model) ---
        Pin = self.Vin * duty * abs(self.IL)
        Pout = self.Vout * abs(self.IL)
        self.efficiency = Pout / (Pin + 1e-9)

        # --- Time integration ---
        self.time += self.dt

    # ------------------- #
    # RESET METHOD        #
    # ------------------- #
    def reset(self):
        """Reset converter state variables."""
        self.IL = 0.0
        self.Vout = 0.0
        self.time = 0.0
        self.efficiency = 0.95
