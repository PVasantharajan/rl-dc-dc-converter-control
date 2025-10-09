import numpy as np

class Battery:
    def __init__(self, C_ah=50.0, V_min=48.0, V_max=60.0, R_int=0.05,
                 T_amb=25.0, C_th=100.0, R_th=2.0, dt=1e-3):
        self.C_ah = C_ah      # Capacity in Ah
        self.V_min = V_min    # Min open-circuit voltage (empty)
        self.V_max = V_max    # Max open-circuit voltage (full)
        self.R_int = R_int    # Internal resistance (ohms)
        self.T_amb = T_amb    # Ambient temp (°C)
        self.C_th = C_th      # Thermal capacitance
        self.R_th = R_th      # Thermal resistance
        self.dt = dt
        self.reset()

    def reset(self):
        self.SOC = 0.5
        self.T = self.T_amb
        self.Vbat = self.V_min + (self.V_max - self.V_min) * self.SOC
        return self.Vbat

    def Voc(self):
        # Simple linear curve — replace with nonlinear if desired
        return self.V_min + (self.V_max - self.V_min) * self.SOC

    def update(self, I_charge):
        # Update SOC (limit between 0–1)
        dSOC = (I_charge * self.dt) / (self.C_ah * 3600)
        self.SOC = np.clip(self.SOC + dSOC, 0, 1)

        # Compute open-circuit voltage
        Voc = self.Voc()

        # Internal resistance (could vary with SOC/T)
        Rint = self.R_int

        # Terminal voltage (charging → positive current)
        self.Vbat = Voc + I_charge * Rint

        # Thermal model
        P_loss = (I_charge ** 2) * Rint
        dT = (P_loss / self.C_th - (self.T - self.T_amb) / (self.R_th * self.C_th)) * self.dt
        self.T += dT

        return self.Vbat
