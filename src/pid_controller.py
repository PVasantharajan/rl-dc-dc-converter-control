import numpy as np

class PID:
    def __init__(self, Kp, Ki, Kd=0, dt=1e-5, out_min=0, out_max=1):
        self.Kp, self.Ki, self.Kd, self.dt = Kp, Ki, Kd, dt
        self.i_term, self.prev_e = 0.0, 0.0
        self.out_min, self.out_max = out_min, out_max
    def update(self, e):
        self.i_term += e * self.Ki * self.dt
        d = (e - self.prev_e) / self.dt
        u = self.Kp*e + self.i_term + self.Kd*d
        self.prev_e = e
        return np.clip(u, self.out_min, self.out_max)
    def reset(self):
        self.i_term = 0
        self.prev_e = 0
