from __future__ import annotations
from typing import List

import numpy as np
from neurogen.utils import Constants as c


class Neuron:
    def __init__(self, idx: str, Vr=-65.0, Ri=1.0, Cm=1.0, I_app: callable = lambda neuron: 0):
        
        self.idx = idx
        
        # Constants
        self.Vr  = Vr      # Resting membrane potential, in mV
        self.Ri  = Ri      # Axial resistance, in kΩ·cm²
        self.Cm  = Cm      # Membrane capacitance, in uF/cm^2

        # State variables
        self.V = self.Vr  # Membrane potential
        self.m = 0.0530   # Na⁺ activation
        self.h = 0.5960   # Na⁺ inactivation
        self.n = 0.3176   # K⁺ activation

        self.dV = 0.0
        self.dm = 0.0
        self.dh = 0.0
        self.dn = 0.0

        # Dendritic dendrites
        self.dendrites: List[Neuron] = []
        self.connected_to: List[Neuron] = []

        self.I_app = I_app # External current applied to the neuron
    
    # Dendrites

    @property
    def num_dendrites(self):
        return len(self.dendrites)
    
    def spawn_dendrite(self):
        self.dendrites.append(Dendrite(idx=f"{self.idx}.{self.num_dendrites}", Vr=self.Vr, Ri=self.Ri, Cm=self.Cm))
    
    # Membrane currents

    @property
    def I_Na(self): # Sodium current (Na⁺), in µA/cm²
        return c.G_NA * self.m**3 * self.h * (self.V - c.E_NA)

    @property
    def I_K(self): # Potassium current (K⁺), in µA/cm²
        return c.G_K * self.n**4 * (self.V - c.E_K)

    @property
    def I_L(self): # Leak current, in µA/cm²
        return c.G_LEAK * (self.V - c.E_LEAK)

    # Channel gating variables

    @property
    def alpha_m(self):
        return 0.1 * (self.V + 40.0) / (1.0 - np.exp(-(self.V + 40.0) / 10.0))

    @property
    def beta_m(self):
        return 4.0 * np.exp(-(self.V + 65.0) / 18.0)

    @property
    def alpha_h(self):
        return 0.07 * np.exp(-(self.V + 65.0) / 20.0)

    @property
    def beta_h(self):
        return 1.0 / (1.0 + np.exp(-(self.V + 35.0) / 10.0))

    @property
    def alpha_n(self):
        return 0.01 * (self.V + 55.0) / (1.0 - np.exp(-(self.V + 55.0) / 10.0))

    @property
    def beta_n(self):
        return 0.125 * np.exp(-(self.V + 65) / 80.0)
    
    # Synaptic current

    @property
    def I_synin(self): # Synaptic input current, in µA/cm²
        return sum(dendrite.I_synout for dendrite in self.dendrites)
    
    @property
    def I_synout(self): # Synaptic output current, in µA/cm²
        return max(0, (self.V - self.Vr) / self.Ri)
    
    @property
    def refractory_period(self):
        return False
        # Define thresholds for h and n to determine the refractory period
        h_threshold = 0.4
        n_threshold = 0.6

        return self.h < h_threshold and self.n > n_threshold
    
    def connect(self, neuron: Neuron, dendrite_index: int = None):
        if self.is_connected_to(neuron):
            return self
        if dendrite_index == -1:
            neuron.dendrites.append(self)
            self.connected_to.append(neuron)
            return self
        if dendrite_index is None:
            neuron.spawn_dendrite()
            dendrite_index = neuron.num_dendrites - 1
        elif dendrite_index < 0 or dendrite_index >= neuron.num_dendrites:
            raise ValueError(f"Invalid dendrite index: {dendrite_index}. Number of dendrites: {neuron.num_dendrites}")
        return self.connect(neuron.dendrites[dendrite_index], -1)
    
    def is_connected_to(self, neuron: Neuron):
        return neuron in self.connected_to or any(dendrite in self.connected_to for dendrite in neuron.dendrites)

    def integrate(self, dt):
        I_in = - self.I_Na - self.I_K - self.I_L
        if not self.refractory_period:
            I_in += self.I_app(self) + self.I_synin
        dVdt = I_in / self.Cm
        self.dV = dVdt * dt

        for _ in range(10):
            self.update_gating_variables(dt)
    
    def update_gating_variables(self, dt):
        dmdt = self.alpha_m * (1.0 - self.m) - self.beta_m * self.m
        dhdt = self.alpha_h * (1.0 - self.h) - self.beta_h * self.h
        dndt = self.alpha_n * (1.0 - self.n) - self.beta_n * self.n

        self.dm = dmdt * dt
        self.dh = dhdt * dt
        self.dn = dndt * dt

        self.m += self.dm
        self.h += self.dh
        self.n += self.dn
    
    def update(self):
        self.V += self.dV


class Dendrite(Neuron):
    def __init__(self, idx, Vr=-65.0, Ri=1.0, Cm=1.0):
        super(Dendrite, self).__init__(idx, Vr=Vr, Ri=Ri, Cm=Cm, I_app=lambda neuron: 0)
    
    def update_gating_variables(self, dt):
        pass


class InputNeuron(Dendrite):
    def __init__(self, idx, I0=0.0):
        super(Dendrite, self).__init__(idx=f"Input.{idx}", Vr=None, Ri=None, Cm=None, I_app=None)
        self.I0 = I0
    
    @property
    def I_synout(self): # Synaptic output current, in µA/cm²
        return self.I0
    
    def set_current(self, I0: float):
        self.I0 = I0
    
    def integrate(self, dt):
        pass

    def update(self):
        pass