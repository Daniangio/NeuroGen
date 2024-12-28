from __future__ import annotations
from typing import List

import numpy as np
import scipy.stats as stats
from neurogen.utils import Constants as c
from enum import Enum


class NeuronState(Enum):
    RESTING = 1
    DEPOLARIZING = 2
    REPOLARIZING = 3
    REDEPOLARIZING = 4


class Neuron:
    def __init__(self, idx: str, Vr=-65.0, Ri=1.0, Cm=1.0, I_app: callable = lambda neuron: 0):
        
        self.idx = idx
        
        # Constants
        self.Vr  = Vr      # Resting membrane potential, in mV
        self.Ri  = Ri      # Axial resistance, in kΩ·cm²
        self.Cm  = Cm      # Membrane capacitance, in uF/cm^2
        self.action_potential_threshold = -55.0  # Action potential threshold, in mV
        self.action_potential_peak = 30.0  # Action potential peak, in mV
        self.refractory_period_duration = 1.0  # Refractory period duration, in ms

        # State variables
        self.V = self.Vr  # Membrane potential
        self.state = NeuronState.RESTING
        self._refractory_period_time = 0.0
        self.m = 0
        self.n = 0

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
        if self.state == NeuronState.DEPOLARIZING:
            self.m = 1 / (1 + np.exp(-(self.V - self.action_potential_threshold) / 10.))
            return c.G_NA * self.m**3 * (self.V - c.E_NA)
        return 0

    @property
    def I_K(self): # Potassium current (K⁺), in µA/cm²
        if self.state == NeuronState.REPOLARIZING:
            self.n = stats.norm.cdf(self.V, -110, 110)
            return c.G_K * self.n**4 * (self.V - c.E_K)
        return 0

    @property
    def I_L(self): # Leak current, in µA/cm²
        return c.G_LEAK * (self.V - c.E_LEAK)
    
    # Synaptic current

    @property
    def I_synin(self): # Synaptic input current, in µA/cm²
        return sum(dendrite.I_synout for dendrite in self.dendrites)
    
    @property
    def I_synout(self): # Synaptic output current, in µA/cm²
        return max(0, (self.V - self.Vr) / self.Ri)
    
    @property
    def is_refractory(self):
        return self.state != NeuronState.RESTING
    
    def update_state_variables(self, dt):
        if self.is_refractory:
            self._refractory_period_time += dt
            if self._refractory_period_time >= self.refractory_period_duration:
                self.state = NeuronState.RESTING
                self._refractory_period_time = 0.0
            elif self.V > self.action_potential_peak:
                self.state = NeuronState.REPOLARIZING
        elif self.V > self.action_potential_threshold:
            self.state = NeuronState.DEPOLARIZING
    
    def update_state_variables(self, dt):
        if self.is_refractory:
            self._refractory_period_time += dt
            if self._refractory_period_time >= self.refractory_period_duration:
                self.state = NeuronState.RESTING
                self._refractory_period_time = 0.0
            elif self.V > self.action_potential_peak:
                self.state = NeuronState.REPOLARIZING
            elif self.V < c.E_K*0.95:
                self.state = NeuronState.REDEPOLARIZING
        elif self.V > self.action_potential_threshold:
            self.state = NeuronState.DEPOLARIZING
    
    def integrate(self, dt):
        self.update_state_variables(dt)
        I_in = - self.I_Na - self.I_K - self.I_L
        if not self.is_refractory:
            I_in += self.I_app(self) + self.I_synin
        dVdt = I_in / self.Cm
        self.V +=  dVdt * dt
    
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

    def update(self, dt):
        pass