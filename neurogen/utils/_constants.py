from typing import Final

G_NA: Final[float] = 120.0 # Sodium (Na) maximum conductance, in mS/cm^2
E_NA: Final[float] = 50.0  # Sodium (Na) Nernst reversal potential, in mV

G_K: Final[float] = 36.0   # Potassium (K) maximum conductance, in mS/cm^2
E_K: Final[float] = -77.0  # Potassium (K) Nernst reversal potential, in mV

"""
Observations in hundreds of experiments on squid giant axons have shown that this leakage
current increases as the axon ages, and that the Na currents deteriorate at the same time.
These observations lead John Moore to speculate that the leakage channels may simply be
degraded Na channels that have lost their voltage sensitivity and ion selectivity.
"""
G_LEAK: Final[float] = 0.3      # Leak maximum conductance, in mS/cm^2
E_LEAK: Final[float] = -54.387  # Leak Nernst reversal potential, in mV
