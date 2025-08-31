# Here we make the mock instruments with qcodes for convenience

from qcodes.instrument_drivers.mock_instruments import DummyInstrument
from qcodes.parameters import MultiParameter, Parameter
import time
import numpy as np


class SineWithNoise(Parameter):
    def __init__(self, name='measure_param', amplitude=1.0, noise_level=0.1):
        super().__init__(name=name, label='Sine with noise', unit='V')
        self.amplitude = amplitude
        self.noise_level = noise_level
        self._dac = dac

    def get_raw(self):
        # Get current DAC values
        x = self._dac.ch1.get()
        freq = self._dac.ch2.get() / 10
          # Scale ch2 to control frequency  
        # Generate sine with frequency controlled by ch2
        clean_sine = self.amplitude * np.cos(freq * x * 2 * np.pi / 500)
        noise = np.random.normal(0, self.noise_level)
        return clean_sine + noise


class SineWithNoiseExponential(Parameter):
    def __init__(self, name='measure_param_exponential', amplitude=1.0, noise_level=0.1, decay_constant=100):
        super().__init__(name=name, label='Sine with noise and exponential decay', unit='V')
        self.amplitude = amplitude
        self.noise_level = noise_level
        self.decay_constant = decay_constant
        self._dac = dac

    def get_raw(self):
        # Get current DAC values
        x = self._dac.ch1.get()
        freq = self._dac.ch2.get() / 10
        # Scale ch2 to control frequency  
        # Generate sine with frequency controlled by ch2
        clean_sine = self.amplitude * np.cos(freq * x * 2 * np.pi / 500)
        noise = np.random.normal(0, self.noise_level)
        # Apply exponential decay envelope
        exponential_envelope = np.exp(-abs(x) / self.decay_constant)
        return (clean_sine + noise) * exponential_envelope




try:
    dac = DummyInstrument('dac', gates=['ch1', 'ch2'])
except: pass

dac.ch1.set(0)
dac.ch2.set(50)
param = Parameter('counter')
measure_param_1 = SineWithNoise(name='measure_param_1', amplitude=1.0, noise_level=0.1)
measure_param_2 = SineWithNoiseExponential(name='measure_param_2', amplitude=1.0, noise_level=0.1, decay_constant=100)
