"""

Use these blocks in parallel or series to modify inputs into a desired form.

Process Block:      --[]--

Constant Block:     -|[x]--
Proportional Block: --[x]--
Derivative Block:   --[d/dt]--
Integral Block:     --[+x*dt]--
Delay Block:        --[1/ds]--

Sum Block:          ==[x+y]--
Product Block:      ==[x*y]--

Clip Block:         --[^-v]-- todo
Default To Block:      --[<o>]-- todo

Time Block:         -|[t]-- todo
Counter Block:      --[+dt|x]-- todo
"""
import numpy as np
from copy import deepcopy


class ProcessBlock:
    def __init__(self):
        self.input_data = None
        self.output_data = None
        self.flags = {}

    def input(self, data):
        self.input_data = data
        return

    def process(self, dt=1.0):
        self.output_data = self.input_data
        return

    def output(self):
        return deepcopy(self.output_data)

    def is_not_used(self):
        pass


class ConstantBlock(ProcessBlock):
    """
    Provide the set constant value when requested.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.input_data = kwargs.get('value', 1.0)

    def input(self, data):
        self.is_not_used()
        return


class ProportionalBlock(ProcessBlock):
    """
    Scale the input by a constant value.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.scale = kwargs.get('scale', 1.0)

    def process(self, dt=1.0):
        self.output_data = self.scale * self.input_data


class DerivativeBlock(ProcessBlock):
    """
    Give the derivative of the input from its last time step (dx/dt).
    """
    def __init__(self):
        super().__init__()
        self.old_input_data = None

    def input(self, data):
        if self.input_data is None:
            self.old_input_data = data
        else:
            self.old_input_data = self.input_data
        self.input_data = data
        return

    def process(self, dt=1.0):
        self.output_data = (self.input_data - self.old_input_data) / dt


class IntegralBlock(ProcessBlock):
    """
    Give the integral of the input from its last time step (dx/dt).
    """

    def __init__(self):
        super().__init__()

    def process(self, dt=1.0):
        if self.output_data is None:
            self.output_data = self.input_data * dt
        else:
            self.output_data += self.input_data * dt


class SumBlock(ProcessBlock):
    """
    Sum the input into a single value.
    """
    def __init__(self, **kwargs):
        super().__init__()

    def process(self, dt=1.0):
        self.output_data = np.sum(self.input_data)


class ProductBlock(ProcessBlock):
    """
    Multiply the input into a single value.
    """
    def __init__(self, **kwargs):
        super().__init__()

    def process(self, dt=1.0):
        self.output_data = np.prod(self.input_data)
