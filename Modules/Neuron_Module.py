from __future__ import annotations
from Controller.Modules.Data_Module import ProcessBlock
from SpatialSystems.Geometric import Geo, convert_to_geo
from Controller.State_Recorder import StateSpace, json_encoder

import numpy as np
import sys
import shutil
from typing import Union
import json
import os
import pickle


class LinearRegressor:
    def __init__(self, src_data: dict = None):
        # prep handlers for input values ---------------------------
        self.states = {'input': StateSpace(),
                       'old_input': StateSpace(),
                       'stimuli': Geo(),
                       'error': Geo(),
                       'probability': Geo(),
                       'output': Geo()}

        self.rewards = {'input': StateSpace(),
                        'old_intput': StateSpace(),
                        'EV': StateSpace(),
                        'output': StateSpace(),
                        'error': StateSpace()}

        # prep handlers for internal values ---------------------------

        self.weights = {'stimuli': {},
                        'EV': {}}

        self.step = 'state output'

        if src_data is not None:
            self._overwrite_from_dict(src_data=src_data)
        return

    # Input handlers ------------------------
    def input_states(self, states: Union[dict, StateSpace]):
        """
        initialize the input states to be evaluated.

        Updates: logic_input_state and old_logic_input_state

        :param states: in units 'S' timestep 't'
        :return:
        """
        self.states['old_input'] = self.states['input'].copy()
        self.states['input'].empty()

        self.states['input']['bias'] = Geo({'+0': 1.0})
        for ky, val in states.items():
            self.states['input'][ky] = convert_to_geo(val)

        self.step = 'state input'
        return

    def input_rewards(self, rewards: Union[dict, StateSpace]):
        """
        initialize the input rewards to be evaluated.

        Updates: state_reward_input and old_state_reward_input

        :param rewards: in units 'R' timestep 't+0.5'
        :return:
        """
        self.rewards['old_input'] = self.rewards['input'].copy()
        self.rewards['input'].empty()

        self.rewards['input'] = rewards.copy()
        self.step = 'reward input'
        return

    # Internal handlers ------------------------
    def _determine_stimulus(self) -> None:
        """
        To be run after updating the input state
        :return:
        """
        self.states['stimuli'] = Geo()

        for ky1, val1 in self.states['input'].items():
            if ky1 not in self.weights['stimuli'].keys():
                self.weights['stimuli'][ky1] = StateSpace()

            for ky2, val2 in self.states['input'].items():
                if ky2 in self.weights['stimuli'][ky1].keys():
                    self.states['stimuli'] += (val1 | self.weights['stimuli'][ky1][ky2] | val2.inverse())
                else:
                    self.weights['stimuli'][ky1][ky2] = Geo()
        return

    def _determine_activation(self) -> None:
        """
        Convert stimuli V(C|I) (i.e. units ~=I*I) to C (i.e. units ~=I)
        :return:
        """
        # perform thresholding as needed to get the logical output
        self.states['probability'] = (self.states['stimuli'] ** 0.5).subset('scalars').min(1+1j).max(0+0j)
        rndm_geo = Geo({'+0': np.random.rand(), '-0': np.random.rand()})
        self.states['output'] = (self.states['probability'] < rndm_geo).subset('scalars')
        return

    def _determine_expected_values(self) -> None:
        """
        To be run after updating the state and after determining the degree of activation.

        This function determines the full expected reward values.
        :return:
        """
        self.rewards['EV'].empty()

        for rwd_type, rwd_wts in self.weights['EV'].items():
            self.rewards['EV'][rwd_type] = Geo()

            for ky1, val1 in self.states['input'].items():
                if ky1 not in self.weights['EV'][rwd_type].keys():
                    self.weights['EV'][rwd_type][ky1] = StateSpace()

                for ky2, val2 in self.states['input'].items():
                    if ky2 in self.weights['EV'][rwd_type][ky1].keys():
                        self.rewards['EV'][rwd_type] += (val1 | self.weights['EV'][rwd_type][ky1][ky2] | val2.inverse())
                    else:
                        self.weights['EV'][rwd_type][ky1][ky2] = Geo()
        return

    def _determine_reward_emission(self) -> None:
        """
        To be run after determining the expected value and degree of activation.

        This function determines the true expected value given the output activity.
        :return:
        """
        self.rewards['output'].empty()
        for rwd_type, rwd_val in self.rewards['EV'].items():
            self.rewards['output'][rwd_type] = (rwd_val | self.states['output'])['+0']
        return

    def _determine_value_error(self) -> None:
        """
        to be run after updating the reward input.
        :return:
        """
        self.rewards['error'].empty()

        for ky in set(self.rewards['input'].keys()).union(self.rewards['EV'].keys()):
            if ky in self.rewards['input'].keys():
                if ky in self.rewards['EV'].keys():
                    self.rewards['error'][ky] = self.rewards['EV'][ky] - self.rewards['input'][ky]
                else:
                    self.rewards['error'][ky] = 0.0 - self.rewards['input'][ky]
            else:
                self.rewards['error'][ky] = 0.0
        return

    def _determine_stimulus_error(self) -> None:
        """
        to be run after updating the reward error.
        If no reward errors exist, use the logical error.
        At the end, include the logical error as a negative reward.
        :return:
        """
        pure_logic_error = self.states['output'] ** 2 - self.states['probability'] ** 2

        if len(self.rewards['error'].keys()):
            self.states['error'] = Geo()

            for rwd_err_val in self.rewards['error'].values():
                self.states['error'] += rwd_err_val
            self.states['error'] *= pure_logic_error
        else:

            self.states['error'] = pure_logic_error

        self.rewards['error']['Logic Error'] = -np.abs(pure_logic_error.magnitude())
        return

    def _determine_value_weights(self) -> None:
        """
        To be run after updating the input state and reward error.

        This function is responsible for expanding partial_value_errors and value_weights to match provided rewards.
        :return:
        """

        for rwd_type, rwd_err_val in self.rewards['error'].items():
            if rwd_type not in self.weights['EV'].keys():
                self.weights['EV'][rwd_type] = StateSpace()

            for ky1, val1 in self.states['old_input'].items():
                if ky1 not in self.weights['EV'][rwd_type].keys():
                    self.weights['EV'][rwd_type][ky1] = StateSpace()

                for ky2, val2 in self.states['old_input'].items():
                    if ky2 in self.weights['EV'][rwd_type][ky1].keys():
                        self.weights['EV'][rwd_type][ky1][ky2] += val1.inverse() | rwd_err_val | val2
                    else:
                        self.weights['EV'][rwd_type][ky1][ky2] = val1.inverse() | rwd_err_val | val2

        return

    def _determine_stimulus_weights(self) -> None:
        """
        To be run after updating the input state and stimulus error.

        This function is responsible for expanding partial_stimuli_errors and stimuli_weights to match provided rewards.
        :return:
        """

        for ky1, val1 in self.states['old_input'].items():
            if ky1 not in self.weights['stimuli'].keys():
                self.weights['stimuli'][ky1] = StateSpace()

            for ky2, val2 in self.states['old_input'].items():
                if ky2 in self.weights['stimuli'][ky1].keys():
                    self.weights['stimuli'][ky1][ky2] += val1.inverse() | self.states['error'] | val2
                else:
                    self.weights['stimuli'][ky1][ky2] = val1.inverse() | self.states['error'] | val2

        return

    def process_activity(self):
        """
        Process the input to determine the degree of activation
        Weights are of units 'C' or 'C*C' (since I * W * 1 / I = W)

        Use the inputs 'I' to determine the strength of stimulation for Cell 'C' i.e. V(C|I)
        Use 'I' to determine the conditional expected value of transitioning from 'I' to 'C' i.e. EV(C|C<--I)
        Use V(C|I) to get the probability of activation P(C|I) and activation 'C'
        :return:
        """
        self._determine_stimulus()
        self._determine_expected_values()

        self._determine_activation()
        self._determine_reward_emission()
        self.step = 'Forward Processing'
        return

    def process_learning(self):
        """
        Process the errors and adjust the weights.
        :return:
        """
        self._determine_value_error()
        self._determine_stimulus_error()

        self._determine_value_weights()
        self._determine_stimulus_weights()
        self.step = 'Backwards Processing'
        return

    # Output handlers ------------------------
    def output_state(self) -> Geo:
        self.step = 'state output'
        return self.states['output'].copy()

    def reward_emission(self) -> StateSpace:
        self.step = 'reward output'
        return self.rewards['output'].copy()

    # ---- conversion methods -----
    def __dict__(self):
        nrn_dict = {'states': self.states,
                    'rewards': self.rewards,
                    'weights': self.weights,
                    'step': self.step}
        return nrn_dict

    def _overwrite_from_dict(self, src_data: dict):
        for ky in set(self.__dict__().keys()).intersection(src_data.keys()):
            if ky == 'states':
                for ky1 in set(self.states.keys()).intersection(src_data[ky].keys()):
                    if ky1 in ('input', 'old_input'):
                        self.states[ky1] = StateSpace(src_data[ky][ky1])
                    else:
                        self.states[ky1] = Geo(src_data[ky][ky1])
            elif ky == 'rewards':
                for ky1 in set(self.rewards.keys()).intersection(src_data[ky].keys()):
                    self.rewards[ky1] = StateSpace(src_data[ky][ky1])
            elif ky == 'weights':
                for ky1 in set(self.weights.keys()).intersection(src_data[ky].keys()):
                    self.weights[ky1] = {}
                    if ky1 == 'stimuli':  # weight space is only 2D
                        for ky2, val in src_data[ky][ky1].items():
                            self.weights[ky1][ky2] = StateSpace(val)
                    else:  # weight space is 3D for rewards
                        for ky2, val in src_data[ky][ky1].items():
                            self.weights[ky1][ky2] = {}
                            for ky3, val1 in val.items():
                                self.weights[ky1][ky2][ky3] = StateSpace(val1)
            elif ky == 'step':
                self.step = src_data[ky]
        return

    def __str__(self):
        return json.dumps(self.__dict__(), sort_keys=True, ensure_ascii=False, indent=4)

    def __reduce_ex__(self, protocol):
        return self.__class__, (self.__dict__(),)

    def __repr__(self) -> str:
        return json.dumps(self.to_json(), sort_keys=True, ensure_ascii=False, indent=4)

    def to_json(self):
        to_return = {}
        for ky0, val0 in self.__dict__().items():
            if isinstance(val0, (dict,)):
                to_return[ky0] = {}
                for ky, val in val0.items():
                    if hasattr(val, 'to_json'):
                        to_return[ky0][ky] = val.to_json()
                    elif hasattr(val, '__dict__'):
                        to_return[ky0][ky] = val.__dict__
                    else:
                        to_return[ky0][ky] = val
            elif hasattr(val0, 'to_json'):
                to_return[ky0] = val0.to_json()
            elif hasattr(val0, '__dict__'):
                to_return[ky0] = val0.__dict__
            else:
                to_return[ky0] = val0
        return to_return

    def save(self, src_path='.', name='state', as_json=False) -> None:
        if not os.path.exists(src_path):
            os.makedirs(src_path, exist_ok=True)

        file_path = f'{src_path}/{name}.pkl'

        with open(file_path, "wb") as a_file:
            pickle.dump(self.__dict__(), a_file)

        if as_json:
            file_path = f'{src_path}/{name}.json'
            with open(file_path, 'w') as json_file:
                json.dump(self.to_json(), json_file, indent=4, default=json_encoder)
        return

    def load(self, src_path='.', name='state') -> bool:

        file_path = f'{src_path}/{name}.pkl'
        if os.path.exists(file_path):
            with open(file_path, "rb") as a_file:
                self._overwrite_from_dict(pickle.load(a_file))
            return True
        return False
