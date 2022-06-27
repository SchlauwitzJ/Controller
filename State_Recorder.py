from __future__ import annotations
from typing import Union
from Controller.Modules.Data_Module import ProcessBlock
from SpatialSystems.Geometric import Geo
import os.path
from dataclasses import dataclass
import json
import pickle
import glob
import numpy as np
from copy import deepcopy

"""
StateDict dataclass allows a flat dictionary to be used, loaded and saved

save_array: take an iterable nD array and save it to the target path
load_array: load an iterable nD array from a target path

Binary storage is for speed and memory efficiency, while json is for human readability.
"""


def json_encoder(obj):
    if hasattr(obj, 'to_json'):
        return obj.to_json()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


@dataclass
class StateSpace:
    def __init__(self, src: Union[dict, StateSpace] = None):
        self.__set = {}
        if src is not None:
            for ky, val in src.items():
                if isinstance(val, (dict, Geo)):
                    self.__set[ky] = Geo(val)
                else:
                    self.__set[ky] = val

    # ---- defined dictionary-like behaviors ----------
    def clear(self, key_list: Union[list, set]):
        for key in key_list:
            self.__set[key] = 0.0
        return

    def clear_all(self):
        for key in self.__set.keys():
            self.__set[key] = 0.0
        return

    def empty(self):
        self.__set = {}

    def __delattr__(self, item) -> None:
        del self.__set[item]

    def keys(self):
        return self.__set.keys()

    def values(self):
        return self.__set.values()

    def items(self):
        return self.__set.items()

    def __getitem__(self, item):
        if item not in self.__set:
            self.__set[item] = 0.0
        return self.__set[item]

    def get(self, key, default):
        if key not in self.keys():
            return default
        return self[key]

    def __setitem__(self, key, value):
        self.__set[key] = value
        return

    def copy(self):
        cpy_dict = StateSpace()
        for ky, val in self.items():
            if hasattr(val, 'copy'):
                cpy_dict[ky] = val.copy()
            else:
                cpy_dict[ky] = deepcopy(val)
        return cpy_dict

    def __iter__(self) -> iter:
        return iter([(ky, self[ky]) for ky in self.keys()])

    def __bool__(self):
        return len(self.keys()) != 0

    # ---- conversion methods -----
    def __dict__(self):
        return {ky: val for ky, val in self}

    def __str__(self):
        return json.dumps(self.to_json(), sort_keys=True, ensure_ascii=False, indent=4)

    def __reduce_ex__(self, protocol):
        return self.__class__, (self.__set,)

    def __repr__(self) -> str:
        return self.__str__()

    def to_json(self):
        to_return = {}
        for ky, val in self.__set.items():
            if hasattr(val, 'to_json'):
                to_return[ky] = val.to_json()
            elif hasattr(val, '__dict__'):
                to_return[ky] = val.__dict__
            else:
                to_return[ky] = val
        return to_return

    def load(self, src_path='.', name='state') -> bool:

        file_path = f'{src_path}/{name}.pkl'
        if os.path.exists(file_path):
            with open(file_path, "rb") as a_file:
                self.__set = pickle.load(a_file)
            return True
        return False

    def save(self, src_path='.', name='state', as_json=False) -> None:
        if not os.path.exists(src_path):
            os.makedirs(src_path, exist_ok=True)

        file_path = f'{src_path}/{name}.pkl'

        with open(file_path, "wb") as a_file:
            pickle.dump(self.__set, a_file)

        if as_json:
            file_path = f'{src_path}/{name}.json'
            with open(file_path, 'w') as json_file:
                json.dump(self.to_json(), json_file, indent=4, default=json_encoder)
        return

    # ---- operations -------
    def __and__(self, other: StateSpace) -> StateSpace:
        """
        Intersection of two sets, multiplying matching dimension keys together.

        the format for ths space state must match else it will try to multiply the elements by default.
        :param other:
        :return:
        """
        rslt = StateSpace()
        for ky in set(self.keys()).intersection(other.keys()):
            if isinstance(self[ky], (Geo,)) or isinstance(other[ky], (Geo,)):
                rslt[ky] = self[ky] | other[ky]
            elif isinstance(self[ky], (StateSpace,)) and isinstance(other[ky], (StateSpace,)):
                rslt[ky] = self[ky] & other[ky]
            else:
                rslt[ky] = self[ky] * other[ky]
        return rslt

    def __or__(self, other: StateSpace) -> StateSpace:
        """
        Union of two sets, adding matching dimension keys together.
        :param other:
        :return:
        """
        rslt = StateSpace()
        for ky in set(self.keys()).union(other.keys()):
            if ky in self.keys():
                if ky in other.keys():
                    if isinstance(self[ky], (Geo,)) or isinstance(other[ky], (Geo,)):
                        rslt[ky] = self[ky] ^ other[ky]
                    else:
                        rslt[ky] = self[ky] + other[ky]
                else:
                    rslt[ky] = self[ky]
            else:
                rslt[ky] = other[ky]
        return rslt

    def __xor__(self, other: StateSpace) -> StateSpace:
        """
        Elements of this set excluded from the other, xor-ing the elements that match.
        :param other:
        :return:
        """
        rslt = StateSpace()
        for ky in set(self.keys()).union(other.keys()):
            if ky in self.keys():
                if ky in other.keys():
                    if isinstance(self[ky], (Geo,)) or isinstance(other[ky], (Geo,)):
                        rslt[ky] = self[ky] | other[ky]
                    else:
                        rslt[ky] = self[ky] + other[ky] - 2 * self[ky] * other[ky]
                else:
                    rslt[ky] = self[ky]
            else:
                rslt[ky] = other[ky]
        return rslt


def del_saves(src_path='.', name='state'):
    if os.path.exists(src_path + '/'):
        filelst = glob.glob(f'{src_path}/{name}.*')
        for filename in filelst:
            filename = os.path.normpath(filename)
            try:
                os.remove(filename)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (filename, e))
    return

# other stuff -----------------------------------
def array_to_dict(arr):
    try:
        arr_dict = {}
        for ind, ele in enumerate(arr):
            arr_dict[ind] = array_to_dict(arr=ele)
        return arr_dict
    except TypeError as te:
        return arr


def dict_to_array(dct: dict):
    try:
        arr_dict = [''] * len(dct)
        for ind, ele in dct.items():
            arr_dict[ind] = dict_to_array(dct=ele)
        return arr_dict
    except TypeError as te:
        return dct


def save_array(array, name: str, path='./', as_bin=False):
    if not os.path.exists(path):
        os.mkdir(path)

    if as_bin:
        file_path = f'{path}/{name}.pkl'

        with open(file_path, "wb") as a_file:
            pickle.dump(array, a_file)

    else:
        file_path = f'{path}/{name}.json'

        dict_arr = array_to_dict(arr=array)
        with open(file_path, 'w') as json_file:
            json.dump(dict_arr, json_file, indent=4)
    return


def load_array(path: str, name: str, as_bin=False):
    arr_dict = []
    if as_bin:
        file_path = f'{path}/{name}.pkl'
        if os.path.exists(file_path):
            with open(file_path, "rb") as a_file:
                arr_dict = pickle.load(a_file)

    else:
        file_path = f'{path}/{name}.json'
        if os.path.exists(file_path):
            with open(file_path, 'r') as json_file:
                flat_dict = json.load(json_file)

            arr_dict = dict_to_array(dct=flat_dict)
    return arr_dict
