from __future__ import annotations
# from typing import Union
import os.path
from dataclasses import dataclass
import json
import pickle

"""
StateDict dataclass allows a flat dictionary to be used, loaded and saved

save_array: take an iterable nD array and save it to the target path
load_array: load an iterable nD array from a target path

Binary storage is for speed and memory efficiency, while json is for human readability.
"""


@dataclass
class StateDict:
    def __init__(self, name):
        self.name = name
        self.flat_dict = {}

    def __repr__(self):
        return json.dumps(self.flat_dict, sort_keys=True, ensure_ascii=False, indent=4)

    def clear(self, key_list: list):
        for key in key_list:
            self.flat_dict[key] = 0.0
        return

    def empty(self):
        self.flat_dict = {}

    def remove(self, item):
        del self.flat_dict[item]

    def clear_all(self):
        for key in self.flat_dict.keys():
            self.flat_dict[key] = 0.0
        return

    def __getitem__(self, item):
        if item not in self.flat_dict:
            self.flat_dict[item] = 0.0
        return self.flat_dict[item]

    def __setitem__(self, key, value):
        self.flat_dict[key] = value
        return

    def save(self, path: str, as_bin=False):
        if not os.path.exists(f'./{path}'):
            os.mkdir(f'./{path}')

        if as_bin:
            file_path = f'./{path}/{self.name}.pkl'

            with open(file_path, "wb") as a_file:
                pickle.dump(self.flat_dict, a_file)

        else:
            file_path = f'./{path}/{self.name}.json'
            with open(file_path, 'w') as json_file:
                json.dump(self.flat_dict, json_file, indent=4)
        return

    def load(self, path: str, as_bin=False) -> bool:

        if as_bin:
            file_path = f'./{path}/{self.name}.pkl'
            if os.path.exists(file_path):
                with open(file_path, "rb") as a_file:
                    self.flat_dict = pickle.load(a_file)
                return True

        else:
            file_path = f'./{path}/{self.name}.json'
            if os.path.exists(file_path):
                with open(file_path, 'r') as json_file:
                    self.flat_dict = json.load(json_file)
                return True
        return False


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


def save_array(path: str, array, name: str, as_bin=False):
    if not os.path.exists(f'./{path}'):
        os.mkdir(f'./{path}')

    if as_bin:
        file_path = f'./{path}/{name}.pkl'

        with open(file_path, "wb") as a_file:
            pickle.dump(array, a_file)

    else:
        file_path = f'./{path}/{name}.json'

        dict_arr = array_to_dict(arr=array)
        with open(file_path, 'w') as json_file:
            json.dump(dict_arr, json_file, indent=4)
    return


def load_array(path: str, name: str, as_bin=False):
    arr_dict = []
    if as_bin:
        file_path = f'./{path}/{name}.pkl'
        if os.path.exists(file_path):
            with open(file_path, "rb") as a_file:
                arr_dict = pickle.load(a_file)

    else:
        file_path = f'./{path}/{name}.json'
        if os.path.exists(file_path):
            with open(file_path, 'r') as json_file:
                flat_dict = json.load(json_file)

            arr_dict = dict_to_array(dct=flat_dict)
    return arr_dict
