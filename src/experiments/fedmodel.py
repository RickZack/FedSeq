from copy import deepcopy
from src.utils.utils import save_pickle
from typing import Dict, Iterable, Tuple


class Param:
    def __init__(self, key: str, value: str or Dict):
        self.key = key
        self.value = value

    def update(self, other):
        assert type(self.value) == type(other.value), "Update Mismatch"
        assert type(self.value) == dict, "Invalid update of a value"
        self.value.update(other.value)

    def __repr__(self):
        return f"{self.value}"

    def as_cmd_arg(self):
        return "'" + f"{self.key}={self.value}".replace("'", "").replace(' ', '') + "'"

    def __iter__(self):
        return self.__dummy_iterator()

    def __dummy_iterator(self):
        yield self


class MultiParam:
    def __init__(self, param: str, values: Iterable or Tuple[str, Iterable] or Dict[str, Iterable]):
        self.key = param
        self.values = values

    @staticmethod
    def key(param: str, values: Iterable):
        return MultiParam(param, values)

    @staticmethod
    def dict(dict_name: str, dict_param: Tuple[str, Iterable] or Dict[str, Iterable]):
        return MultiParam(dict_name, dict_param)

    def __iter__(self):
        return multi_param_iterator(self)


def multi_param_iterator(param: MultiParam):
    if type(param.values) == tuple:
        (key, valList) = param.values
        for value in valList:
            yield Param(param.key, {key: value})
    elif type(param.values) == dict:
        keys, values = param.values.keys(), param.values.values()
        values_len = [len(v) for v in values]
        assert len(values_len) != 0 and all(
            l == values_len[0] for l in values_len), "Error in dict values, dimension mismatch or empty"
        for i in range(values_len[0]):
            d = dict()
            for k, v in zip(keys, values):
                d[k] = v[i]
            yield Param(param.key, d)
    else:
        for value in param.values:
            yield Param(param.key, value)


class FedModel:
    def __init__(self, params: Dict[str, Param]):
        self.__params = params
        self.success: bool or None = None

    def update(self, key: str, value: Param):
        if key not in self.__params:
            self.__params[key] = deepcopy(value)
        else:
            self.__params[key].update(value)

    def set_param(self, key: str, value: Param):
        self.__params[key] = value

    def save_to_file(self, filename: str):
        d = dict([(key, param.value) for key, param in self.__params.items()])
        save_pickle(d, filename)

    def set_success(self, success: bool):
        self.success = success

    @property
    def params(self) -> Dict[str, Param]:
        return self.__params

    def merge(self, other):
        source, dest = other.__params, self.__params
        for key in source.keys():
            if key in dest.keys():
                assert type(dest[key]) == type(source[key]), "Values mismatch in recursive update"
                if hasattr(dest[key], 'update'):
                    dest[key].update(source[key])
                    continue
            dest[key] = source[key]

    def get_keyvalue_list(self):
        keyvalues = []
        for key, param in self.__params.items():
            if type(param.value) == dict:
                keyvalues.extend([f"{key}.{subKey}={subValue}" for subKey, subValue in param.value.items()])
            else:
                keyvalues.append(f"{key}={param.value}")
        return keyvalues
