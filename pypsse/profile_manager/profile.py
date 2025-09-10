import copy
import datetime

import numpy as np
from loguru import logger

from pypsse.profile_manager.common import PROFILE_VALIDATION


class Profile:
    "Class defination for single profile"

    DEFAULT_SETTINGS = {"multiplier": 1, "normalize": False, "interpolate": False}

    def __init__(self, profile_obj, solver, mapping_dict, buffer_size=10, neglect_year=True):
        self.value_settings = {f"{x['bus']}__{x['id']}": {**self.DEFAULT_SETTINGS, **x} for x in mapping_dict}
        self.mapping_dict = mapping_dict
        self.buffer_size = buffer_size
        self.buffer = np.zeros(buffer_size)
        self.profile = profile_obj
        self.neglect_year = neglect_year
        self.solver = solver
        self.attrs = self.profile.attrs
        s = self.attrs["sTime"].decode()
        stime = s if "." in s else s + ".00"
        e = self.attrs["eTime"].decode()
        etime = e if "." in e else e + ".00"
        self.stime = datetime.datetime.strptime(stime, "%Y-%m-%d %H:%M:%S.%f").astimezone(None)
        self.etime = datetime.datetime.strptime(etime, "%Y-%m-%d %H:%M:%S.%f").astimezone(None)
        self.sim_res = self.solver.get_step_size_cec()
        self.time = copy.deepcopy(self.solver.get_time())
        self.columns = self.attrs["units"]
        self.dtype = self.attrs["type"].decode()

    def update(self, update_object_properties=True):
        "Returns value at the current timestep in the given profile"
        self.time = copy.deepcopy(self.solver.get_time()).astimezone(None)
        if self.time < self.stime or self.time > self.etime:
            value = np.array([0] * len(self.profile[0]))
            value1 = np.array([0] * len(self.profile[0]))
        else:
            dt = (self.time - self.stime).total_seconds()
            n = int(dt / self.attrs["resTime"])
            value = np.array(list(self.profile[n]))
            try:
                valuen1 = np.array(list(self.profile[n + 1]))
            except Exception as _:
                valuen1 = value

            dt2 = (
                self.time - (self.stime + datetime.timedelta(seconds=int(n * self.attrs["resTime"])))
            ).total_seconds()
            value1 = value + (valuen1 - value) * dt2 / self.attrs["resTime"]

        if update_object_properties:
            for obj_name in self.value_settings:
                bus, object_id = obj_name.split("__")
                if self.value_settings[obj_name]["interpolate"]:
                    value = value1
                mult = self.value_settings[obj_name]["multiplier"]
                if isinstance(mult, list):
                    mult = np.array(mult)
                if self.value_settings[obj_name]["normalize"]:
                    value_f = value / self.attrs["max"] * mult
                else:
                    value_f = value * mult
                value_f = self.fill_missing_values(value_f)
                # if self.dtype == "Machine":
                #     value_f["realar1"] = value_f["realar1"]
                self.solver.update_object(self.dtype, bus, object_id, value_f)
                logger.info(f"Object updated: {object_id}.{bus}.{self.dtype}={value_f}")
        return value

    def fill_missing_values(self, value):
        "Fixes issues in profile data"
        idx = [f"realar{PROFILE_VALIDATION[self.dtype].index(c) + 1}" for c in self.columns]
        x = dict(zip(idx, list(value)))
        return x
