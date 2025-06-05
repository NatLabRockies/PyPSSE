import datetime
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import pandas as pd
import toml
from loguru import logger

from pypsse.common import DEFAULT_PROFILE_MAPPING_FILENAME, DEFAULT_PROFILE_STORE_FILENAME, PROFILES_FOLDER
from pypsse.exceptions import InvalidParameterError
from pypsse.models import SimulationSettings
from pypsse.modes.dynamic import Dynamic
from pypsse.modes.pcm import ProductionCostModel
from pypsse.modes.snap import Snap
from pypsse.modes.static import Static
from pypsse.profile_manager.common import PROFILE_VALIDATION, ProfileTypes
from pypsse.profile_manager.profile import Profile


class ProfileManager:
    """Implentation for the profile manager for PyPSSE.
    Enables attacheing profilse to all PSSE objects and associated properties"""

    def __init__(
        self, solver: Union[ProductionCostModel, Snap, Static, Dynamic], settings: SimulationSettings, mode: str = "r+"
    ):
        """Creates an instance of the profile manager

        Args:
            solver (Union[ProductionCostModel, Snap, Static, Dynamic]): instance of simulation controller
            settings (SimulationSettings): simulation settings
            mode (str, optional): file update mode. Defaults to "r+".
        """

        self.solver = solver
        self.settings = settings

        file_path = settings.simulation.project_path / PROFILES_FOLDER / DEFAULT_PROFILE_STORE_FILENAME

        if file_path.exists():
            logger.info("Loading existing h5 store")
            self.store = h5py.File(file_path, mode)
        else:
            logger.info("Creating new h5 store")
            self.store = h5py.File(file_path, "w")
            for profile_group in [p.value for p in ProfileTypes]:
                self.store.create_group(profile_group)

    def load_data(self, file_path: Path) -> dict:
        """Load in external profile data

        Args:
            file_path (Path): path to profile mapping file

        Returns:
            dict: profile mapping dictionary
        """

        toml_dict = toml.load(file_path)
        return toml_dict

    def setup_profiles(self):
        """sets up all profiles in the profile manager

        Raises:
            Exception: raised if mapped object not found in profile DB
        """

        mapping_path = self.settings.simulation.project_path / PROFILES_FOLDER / DEFAULT_PROFILE_MAPPING_FILENAME
        if mapping_path.exists():
            self.profile_mapping = self.load_data(mapping_path)
            self.profiles = {}
            for group, profile_map in self.profile_mapping.items():
                if group in self.store:
                    grp = self.store[group]
                    for profile_name, mapping_dict in profile_map.items():
                        if profile_name in grp:
                            self.profiles[f"{group}/{profile_name}"] = Profile(
                                grp[profile_name], self.solver, mapping_dict
                            )
                        else:
                            logger.warning(rf"Group {group} \ data set {profile_name} not found in the h5 store")
                else:
                    logger.warning(f"Group {group} not found in the h5 store")
        else:
            msg = f"Profile_mapping.toml file does not exist in path {mapping_path}"
            raise Exception(msg)

    def create_dataset(
        self,
        dname: str,
        p_type: ProfileTypes,
        data: pd.DataFrame,
        start_time: datetime.datetime,
        resolution: float,
        units:str,
        info: str,
    ):
        """Create new profile datasets

        Args:
            dname (str): dateset name
            p_type (ProfileTypes): profile type
            data (pd.DataFrame): data
            start_time (datetime.datetime): profile start time
            resolution (float): profile resolution
            _ (_type_): _description_
            info (:str): profile description

        Raises:
            Exception: raised if dataset already exists
        """

        grp = self.store[p_type]
        if dname not in grp:
            sa, sa_type = self.df_to_sarray(data)
            dset = grp.create_dataset(
                dname, data=sa, chunks=True, compression="gzip", compression_opts=4, shuffle=True, dtype=sa_type
            )
            self.create_metadata(dset, start_time, resolution, data, list(data.columns), info, p_type)
        else:
            logger.error(f'Data set "{dname}" already exists in group "{p_type}".')
            msg = f'Data set "{dname}" already exists in group "{p_type}".'
            raise Exception(msg)

    def df_to_sarray(self, df: pd.DataFrame) -> [str, str]:
        """Enables data converson

        Args:
            df (pd.DataFrame): _description_

        Raises:
            SystemError: raised if unable to convert

        Returns:
            [str, str]: returns column name and datatype
        """

        def make_col_type(col_type, col):
            try:
                if "numpy.object_" in str(col_type.type):
                    maxlens = col.dropna().str.len()
                    if maxlens.any():
                        maxlen = maxlens.max().astype(int)
                        col_type = ("S%s" % maxlen, 1)
                    else:
                        col_type = "f2"
                return col.name, col_type
            except:
                raise SystemError("Unable to convert dataframe to np array")

        v = df.values
        types = df.dtypes
        numpy_struct_types = [make_col_type(types[col], df.loc[:, col]) for col in df.columns]
        dtype = np.dtype(numpy_struct_types)
        z = np.zeros(v.shape[0], dtype)
        for i, k in enumerate(z.dtype.names):
            # This is in case you have problems with the encoding, remove the if branch if not
            try:
                if dtype[i].str.startswith("|S"):
                    z[k] = df[k].str.encode("latin").astype("S")
                else:
                    z[k] = v[:, i]
            except:
                raise

        return z, dtype

    def add_profiles_from_csv(
        self,
        csv_file: Path,
        name: str,
        p_type: ProfileTypes,
        start_time: datetime.date,
        resolution_sec: float = 900,
        units: str = "",
        info: str = "",
    ):
        """enables profiles from existing csv files

        Args:
            csv_file (Path): path to profiles in a csv file
            name (str): profile name
            p_type (ProfileTypes): profile type
            start_time (datetime.date): profile start time
            resolution_sec (float, optional): profile resolution in seconds. Defaults to 900.
            units (str, optional): profile units. Defaults to "".
            info (str, optional): profile into. Defaults to "".

        Raises:
            ValueError: rasied if invalid profile name passed
            ValueError: rasied if invalid profile type passed
        """

        if p_type not in [p.value for p in ProfileTypes]:
            msg = f"Valid profile types are: {list(PROFILE_VALIDATION.keys())}"
            raise ValueError(msg)
        
        p_type = getattr(ProfileTypes,[p.name for p in ProfileTypes if p.value == p_type][0])
        logger.debug("Reading profile")
        data = pd.read_csv(csv_file)

        for c in data.columns:
            if c not in PROFILE_VALIDATION[p_type]:
                msg = f"{c} is not valid, Valid subtypes for '{p_type}' are: {PROFILE_VALIDATION[p_type]}"
                raise ValueError(msg)
        
        logger.debug("Attempting to add a profile")
        self.add_profiles(name, data, p_type, start_time, resolution_sec=resolution_sec, units=units, info=info)

    def add_profiles(
        self,
        name: str,
        data: object,
        p_type: ProfileTypes,
        start_time: datetime.date,
        resolution_sec: float = 900,
        units: str = "",
        info: str = "",
    ):
        """adds a profile to the profile manager

        Args:
            name (str): profile name
            data (object): profile data object
            p_type (ProfileTypes): profile type
            start_time (datetime.date): profile start time
            resolution_sec (float, optional): profile resolution in seconds. Defaults to 900.
            units (str, optional): profile units. Defaults to "".
            info (str, optional): profile into. Defaults to "".

        Raises:
            InvalidParameterError: raised if start_time not a datetime object
            InvalidParameterError: raised if invalid profile type passed
        """

        if type(start_time) is not datetime.datetime:
            msg = "start_time should be a python datetime object"
            raise InvalidParameterError(msg)
        
        if p_type not in ProfileTypes:
            msg = f"Valid values for p_type are {ProfileTypes.names()}"
            raise InvalidParameterError(msg)
        self.create_dataset(name, p_type, data, start_time, resolution_sec, units=units, info=info)

    def create_metadata(
        self,
        d_set: str,
        start_time: datetime.datetime,
        resolution: float,
        data: object,
        units: str,
        info: str,
        p_type: ProfileTypes,
    ):
        """adds a metadata to a new profile

        Args:
            d_set (str): dataset name
            start_time (datetime.date): profile start time
            resolution (float): profile resolution
            data (object): profile data object
            units (str): profile units
            info (str): profile info
            p_type (ProfileTypes): profile type
        """

        metadata = {
            "sTime": str(start_time),
            "eTime": str(start_time + datetime.timedelta(seconds=resolution * len(data))),
            "resTime": resolution,
            "npts": len(data),
            "min": data.min(),
            "max": data.max(),
            "mean": np.mean(data),
            "units": units,
            "info": info,
            "type": p_type,
        }
        for key, value in metadata.items():
            if isinstance(value, str):
                value_mod = np.string_(value)
            else:
                value_mod = value
            d_set.attrs[key] = value_mod

    def update(self) -> dict:
        """returns data for the current timestep for all mapped profiles

        Returns:
            dict: values for profiles at the current time step
        """

        results = {}
        for profile_name, profile_obj in self.profiles.items():
            result = profile_obj.update()
            results[profile_name] = result
            
        return results

    # def __del__(self):
    #     self.store.flush()
