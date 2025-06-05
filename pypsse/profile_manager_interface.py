from datetime import timedelta
from pathlib import Path

from loguru import logger
import pandas as pd
import numpy as np
import h5py
import toml

from pypsse.common import DEFAULT_PROFILE_MAPPING_FILENAME, DEFAULT_PROFILE_STORE_FILENAME, PROFILES_FOLDER, DEFAULT_PROFILE_EXPORT_FILE
from pypsse.models import SimulationSettings



class ProfileManagerInterface:

    def __init__(self, settings: SimulationSettings):
        assert settings.simulation.use_profile_manager, "Profile manager is not enabled in the simulation settings"
        
        project_path = settings.simulation.project_path
        store_file = project_path / PROFILES_FOLDER / DEFAULT_PROFILE_STORE_FILENAME
        toml_file = project_path / PROFILES_FOLDER / DEFAULT_PROFILE_MAPPING_FILENAME
        self._store = h5py.File(store_file, 'r')
        self._toml_dict = toml.load(toml_file)
        self._start_time = settings.simulation.start_time
        self._simulation_duration = settings.simulation.simulation_time
        self._end_time = self._start_time + self._simulation_duration
        self._export_file = project_path / PROFILES_FOLDER / DEFAULT_PROFILE_EXPORT_FILE
    
    @classmethod
    def from_setting_files(cls, simulation_settings_file: Path):
        simulation_settiings = toml.load(simulation_settings_file)
        simulation_settiings = SimulationSettings(**simulation_settiings)
        return cls(simulation_settiings)

    def get_profiles(self) -> list:
        all_datasets = []
        for model_type, model_info in self._toml_dict.items():
            logger.info(f"model_type {model_type}")
            logger.info(f"model_info {model_info}")
            for profile_id, model_maps in model_info.items():
                for model_map in model_maps:
                    bus_id : str = model_map["bus"]
                    model_id : str = model_map["id"]
                    mult : float = model_map.get("multiplier")
                    norm: True = model_map.get("normalize")
                    dataset = self._store[f"{model_type}/{profile_id}"]
                    data = np.array(np.array(dataset).tolist())
              
                    if norm:
                        data_max = np.array(dataset.attrs["max"])
                        data = data / data_max
                    if mult:
                        data = data * mult

                    data = pd.DataFrame(data)                    
                    even_sum = data.iloc[:, ::2].sum(axis=1)
                    odd_sum = data.iloc[:, 1::2].sum(axis=1)
                    data = [even_sum, odd_sum]    
                    final_df = pd.concat(data, axis=1)
                    final_df.columns = [f"{model_type}_{model_id}_{bus_id}_P", f"{model_type}_{model_id}_{bus_id}_Q"]
                    
                    start_time = str(dataset.attrs["sTime"].decode('utf-8'))
                    end_time = str(dataset.attrs["eTime"].decode('utf-8'))
                    timestep = timedelta(seconds=dataset.attrs["resTime"])
                    date_range =pd.date_range(
                        start_time, 
                        end_time, 
                        freq=timestep)
                    final_df.index = date_range[:-1]
                    filtered_df = final_df.loc[(final_df.index >= self._start_time) & (final_df.index <= self._end_time)]
                    all_datasets.append(filtered_df)
                    
        final_df = pd.concat(all_datasets, axis=1)
        final_df.to_csv(self._export_file)
        logger.info(f"Profiles exported to {self._export_file}")