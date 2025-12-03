from datetime import timedelta
from pathlib import Path
import os

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
            for profile_id, model_maps in model_info.items():
                logger.info(f"model_type: {model_type}, model_info: {model_info}")
                for model_map in model_maps:
                    bus_id : str = model_map["bus"]
                    model_id : str = model_map["id"]
                    mult : float = model_map.get("multiplier")
                    norm: True = model_map.get("normalize")
                    dataset = self._store[f"{model_type}/{profile_id}"]
                    data = np.array(np.array(dataset).tolist())
                    # logger.info(f"data: {data[:10]}")
                    # os.system("PAUSE")
                    if model_type == "Load":
                        if norm:
                            data_max = np.array(dataset.attrs["max"])
                            data = data / data_max
                        if mult:
                            data = data * mult

                        data = pd.DataFrame(data)                    
                        P_even_sum = data.iloc[:, ::2].sum(axis=1)
                        Q_odd_sum = data.iloc[:, 1::2].sum(axis=1)
                        data = [P_even_sum, Q_odd_sum]
                    elif model_type == "Machine":
                        # Added by Fuhong, for a 'good' generation profile. Pgen and Qgen are strictly limited within the geneator maximum and minimum ranges.
                        data = pd.DataFrame(data)
                        P_gen = data.iloc[:, 0]
                        Q_gen = data.iloc[:, 1]
                        data = [P_gen, Q_gen]
                    
                    elif model_type == "Plant":
                        # Added by Fuhong
                        data = pd.DataFrame(data)
                        vs_plant = data.iloc[:, 0]
                        rmpct_plant = data.iloc[:, 1]
                        data = [vs_plant, rmpct_plant]

                    elif model_type == "Induction_machine":
                        # TODO
                        pass

                    elif model_type == "Load_status":
                        data = pd.DataFrame(data)
                        load_status = data.iloc[:, 0]
                        data = [load_status]

                    elif model_type == "Line_status":
                        data = pd.DataFrame(data)
                        line_status = data.iloc[:, 0]
                        data = [line_status]

                    elif model_type == "Machine_status":
                        data = pd.DataFrame(data)
                        machine_status = data.iloc[:, 0]
                        data = [machine_status]
                    else:
                        raise TypeError
                    
                    final_df = pd.concat(data, axis=1)
                    if model_type == "Load" or model_type == "Machine":
                        final_df.columns = [f"{model_type}_{model_id}_{bus_id}_P", f"{model_type}_{model_id}_{bus_id}_Q"]
                    elif model_type == "Plant":
                        final_df.columns = [f"{model_type}_{model_id}_{bus_id}_V", f"{model_type}_{model_id}_{bus_id}_RMPCT"]
                    elif model_type == "Load_status" or model_type == "Line_status" or model_type == "Machine_status":
                        final_df.columns = [f"{model_type}_{model_id}_{bus_id}_Status"]
                    else:
                        raise TypeError
                    
                    start_time = str(dataset.attrs["sTime"].decode('utf-8'))
                    end_time = str(dataset.attrs["eTime"].decode('utf-8'))
                    timestep = timedelta(seconds=int(dataset.attrs["resTime"]))
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