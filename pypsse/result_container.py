import datetime
from typing import Union

import pandas as pd
from loguru import logger

from pypsse.common import EXPORTS_FOLDER, MAPPED_CLASS_NAMES
from pypsse.data_writers.data_writer import DataWriter
from pypsse.enumerations import BulkWriteModes, StreamedWriteModes
from pypsse.models import ExportAssetTypes, ModelTypes, SimulationSettings, ExportFileOptions


class Container:
    "Class defination for the simulation result container"

    BULK_WRITE_MODES = [m.value for m in BulkWriteModes]
    STREAMED_WRITE_MODES = [m.value for m in StreamedWriteModes]

    def __init__(self, settings: SimulationSettings, export_settings: ExportFileOptions):
        """Sets up the result container object

        Args:
            settings (SimulationSettings): _description_
            export_settings (ExportAssetTypes): _description_
        """

        export__list = [m.value for m in ModelTypes]
        self.export_path = settings.simulation.project_path / EXPORTS_FOLDER
        self.export_settings = export_settings
        self.settings = settings
        self.results = {}
        self.export_vars = {}
        for class_name in export__list:
            mapped_name = MAPPED_CLASS_NAMES[class_name.lower()]
            variables = getattr(export_settings, class_name.lower())
            if variables:
                for variable in variables:
                    self.results[f"{mapped_name}_{variable.value}"] = None
                    if mapped_name not in self.export_vars:
                        self.export_vars[mapped_name] = []
                    self.export_vars[mapped_name].append(variable.value)

        time_steps = int(
            self.settings.simulation.simulation_time.total_seconds()
            / self.settings.simulation.simulation_step_resolution.total_seconds()
        )
        if self.export_settings.file_format not in self.BULK_WRITE_MODES:
            self.dataWriter = DataWriter(self.export_path, export_settings.file_format.value, time_steps,  self.export_settings.filename_prefix)

    def update_export_variables(self, params: Union[ExportAssetTypes, dict]) -> dict:
        """Updates the container with current system state.
        Method is called iteratively to store results as a simulation executes

        Args:
            params (Union[ExportAssetTypes, dict]): _description_

        Returns:
            dict: mapping of export variables to values
        """
        export__list = [m.value for m in ModelTypes]
        self.results = {}
        self.export_vars = {}
        if params:
            if isinstance(params, ExportAssetTypes):
                class_assets = params
            else:
                class_assets = ExportAssetTypes(**params)
        else:
            class_assets = self.export_settings

        inv_map = {v: k for k, v in MAPPED_CLASS_NAMES.items()}

        for class_name in export__list:
            if inv_map[class_name] in params:
                variables = getattr(class_assets, inv_map[class_name])
                if variables:
                    for variable in variables:
                        self.results[f"{class_name}_{variable.value}"] = None
                        if class_name not in self.export_vars:
                            self.export_vars[class_name] = []
                        self.export_vars[class_name].append(variable.value)

        return self.export_vars

    def get_export_variables(self) -> dict:
        """Queries and return results from the current timestep

        Returns:
            dict: mapping of export variables to values
        """

        return self.export_vars

    def update(self, bus_data: dict, _, time: datetime.datetime, has_converged: bool):
        """Updates the results cotainer

        Args:
            bus_data (dict): mapping of vairables to values
            _ (_type_): _description_
            time (datetime.datetime): simulation time
            has_converged (bool): flag showing if simulation converged
        """

        if self.export_settings.file_format not in self.BULK_WRITE_MODES:
            self.dataWriter.write(time, bus_data, has_converged)
        else:
            for variable_name, _ in bus_data.items():
                if not isinstance(self.results[f"{variable_name}"], pd.DataFrame):
                    self.results[f"{variable_name}"] = pd.DataFrame(bus_data[variable_name], index=[0])
                else:
                    df1 = self.results[f"{variable_name}"]
                    df2 = pd.DataFrame.from_dict([bus_data[variable_name]])
                    concatenated = pd.concat([df1, df2])
                    self.results[f"{variable_name}"] = concatenated
        logger.debug("result container updated")

    def export_results(self):
        """exports all results stored to an external file"""

        if self.export_settings.file_format in self.BULK_WRITE_MODES:
            for df_name, df in self.results.items():
                export_path = (
                    self.settings.simulation.project_path
                    / EXPORTS_FOLDER
                    / f'{df_name}.{self.export_settings.file_format}'
                )
                if self.export_settings.file_format == BulkWriteModes.CSV:
                    if isinstance(df, pd.DataFrame):
                        df.to_csv(export_path)
                elif self.export_settings.file_format == BulkWriteModes.PKL:
                    df.to_pickle(export_path)
                logger.info(f"results exported to {export_path}")
