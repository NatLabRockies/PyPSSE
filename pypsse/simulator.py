"""
@author:Aadil Latif
@file:pyPSSE.py
@time:2/4/2020
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import toml
from loguru import logger
from networkx import Graph
import pssepath
import pypsse.contingencies as c
import pypsse.simulation_controller as sc
from pypsse.common import (
    EXPORTS_SETTINGS_FILENAME,
    LOGS_FOLDER,
    MAX_PSSE_BUSSYSTEMS,
    SIMULATION_SETTINGS_FILENAME,
)
from pypsse.contingencies import (
    BusFaultObject,
    BusTripObject,
    LineFaultObject,
    LineTripObject,
    MachineTripObject,
)
from pypsse.enumerations import SimulationStatus
from pypsse.helics_interface import HelicsInterface
from pypsse.models import (
    BusSubsystems,
    ExportAssetTypes,
    ExportFileOptions,
    SimulationModes,
    SimulationSettings,
)
from pypsse.parsers import gic_parser as gp
from pypsse.parsers import reader as rd
from pypsse.profile_manager.profile_store import ProfileManager
from pypsse.result_container import Container
from pypsse.models import Contingencies
from pypsse.enumerations import PSSE_VERSIONS
from pypsse.utils.dynamic_utils import DynamicUtils

USING_NAERM = 0

N_BUS = 200000


class Simulator(DynamicUtils):
    "Base class for the simulator"

    _status: SimulationStatus = SimulationStatus.NOT_INITIALIZED

    def __init__(
        self,
        settings: SimulationSettings,
        export_settings: Union[ExportFileOptions, None] = None,
        psse_version: str = PSSE_VERSIONS.PSSE35.value,
    ):
        """Load a valid PyPSSE project and sets up simulation

        Args:
            settings (SimulationSettings): simulation settings
            export_settings (Union[ExportFileOptions, None]): export settings
            psse_path (Union[str, Path], optional): Path to python environment within the PSS/e install directory
        """

        self._status = SimulationStatus.STARTING_INSTANCE
        self.settings = settings

        logger.debug(f"Instantiating psse version {psse_version}")
        pssepath.add_pssepath(35.4) 
        __import__(psse_version, fromlist=[""])  # noqa: F401

        import dyntools
        import psspy
        logger.debug(f"Imported psspy suscessfully")
        
        ierr = psspy.psseinit(N_BUS)
        assert ierr == 0, f"Error code: {ierr}"
        logger.debug(f"PSSE initialized sucessfully suscessfully")
        if export_settings is None:
            export_settings_path = (
                Path(self.settings.simulation.project_path)
                / EXPORTS_SETTINGS_FILENAME
            )
            assert (
                export_settings_path.exists()
            ), f"{export_settings_path} does nor exist"
            export_settings = toml.load(export_settings_path)
            export_settings = ExportFileOptions(**export_settings)

        self.export_settings = export_settings
        log_path = os.path.join(
            self.settings.simulation.project_path, LOGS_FOLDER
        )
        logger.debug("Starting PSSE instance")

        self.dyntools = dyntools
        self.psse = psspy
        # logger.debug('Initializing PSS/E. connecting to license server')

        self.start_simulation()
        self.init()
        self._status = SimulationStatus.INITIALIZATION_COMPLETE

    @classmethod
    def from_setting_files(
        cls, simulation_settiings_file: Path, export_Settings_file: Path = None
    ):
        """build 'Simulator' from toml settings files

        Args:
            simulation_settiings_file (Path): simulation settings
            export_Settings_file (Path): export settings
        """
        simulation_settiings = toml.load(simulation_settiings_file)
        if export_Settings_file:
            export_Settings = toml.load(export_Settings_file)
        else:
            export_Settings = toml.load(
                simulation_settiings_file.parent / EXPORTS_SETTINGS_FILENAME
            )

        simulation_settiings = SimulationSettings(**simulation_settiings)
        export_Settings = ExportFileOptions(**export_Settings)
        return cls(simulation_settiings, export_Settings)

    def dump_settings(
        self,
        dest_dir: Path,
        simulation_file: str = SIMULATION_SETTINGS_FILENAME,
        export_file: str = EXPORTS_SETTINGS_FILENAME,
    ):
        """Dumps simulation settings to a provided path

        Args:
            dest_dir (Path): Directory where settins are dumped
            simulation_file (str, optional): simulation filename. Defaults to SIMULATION_SETTINGS_FILENAME.
            export_file (str, optional): export setting filename. Defaults to EXPORTS_SETTINGS_FILENAME.
        """

        settings_json = self.settings.model_dump_json()
        json.dump(settings_json, open(dest_dir / simulation_file, "w"))

        export_settings_json = self.settings.model_dump_json()
        json.dump(export_settings_json, open(dest_dir / export_file, "w"))

    def start_simulation(self):
        """Starts a loaded simulation

        Raises:
            Exception: Please pass a RAW or SAV file in the settings dictionary
        """

        self.hi = None
        self.simStartTime = time.time()
        
        if self.settings.simulation.case_study is not None and self.settings.simulation.case_study.exists():
            self.psse.case(str(self.settings.simulation.case_study))
        elif self.settings.simulation.raw_file is not None and self.settings.simulation.raw_file.exists():
            self.psse.read(0, str(self.settings.simulation.raw_file))
        else:
            msg = "Please pass a RAW or SAV file in the settings dictionary"
            raise Exception(msg)

        logger.info(
            f"Trying to read a file >>{self.settings.simulation.case_study}"
        )
        self.raw_data = rd.Reader(self.psse)
        (
            self.bus_subsystems,
            self.all_subsysten_buses,
        ) = self.define_bus_subsystems()

        if self.export_settings.defined_subsystems_only:
            valid_buses = self.all_subsysten_buses
        else:
            valid_buses = self.raw_data.buses

        self.sim = sc.sim_controller(
            self.psse,
            self.dyntools,
            self.settings,
            self.export_settings,
            valid_buses,
            self.raw_data,
        )

        self.contingencies = self.build_contingencies()

        if self.settings.helics and self.settings.helics.cosimulation_mode:
            if self.settings.simulation.simulation_mode in [
                SimulationModes.DYNAMIC,
                SimulationModes.SNAP,
            ]:
                ...
            self.hi = HelicsInterface(
                self.psse,
                self.sim,
                self.settings,
                self.export_settings,
                self.bus_subsystems,
            )
            self.publications = self.hi.register_publications(
                self.bus_subsystems
            )
            if self.settings.helics.create_subscriptions:
                self.subscriptions = self.hi.register_subscriptions()

        if self.settings.simulation.gic_file:
            self.network_graph = self.parse_gic_file()
            self.bus_ids = self.network_graph.nodes.keys()
        else:
            self.network_graph = None

        self.results = Container(self.settings, self.export_settings)
        self.exp_vars = self.results.get_export_variables()
        self.inc_time = True

    def init(self):
        """Initializes the model"""

        self.sim.init(self.bus_subsystems)

        if self.settings.simulation.use_profile_manager:
            self.pm = ProfileManager(self.sim, self.settings)
            self.pm.setup_profiles()
        if self.settings.helics and self.settings.helics.cosimulation_mode:
            self.hi.enter_execution_mode()

    def parse_gic_file(self) -> Graph:
        """Parses the GIC file (if included in the project)

        Returns:
            Graph: Networkx graph representation for the model
        """

        gicdata = gp.GICParser(self.settings)
        return gicdata.psse_graph

    def define_bus_subsystems(self) -> (dict, list):
        """Defines a bussystem in the loaded PSSE model

        Raises:
            LookupError: Failed to create bus subsystem chosen buses.
            ValueError: Number of subsystems can not be more that 12. See PSSE documentation
            RuntimeError: Failed to add buses to bus subsystem

        Returns:
            dict: mapping of bus subsystems to buses
            list: List of bus subsystems
        """

        bus_subsystems_dict = {}
        bus_subsystems = self.get_bus_indices()
        # valid bus subsystem ID. Valid bus subsystem IDs range from 0 to 11 (PSSE documentation)
        if len(bus_subsystems) > MAX_PSSE_BUSSYSTEMS:
            msg = "Number of subsystems can not be more that 12. See PSSE documentation"
            raise ValueError(msg)

        all_subsysten_buses = []
        for i, buses in enumerate(bus_subsystems):
            if not buses:
                continue

            all_subsysten_buses.extend(buses)
            ierr = self.psse.bsysinit(i)
            if ierr:
                msg = "Failed to create bus subsystem chosen buses."
                raise LookupError(msg)
            else:
                logger.debug(f'Bus subsystem "{i}" created')

            ierr = self.psse.bsys(sid=i, numbus=len(buses), buses=buses)
            if ierr:
                msg = "Failed to add buses to bus subsystem."
                raise RuntimeError(msg)
            else:
                bus_subsystems_dict[i] = buses
                logger.debug(f'Buses {buses} added to subsystem "{i}"')
        all_subsysten_buses = [str(x) for x in all_subsysten_buses]
        return bus_subsystems_dict, all_subsysten_buses

    def get_bus_indices(self) -> BusSubsystems:
        """Returns bus indices for bus subsystems

        Returns:
            BusSubsystems: Bus subsystem model
        """

        if self.settings.bus_subsystems.from_file:
            bus_file = self.settings.bus_subsystems.bus_file
            bus_info = pd.read_csv(bus_file, index_col=None)
            bus_info = bus_info.values
            _, n_cols = bus_info.shape
            bus_data = []
            for col in range(n_cols):
                data = [int(x) for x in bus_info[:, col] if not np.isnan(x)]
                bus_data.append(data)
        else:
            bus_data = self.settings.bus_subsystems.bus_subsystem_list
        return bus_data

    def run(self):
        """Launches the simulation"""

        self._status = SimulationStatus.RUNNING_SIMULATION
        if self.sim.initialization_complete:
            if self.settings.plots and self.settings.plots.enable_dynamic_plots:
                bokeh_server_proc = subprocess.Popen(
                    ["bokeh", "serve"], stdout=subprocess.PIPE
                )  # noqa: S603,S607
            else:
                bokeh_server_proc = None

            logger.debug(
                 f"Running {self.settings.simulation.simulation_mode.value} simulation for time {self.settings.simulation.simulation_time.total_seconds()} sec"
            )
            total_simulation_time = (
                self.settings.simulation.simulation_time.total_seconds()
            )
            t = 0
            while True:
                self.step(t)
                if self.inc_time:
                    t += (
                        self.settings.simulation.simulation_step_resolution.total_seconds()
                    )
                if t >= total_simulation_time:
                    break

            self.psse.pssehalt_2()
            if not self.export_settings.export_results_using_channels:
                self.results.export_results()
            else:
                self.sim.export()

            if bokeh_server_proc is not None:
                bokeh_server_proc.terminate()
        else:
            logger.error(
                "Run init() command to initialize models before running the simulation"
            )
        self._status = "Simulation complete"

    def get_bus_ids(self) -> list:
        """Returns bus IDs

        Returns:
            list: Array of bus numbers
        """

        ierr, iarray = self.psse.abusint(-1, 1, "NUMBER")
        assert ierr == 0, f"Error code: {ierr}"
        return iarray

    def step(self, t: float) -> dict:
        """Steps through a single simulation time step. Is called iteratively to increment the simualtion

        Args:
            t (float): time step for the simulation

        Returns:
            dict: results from the current timestep
        """

        self.update_contingencies(t)
        if self.settings.simulation.use_profile_manager:
            self.pm.update()
        ctime = time.time() - self.simStartTime
        ierr, rval = self.psse.dsrval('TIME', 0)
        logger.debug(
            f"Simulation time: {t} seconds\nRun time: {ctime}\npsse time: {self.sim.get_time()} \nsolver time: {rval}"
        )
        if self.settings.helics and self.settings.helics.cosimulation_mode:
            if self.settings.helics.create_subscriptions:                
                # logger.debug(f"Time requested: {t}")
                # self.inc_time, helics_time = self.update_federate_time(t)
                # logger.debug(f"Time granted: {helics_time}")
                self.update_subscriptions()

        # pypsse_update_agian = "y"
        # while pypsse_update_agian != "no":
        #     pypsse_update_agian = input('enter your pypsse update command: ')
        #     if pypsse_update_agian == "y":
        #         if self.settings.helics and self.settings.helics.cosimulation_mode:
        #             if self.settings.helics.create_subscriptions:
        #                 self.update_subscriptions()
        #                 logger.debug(f"Time requested: {t}")
        #                 self.inc_time, helics_time = self.update_federate_time(t)
        #                 logger.debug(f"Time granted: {helics_time}")
        #     elif pypsse_update_agian == "u":
        #         if self.settings.helics and self.settings.helics.cosimulation_mode:
        #             self.publish_data()

        if self.inc_time:
            logger.debug(f"run PSSE simulation at time: {t}")
            self.sim.step(t)
            # os.system("PAUSE")
        else:
            self.sim.resolve_step()

        if self.settings.helics and self.settings.helics.cosimulation_mode:
            self.publish_data()

        if self.settings.helics and self.settings.helics.cosimulation_mode:
            if self.settings.helics.create_subscriptions:
                # self.update_subscriptions()
                logger.debug(f"Time requested: {t}")
                self.inc_time, helics_time = self.update_federate_time(t)
                logger.debug(f"Time granted: {helics_time}")

        curr_results = self.update_result_container(t)
        # if t >= 0.2:
        # The following logic only runs when the helics interface is enabled
            # self.disable_load_models_for_coupled_buses()
            # self.disable_generation_for_coupled_buses()
        # os.system("PAUSE")

        return curr_results

    def update_result_container(self, t: float) -> dict:
        """Updates the result container with results from the current time step

        Args:
            t (float): simulation time in seconds

        Returns:
            dict: simulation reults from the current time step
        """

        if self.export_settings.defined_subsystems_only:
            logger.debug(f"self.exp_vars : {self.exp_vars}")
            curr_results = self.sim.read_subsystems(
                self.exp_vars, self.all_subsysten_buses
            )
        else:
            curr_results = self.sim.read_subsystems(
                self.exp_vars, self.raw_data.buses
            )

        if not USING_NAERM:
            if not self.export_settings.export_results_using_channels:
                # logger.debug(f"curr_results : {curr_results}")
                self.results.update(
                    curr_results,
                    t,
                    self.sim.get_time(),
                    self.sim.has_converged(),
                )
        return curr_results

    def update_subscriptions(self):
        """Updates subscriptions (co-simulation mode only)"""

        self.hi.subscribe()

    def update_federate_time(self, t: float) -> (float, float):
        """Makes a time request to teh HELICS broker (co-simulation mode only)

        Args:
            t (float): simulation time in seconds

        Returns:
            float: requested time in seconds
            float: current simualtion time in seconds
        """

        inc_time, curr_time = self.hi.request_time(t)
        return inc_time, curr_time

    def publish_data(self):
        """Updates publications (co-simulation mode only)"""

        self.hi.publish()

    def get_results(self, params: Union[ExportAssetTypes, dict]) -> dict:
        """Returns queried simulation results

        Args:
            params (Union[ExportAssetTypes, dict]): _description_

        Returns:
            dict: simulation results
        """

        self._status = SimulationStatus.STARTING_RESULT_EXPORT
        self.exp_vars = self.results.update_export_variables(params)
        curr_results = (
            self.sim.read_subsystems(self.exp_vars, self.all_subsysten_buses)
            if self.export_settings.defined_subsystems_only
            else self.sim.read_subsystems(self.exp_vars, self.raw_data.buses)
        )
        self._status = SimulationStatus.RESULT_EXPORT_COMPLETE
        return curr_results

    def status(self) -> SimulationStatus:
        """returns current simulation status

        Returns:
            SimulationStatus: state of the simulator
        """
        return self._status.value

    def build_contingencies(
        self,
    ) -> List[
        Union[
            BusTripObject,
            BusFaultObject,
            LineTripObject,
            LineFaultObject,
            MachineTripObject,
        ]
    ]:
        """Builds user defined contengingies

        Returns:
            List[Union[BusFault, LineFault, LineTrip, BusTrip, MachineTrip]]: List of contingencies
        """

        contingencies = c.build_contingencies(self.psse, self.settings)
        return contingencies

    def inject_contingencies_external(self, contigencies: Contingencies):
        """Inject external contingencies.

        Args:
            contigencies (Contingencies): Contigencies Object
        """
        contingencies = c.build_contingencies(self.psse, contigencies)
        self.contingencies.extend(contingencies)

    def update_contingencies(self, t: float):
        """Updates contingencies during the simualtion run

        Args:
            t (float): simulation time in seconds
        """

        for contingency in self.contingencies:
            contingency.update(t)

    def force_psse_halt(self):
        """forces cleaup of pypss imort"""
        ierr = self.psse.pssehalt_2()
        assert ierr == 0, f"failed to halt PSSE. Error code - {ierr}"

    def __del__(self):
        try:
            self.force_psse_halt()
        except:
            pass