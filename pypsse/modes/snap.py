import numpy as np
from loguru import logger
import os

from pypsse.models import SimulationSettings, ExportFileOptions
from pypsse.modes.abstract_mode import AbstractMode
from pypsse.modes.constants import DYNAMIC_ONLY_PPTY, converter, dyn_only_options
from pypsse.utils.dynamic_utils import DynamicUtils


class Snap(AbstractMode, DynamicUtils):
    "Class defination for snapshat simulation mode (uses snp and sav files)"

    def __init__(
        self,
        psse,
        dyntools,
        settings: SimulationSettings,
        export_settings: ExportFileOptions,
        subsystem_buses,
        raw_data,
    ):
        super().__init__(psse, dyntools, settings, export_settings, subsystem_buses, raw_data)
        self.time = settings.simulation.start_time
        self._StartTime = settings.simulation.start_time
        self.incTime = settings.simulation.simulation_step_resolution
        self.init(subsystem_buses)

    def init(self, bus_subsystems):
        "Initializes the simulation"
        super().init(bus_subsystems)

        self.iter_const = 100.0
        self.xTime = 0

        ierr = self.psse.case(str(self.settings.simulation.case_study))

        self.load_setup_files()
        self.convert_load()

        logger.info(f"Load snap file: {self.settings.simulation.snp_file}")
        ierr = self.psse.rstr(str(self.settings.simulation.snp_file))
        #

        # # The following logic only runs when the helics interface is enabled
        self.disable_load_models_for_coupled_buses()
        self.disable_generation_for_coupled_buses()
        # self.save_model()
        ############# ------------------------------------- ###############
        outx_file  = str(self.settings.export.outx_file).split("\\")
        outx_file[-1] = self.export_settings.filename_prefix + "_" + outx_file[-1]
        outx_file = "\\".join(outx_file)
        
        ierr = self.psse.strt_2([0, 1],  outx_file)

        if ierr == 1:
            self.psse.cong(0)
            ierr = self.psse.strt_2([0, 1],  outx_file)

        elif ierr > 1:
            msg = "Error starting simulation"
            raise Exception(msg)

        self.load_user_defined_models()

        if self.settings.helics and self.settings.helics.cosimulation_mode:
            if self.settings.helics.iterative_mode:
                sim_step = self.settings.simulation.psse_solver_timestep.total_seconds() / self.iter_const
            else:
                sim_step = self.settings.simulation.psse_solver_timestep.total_seconds()
        else:
            sim_step = self.settings.simulation.psse_solver_timestep.total_seconds()

        self.psse.dynamics_solution_param_2(
            [60, self._i, self._i, self._i, self._i, self._i, self._i, self._i],
            [0.4, self._f, sim_step, self._f, self._f, self._f, self._f, self._f],
        )

        self.psse.delete_all_plot_channels()

        self.setup_all_channels()
        # print("111111111111111111111111111111111111111111111111111111111111111111")
        logger.debug("pyPSSE initialization complete!")
        self.initialization_complete = True
        # print("222222222222222222222222222222222222222222222222222222222222222222")
        return self.initialization_complete

    def step(self, t):
        "Increments the simulation"
        self.time = self.time + self.incTime
        self.xTime = 0
        return self.psse.run(0, t + self.incTime.total_seconds(), 1, 1, 1)

    def resolve_step(self, t):
        "Resolves the current time step"
        self.xTime += 1
        return self.psse.run(0, t + self.xTime * self.incTime / self.iter_const, 1, 1, 1)

    def get_load_indices(self, bus_subsystems):
        "Returns load indices"
        all_bus_ids = {}
        for bus_subsystem_id in bus_subsystems.keys():
            load_info = {}
            ierr, load_data = self.psse.aloadchar(bus_subsystem_id, 1, ["ID", "NAME", "EXNAME"])

            load_data = np.array(load_data)
            ierr, bus_data = self.psse.aloadint(bus_subsystem_id, 1, ["NUMBER"])

            bus_data = bus_data[0]
            for i, bus_id in enumerate(bus_data):
                load_info[bus_id] = {
                    "Load ID": load_data[0, i],
                    "Bus name": load_data[1, i],
                    "Bus name (ext)": load_data[2, i],
                }
            all_bus_ids[bus_subsystem_id] = load_info
        return all_bus_ids

    def get_time(self):
        "Returns current simulator time"
        return self.time

    def get_total_seconds(self):
        "Returns total simulation time"
        return (self.time - self._StartTime).total_seconds()

    def get_step_size_cec(self):
        "Returns simulation timestep resolution"
        return self.settings.simulation.simulation_step_resolution.total_seconds()

    @converter
    def read_subsystems(self, quantities, subsystem_buses, ext_string2_info=None, mapping_dict=None):
        "Queries the result container for current results"
        if ext_string2_info is None:
            ext_string2_info = {}
        if mapping_dict is None:
            mapping_dict = {}
        
        results = super().read_subsystems(
            quantities, subsystem_buses, mapping_dict=mapping_dict, ext_string2_info=ext_string2_info
        )
        # logger.debug(f"snap.py results : {results}")
        # logger.debug(f"quantities : {quantities}")
        # logger.debug(f"subsystem_buses : {subsystem_buses}")
        # logger.debug(f"mapping_dict : {mapping_dict}")
        # logger.debug(f"ext_string2_info : {ext_string2_info}")
        # raise Exception(1)
        poll_results = self.poll_channels()
        results.update(poll_results)
        # logger.debug(f"snap.py updated results : {results}")
        # os.system("PAUSE")
        """ Add """
        for class_name, var_list in quantities.items():
            if class_name in dyn_only_options:
                for v in var_list:
                    if v in DYNAMIC_ONLY_PPTY[class_name]:
                        for func_name in dyn_only_options[class_name]:
                            if v in dyn_only_options[class_name][func_name]:
                                con_ind = dyn_only_options[class_name][func_name][v]
                                for bus in subsystem_buses:
                                    if class_name == "Loads":
                                        ierr = self.psse.inilod(int(bus))

                                        ierr, ld_id = self.psse.nxtlod(int(bus))

                                        if ld_id is not None:
                                            ierr, con_index = getattr(self.psse, func_name)(
                                                int(bus), ld_id, "CHARAC", "CON"
                                            )

                                            if con_index is not None:
                                                act_con_index = con_index + con_ind
                                                ierr, value = self.psse.dsrval("CON", act_con_index)

                                                res_base = f"{class_name}_{v}"
                                                if res_base not in results:
                                                    results[res_base] = {}
                                                obj_name = f"{bus}_{ld_id}"
                                                results[res_base][obj_name] = value
            else:
                logger.warning(f"{class_name} not in the option")

        return results
