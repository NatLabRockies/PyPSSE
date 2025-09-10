from loguru import logger

from pypsse.modes.constants import DYNAMIC_ONLY_PPTY, converter, dyn_only_options
from pypsse.models import ExportFileOptions, SimulationSettings
from pypsse.modes.abstract_mode import AbstractMode
from pypsse.utils.dynamic_utils import DynamicUtils


class Dynamic(AbstractMode, DynamicUtils):
    "Class defination for dynamic simulation mode (uses dyr and raw files)"

    def __init__(
        self,
        psse,
        dyntools,
        der,
        settings: SimulationSettings,
        export_settings: ExportFileOptions,
        subsystem_buses,
        raw_data,
    ):
        super().__init__(psse, dyntools, der, settings, export_settings, subsystem_buses, raw_data)
        self.time = settings.simulation.start_time
        self._StartTime = settings.simulation.start_time
        self.incTime = settings.simulation.simulation_step_resolution
        self.init(subsystem_buses)

    def init(self, bus_subsystems):
        "Initializes the simulation"
        super().init(bus_subsystems)
        self.iter_const = 100.0

        if self.settings.simulation.rwm_file:
            self.psse.mcre([1, 0], self.rwn_file)

        self.psse.fnsl([0, 0, 0, 1, 0, 0, 0, self._i])

        self.load_setup_files()
        self.convert_load()

        self.psse.gnet(1, 0)
        self.psse.fdns([1, 1, 0, 1, 1, 0, 0, 0])
        self.psse.fnsl([1, 1, 0, 1, 1, 0, 0, 0])
        self.psse.cong(0)
        # Solve for dynamics
        self.psse.ordr(0)
        self.psse.fact()
        self.psse.tysl(0)
        self.psse.tysl(0)
        # self.psse.save(self.study_case_path.split('.')[0] + ".sav")
        dyr_path = self.settings.simulation.dyr_file
        assert dyr_path and dyr_path.exists
        logger.debug(f"Loading dynamic model....{dyr_path}")
        self.psse.dynamicsmode(1)
        ierr = self.psse.dyre_new([1, 1, 1, 1], str(dyr_path), r"""conec""", r"""conet""", r"""compile""")

        if self.settings.helics and self.settings.helics.cosimulation_mode:
            if self.settings.helics.iterative_mode:
                sim_step = self.settings.simulation.psse_solver_timestep.total_seconds() / self.iter_const
            else:
                sim_step = self.settings.simulation.psse_solver_timestep.total_seconds()
        else:
            sim_step = self.settings.simulation.psse_solver_timestep.total_seconds()

        ierr = self.psse.dynamics_solution_param_2(
            [60, self._i, self._i, self._i, self._i, self._i, self._i, self._i],
            [0.4, self._f, sim_step, self._f, self._f, self._f, self._f, self._f],
        )

        if ierr:
            msg = f'Error loading dynamic model file "{dyr_path}". Error code - {ierr}'
            raise Exception(msg)
        else:
            logger.debug(f"Dynamic file {dyr_path} sucessfully loaded")

        self.disable_load_models_for_coupled_buses()

        if self.export_settings.export_results_using_channels:
            self.setup_channels()

        self.psse.delete_all_plot_channels()

        self.setup_all_channels()

        # Load user defined models
        self.load_user_defined_models()

        # Load flow settings
        self.psse.fdns([0, 0, 0, 1, 1, 0, 99, 0])
        # initialize
        outx_file  = str(self.settings.export.outx_file).split("\\")
        outx_file[-1] = self.export_settings.filename_prefix + "_" + outx_file[-1]
        outx_file = "\\".join(outx_file)
        ierr = self.psse.strt_2(
            [
                1,
                self.settings.generators.missing_machine_model,
            ],
            outx_file,
        )
        if ierr:
            self.initialization_complete = False
            msg = f"Dynamic simulation failed to successfully initialize. Error code - {ierr}"
            raise Exception(msg)
        else:
            self.initialization_complete = True
            logger.debug("Dynamic simulation initialization sucess!")
        # get load info for the sub system
        # self.load_info = self.get_load_indices(bus_subsystems)

        logger.debug("pyPSSE initialization complete!")

        self.xTime = 0

        return self.initialization_complete

    def step(self, t):
        "Increments the simulation"

        self.time = self.time + self.incTime
        self.xTime = 0
        # self.psse.run(0, t + self.incTime.total_seconds(), 1, 1, 1)
        # self.psse.run(0, t + self.incTime.total_seconds(), 1, 1, 1)
        return self.psse.run(0, t + self.incTime.total_seconds(), 1, 1, 1)

    # @kapil do you need this?
    
    # def get_load_indices(self, bus_subsystems):
    #     "Returns load indices"

    #     all_bus_ids = {}
    #     for bus_subsystem_id in bus_subsystems.keys():
    #         load_info = {}
    #         ierr, load_data = self.psse.aloadchar(bus_subsystem_id, 1, ["ID", "NAME", "EXNAME"])

    #         load_data = np.array(load_data)
    #         ierr, bus_data = self.psse.aloadint(bus_subsystem_id, 1, ["NUMBER"])

    #         bus_data = bus_data[0]
    #         for i, bus_id in enumerate(bus_data):
    #             load_info[bus_id] = {
    #                 "Load ID": load_data[0, i],
    #                 "Bus name": load_data[1, i],
    #                 "Bus name (ext)": load_data[2, i],
    #             }
    #         all_bus_ids[bus_subsystem_id] = load_info
    #     return all_bus_ids

    def resolve_step(self, t):
        "Resolves the current time step"

        err = self.psse.run(0, t + self.xTime * self.incTime / self.iter_const, 1, 1, 1)
        self.xTime += 1
        return err

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

        poll_results = self.poll_channels()
        results.update(poll_results)
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
                logger.warning("Extend function 'read_subsystems' in the Dynamic class (Dynamic.py)")
        return results
