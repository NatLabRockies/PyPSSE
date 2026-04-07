# Standard imports
import os

from loguru import logger

# Internal imports
from pypsse.modes.abstract_mode import AbstractMode


class Static(AbstractMode):
    def __init__(self, psse, dyntools, der, settings, export_settings, subsystem_buses, raw_data):
        "Class defination for steady-state simulation mode"

        super().__init__(psse, dyntools, der, settings, export_settings, subsystem_buses, raw_data)
        self.time = settings.simulation.start_time
        self._StartTime = settings.simulation.start_time
        self.incTime = settings.simulation.simulation_step_resolution

    def init(self, bussubsystems):
        super().init(bussubsystems)
        self.initialization_complete = True

    def step(self, _):
        "Increments the simulation"
        ierr = self.psse.fnsl()
        # check if powerflow completed successfully
        if ierr == 0:
            self.time = self.time + self.incTime
        else:
            msg = f"Error code {ierr} returned from PSSE while running powerflow, please follow \
                            PSSE doumentation to know more about error"
            raise Exception(msg)

    def resolve_step(self):
        "Resolves the current time step"
        ierr = self.psse.fnsl()
        if ierr > 0:
            msg = f"Error code {ierr} returned from PSSE while running powerflow, please follow \
                                        PSSE doumentation to know more about error"
            raise Exception(msg)

    def get_time(self):
        "Returns current simulator time"
        return self.time

    def get_total_seconds(self):
        "Returns total simulation time"
        return (self.time - self._StartTime).total_seconds()

    def get_step_size_cec(self):
        "Returns simulation timestep resolution"
        return self.settings.simulation.simulation_step_resolution.total_seconds()

    def export(self):
        "Exports simulation results"
        logger.debug("Starting export process. Can take a few minutes for large files")
        excelpath = os.path.join(self.export_path, self.settings["Excel file"])
        achnf = self.dyntools.CHNF(self.outx_path)
        achnf.xlsout(channels="", show=False, xlsfile=excelpath, outfile="", sheet="Sheet1", overwritesheet=True)
        logger.debug(f"{self.settings.export.excel_file} exported")
