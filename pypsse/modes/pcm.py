# Standard imports
import datetime
import os

from loguru import logger

# Internal imports
from pypsse.modes.abstract_mode import AbstractMode
from pypsse.utils.dc2ac.dc_ac_algorithm import DC2ACconverter


class ProductionCostModel(AbstractMode):
    def __init__(self, psse, dyntools, der, settings, export_settings, subsystem_buses, raw_data):
        super().__init__(psse, dyntools, der, settings, export_settings, subsystem_buses, raw_data)
        self.time = datetime.datetime.strptime(settings["Simulation"]["Start time"], "%m/%d/%Y %H:%M:%S").astimezone(
            None
        )
        self._StartTime = datetime.datetime.strptime(
            settings["Simulation"]["Start time"], "%m/%d/%Y %H:%M:%S"
        ).astimezone(None)
        self.incTime = settings["Simulation"]["Step resolution (sec)"]
        self.convergence_helper = DC2ACconverter(psse, self, settings, raw_data)

    def init(self, bussubsystems):
        super().init(bussubsystems)
        self.initialization_complete = True

    def step(self):
        ierr = self.PSSE.fnsl()
        assert ierr == 0, f"Error code: {ierr}"
        solved_flag = self.PSSE.solved()
        # check if powerflow completed successfully
        if solved_flag == 0:
            self.time = self.time + datetime.timedelta(seconds=self.incTime)
            return 0
        else:
            self.convergence_helper.run()
            if not self.convergence_helper.has_converged:
                logger.error(
                    f"Error code {solved_flag} returned from PSSE while running powerflow, please follow \
                                PSSE doumentation to know more about error"
                )
                return 1
            else:
                self.time = self.time + datetime.timedelta(seconds=self.incTime)
                return 0

    def reload(self):
        pass

    def resolve_step(self):
        ierr = self.PSSE.fnsl()
        assert ierr == 0, f"Error code: {ierr}"
        solved_flag = self.PSSE.solved()
        if solved_flag > 0:
            self.convergence_helper.run()
            if not self.convergence_helper.has_converged:
                msg = f"Error code {solved_flag} returned from PSSE while running powerflow, please follow \
                                            PSSE doumentation to know more about error"
                raise Exception(msg)

    def get_time(self):
        return self.time

    def get_total_seconds(self):
        return (self.time - self._StartTime).total_seconds()

    def get_step_size_cec(self):
        return self.settings["Simulation"]["Step resolution (sec)"]

    def export(self):
        logger.debug("Starting export process. Can take a few minutes for large files")
        excelpath = os.path.join(self.export_path, self.settings["Excel file"])
        achnf = self.dyntools.CHNF(self.outx_path)
        achnf.xlsout(channels="", show=False, xlsfile=excelpath, outfile="", sheet="Sheet1", overwritesheet=True)
        logger.debug("{} export to {}".format(self.settings["Simulation"]["Excel file"], self.export_path))
