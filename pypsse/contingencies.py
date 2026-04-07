"This module manages contingency modeling in PyPSSE"


from abc import ABCMeta
from typing import List, Union

from loguru import logger

# from pyPSSE import
from pypsse.models import (
    BusFault,
    Contingencies,
    BusTrip,
    LineFault,
    LineTrip,
    MachineTrip,
    SimulationSettings,
)


def add_contingency(contingency, cont_dict, dt, system_contingencies):
    "Adds a new contingency"
    ontingency_type = contingency.__class__.__name__
    if ontingency_type in contingencies:
        system_contingencies.append(contingencies[ontingency_type](**cont_dict))
    else:
        msg = "Invalid contingency type. Valid values are: {}".format(
            ",".join(contingencies.keys())
        )
        raise Exception(msg)
    return system_contingencies, dt


class BaseFault:
    "Base class defination for all fault types"
    __metaclass__ = ABCMeta

    requirements = []
    fault_settings = {}
    fault_method = ""
    element = None

    def __init__(self, psse, settings, contingency_type):
        self.contingency_type = contingency_type
        self.settings = settings
        self.psse = psse
        self.enabled = False
        self.tripped = False
        logger.debug(
                f"contingency_type : {contingency_type}"
            )
        logger.debug(
                f"settings : {settings}"
            )
        # os.system("PAUSE")
        
    def update(self, t: float):
        """updates a fault event

        Args:
            t (float): simuation time in seconds
        """
        self.t = t
        if hasattr(self.settings, "duration"):
            if (self.settings.time + self.settings.duration > t >= self.settings.time and not self.enabled):
                self.enabled = True
                self.enable_fault()
            if (t >= self.settings.time + self.settings.duration and self.enabled):
                self.enabled = False
                self.disable_fault()
        elif (not hasattr(self.settings, "duration") and t >= self.settings.time and not self.tripped):
            self.enable_fault()
            self.tripped = True

    def enable_fault(self):
        """enables a fault event"""
        data_check = getattr(self.psse, self.fault_method)
        logger.debug(
                f"data_check : {data_check}"
            )
        logger.debug(
                f"data_check : {self.fault_method}"
            )
        # os.system("PAUSE")

        err = getattr(self.psse, self.fault_method)(**self.fault_settings)
        if err:
            logger.warning(
                f"Unable to enable {self.fault_method} at element {self.element}"
            )
        else:
            logger.debug(
                f"{self.fault_method} applied to {self.element} at time {self.t} seconds"
            )

    def disable_fault(self):
        """disables a fault event"""
        err = self.psse.dist_clear_fault()
        if err:
            logger.warning(
                f"Unable to clear {self.fault_method} at element {self.element}"
            )
        else:
            logger.debug(
                f"{self.fault_method} cleared at element {self.element} at time {self.t} seconds"
            )

    def is_enabled(self):
        """Returns enabled status

        Returns:
            _type_: rue if the fault object is enabled else false
        """
        return self.enabled

    def is_tripped(self) -> bool:
        """Returns trip status

        Returns:
            bool: true if the fault object is tripped else false
        """
        return self.tripped


class BusFaultObject(BaseFault):
    "Class defination for a bus fault"
    fault_method = "dist_bus_fault"
    fault_settings = {}

    def __init__(self, psse: object, settings: BusFault, contingency_type: str):
        """bus fault object

        Args:
            psse (object): simulator type
            settings (BusFault): bus fault object
            contingency_type (str): contingency type
        """
        super().__init__(psse, settings, contingency_type)
        self.fault_settings["ibus"] = settings.bus_id
        self.fault_settings["units"] = 3
        self.fault_settings["values"] = settings.fault_impedance
        self.fault_settings["basekv"] = 0.0
        self.element = settings.bus_id


class LineFaultObject(BaseFault):
    "Class defination for a line fault"
    fault_method = "dist_branch_fault"
    fault_settings = {}

    def __init__(
        self, psse: object, settings: LineFault, contingency_type: str
    ):
        """line fault model

        Args:
            psse (object): simulator instance
            settings (LineFault): line fault model
            contingency_type (str): contingecy type
        """
        super().__init__(psse, settings, contingency_type)
        self.fault_settings["ibus"] = settings.bus_ids[0]
        self.fault_settings["jbus"] = settings.bus_ids[1]
        self.fault_settings["id"] = settings.bus_ids[2]
        self.fault_settings["units"] = 3
        self.fault_settings["values"] = settings.fault_impedance
        self.fault_settings["basekv"] = 0.0
        self.element = settings.bus_ids


class LineTripObject(BaseFault):
    "Class defination for a line trip"
    fault_method = "dist_branch_trip"
    fault_settings = {}

    def __init__(self, psse: object, settings: LineTrip, contingency_type: str):
        """line trip model

        Args:
            psse (object): simulator instance
            settings (LineTrip): line trip model
            contingency_type (str): contingency type
        """
        super().__init__(psse, settings, contingency_type)
        self.fault_settings["ibus"] = settings.bus_ids[0]
        self.fault_settings["jbus"] = settings.bus_ids[1]
        self.element = settings.bus_ids


class BusTripObject(BaseFault):
    "Class defination for a bus trip"
    fault_method = "dist_bus_trip"
    fault_settings = {}

    def __init__(self, psse: object, settings: BusTrip, contingency_type: str):
        """Bus trip contingency

        Args:
            psse (object): simulator instance
            settings (BusTrip): bus trip model
            contingency_type (str): type of contingency
        """
        super().__init__(psse, settings, contingency_type)
        self.fault_settings["ibus"] = settings.bus_id
        self.element = settings.bus_id


class MachineTripObject(BaseFault):
    "Class defination for a machine fault"
    fault_method = "dist_machine_trip"
    fault_settings = {}

    def __init__(
        self, psse: object, settings: MachineTrip, contingency_type: str
    ):
        """Machine trip contingency

        Args:
            psse (object): simulator instance
            settings (MachineTrip): machine trip model
            contingency_type (str): type of contingency
        """
        super().__init__(psse, settings, contingency_type)
        self.fault_settings["ibus"] = settings.bus_id
        self.fault_settings["id"] = settings.machine_id
        self.element = settings.bus_id


contingencies = {
    "BusFault": BusFaultObject,
    "LineFault": LineFaultObject,
    "LineTrip": LineTripObject,
    "BusTrip": BusTripObject,
    "MachineTrip": MachineTripObject,
}


def build_contingencies(
    psse: object, contingencies_: Union[Contingencies, SimulationSettings]
) -> List[
    Union[
        BusFaultObject,
        BusTripObject,
        LineFaultObject,
        LineTripObject,
        MachineTripObject,
    ]
]:
    """Builds all contingencies defined in the settings file

    Args:
        psse (object): simulator instance
        settings (SimulationSettings): simulation settings

    Returns:
        List[Union[BusFaultObject, BusTripObject, LineFaultObject, LineTripObject, MachineTripObject]]: list of contingencies
    """

    system_contingencies = []
    if contingencies_.contingencies:
        for contingency in contingencies_.contingencies:
            contingency_type = contingency.__class__.__name__
            if contingency_type in contingencies:
                system_contingencies.append(
                    contingencies[contingency_type](
                        psse, contingency, contingency_type
                    )
                )
                logger.debug(f'Contingency of type "{contingency_type}" added')
            else:
                logger.warning(
                    "Invalid contingency type. Valid values are: {}".format(
                        ",".join(contingencies)
                    )
                )
    else:
        logger.debug("No contingencies to build")
    return system_contingencies
