from typing import Union

from loguru import logger

from pypsse.models import ExportFileOptions, SimulationSettings
from pypsse.modes.dynamic import Dynamic
from pypsse.modes.pcm import ProductionCostModel
from pypsse.modes.snap import Snap
from pypsse.modes.static import Static
from pypsse.parsers.reader import Reader


def sim_controller(
    psse: object,
    dyntools: object,
    der: object,
    settings: SimulationSettings,
    export_settings: ExportFileOptions,
    subsystem_buses: dict,
    raw_data: Reader,
) -> Union[Dynamic, ProductionCostModel, Snap, Static]:
    """sets up an appropriate simualtion controller based on user input

    Args:
        psse (object): simulator instance
        dyntools (object): psse dyntools instance
        settings (SimulationSettings): simulation settings
        export_settings (ExportFileOptions): export settings
        subsystem_buses (dict): mapping of bus subsystems to buses
        raw_data (Reader): instance of model reader

    Returns:
        Union[Dynamic, ProductionCostModel, Snap, Static]: simulator controller instance
    """

    sim_modes = {"Dynamic": Dynamic, "Steady-state": Static, "Snap": Snap, "ProductionCostModel": ProductionCostModel}
    # print("0000000000000000000000000000000000000000000000000000")
    sim = sim_modes[settings.simulation.simulation_mode.value](
        psse, dyntools, der, settings, export_settings, subsystem_buses, raw_data
    )
    # print("33333333333333333333333333333333333333333333333333333")
    logger.debug(f"Simulator contoller of type {settings.simulation.simulation_mode.value} created")
    # print("4444444444444444444444444444444444444444444444444444444")
    return sim
