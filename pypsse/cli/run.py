"""
CLI to run a PyDSS project
"""

from pathlib import Path

from loguru import logger
import click
import toml
import os

from pypsse.models import SimulationSettings
from pypsse.common import SIMULATION_SETTINGS_FILENAME
from pypsse.simulator import Simulator


@click.argument(
    "project-path",
)
@click.option(
    "-s",
    "--simulations-file",
    required=False,
    default=SIMULATION_SETTINGS_FILENAME,
    show_default=True,
    help="scenario toml file to run (over rides default)",
)
@click.command()
def run(project_path, simulations_file=None):
    """Runs a valid PyPSSE simulation."""
    file_path = Path(project_path) / simulations_file
    msg = "Simulation file not found. Use -s to choose a valid settings file"
    "if its name differs from the default file name."
    assert file_path.exists(), msg
    
    simulation_settiings = toml.load(file_path)
    simulation_settiings = SimulationSettings(**simulation_settiings)
    logger.level(simulation_settiings.log.logging_level.value)
    if simulation_settiings.log.log_to_external_file:
        log_path = Path(project_path) / "Logs" / "pypsse.log"
        if simulation_settiings.log.clear_old_log_file:
            logger.add(log_path, mode="w")
        else:
            logger.add(log_path)
    
    x = Simulator.from_setting_files(file_path)

    x.run()
