"""
 CLI to run a PyDSS project
 """

from pathlib import Path
 
from loguru import logger
import click
import toml
 
from pypsse.models import SimulationSettings
from pypsse.common import SIMULATION_SETTINGS_FILENAME
from pypsse.profile_manager_interface import ProfileManagerInterface
 
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
def get_profiles(project_path, simulations_file=None):
    """Runs a valid PyPSSE simulation."""
    file_path = Path(project_path) / simulations_file
    msg = "Simulation file not found. Use -s to choose a valid settings file"
    "if its name differs from the default file name."
    assert file_path.exists(), msg
    
    simulation_settings = toml.load(file_path)
    simulation_settings = SimulationSettings(**simulation_settings)
    
    logger.level(simulation_settings.log.logging_level.value)
    if simulation_settings.log.log_to_external_file:
        log_path = Path(project_path) / "Logs" / "pypsse.log"
        logger.add(log_path)
    
    profile_interface = ProfileManagerInterface.from_setting_files(file_path)
    profile_interface.get_profiles()
