"""
CLI to run a PyDSS project
"""


import datetime as dt
import os

import click
import toml
from loguru import logger

from pypsse.common import SIMULATION_SETTINGS_FILENAME
from pypsse.profile_manager.common import (
    DEFAULT_PROFILE_NAME,
    DEFAULT_PROFILE_RESOLUTION,
    DEFAULT_PROFILE_TYPE,
    DEFAULT_START_TIME,
    PROFILE_VALIDATION,
)
from pypsse.profile_manager.profile_store import ProfileManager
from pypsse.models import SimulationSettings

@click.argument(
    "project-path",
)
@click.option(
    "-f",
    "--csv-file-path",
    help="Path to a csv valid file",
    required=False,
    default="",
)
@click.option(
    "-p",
    "--profile-folder",
    help="""Path to folder containing csv profiles.
    CSV file names should follow the following format: {profile-type}_{profile-name}""",
    required=False,
    default="",
)
@click.option(
    "-n",
    "--profile-name",
    required=False,
    default=DEFAULT_PROFILE_NAME,
    show_default=True,
    help="Profile name",
)
@click.option(
    "-t",
    "--profile-type",
    required=False,
    default=DEFAULT_PROFILE_TYPE,
    show_default=True,
    help=f"Profile type; Possible values: {list(PROFILE_VALIDATION.keys())}",
)
@click.option(
    "-T",
    "--start-time",
    required=False,
    default=DEFAULT_START_TIME,
    show_default=True,
    help="Time index for the first time step, format = Y-m-d H:M:S.f",
)
@click.option(
    "-r",
    "--profile-res",
    required=False,
    default=DEFAULT_PROFILE_RESOLUTION,
    show_default=True,
    help="Profile time resolution in seconds",
)
@click.option(
    "-i",
    "--profile-info",
    required=False,
    default="",
    show_default=True,
    help="Profile time resolution in seconds",
)
@click.command()
def create_profiles(
    project_path, csv_file_path, profile_folder, profile_name, profile_type, start_time, profile_res, profile_info
):
    """Creates profiles for PyPSSE project."""
    settings_file = os.path.join(project_path, SIMULATION_SETTINGS_FILENAME)
    if os.path.exists(settings_file):
        if csv_file_path and os.path.exists(csv_file_path):
            settings = toml.load(settings_file)
            settings = SimulationSettings.model_validate(settings)
            
            a = ProfileManager(None, settings)
            a.add_profiles_from_csv(
                csv_file=csv_file_path,
                name=profile_name,
                p_type=profile_type,
                start_time=dt.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S.%f").astimezone(None),
                resolution_sec=profile_res,
                info=profile_info,
            )
            logger.info(f"Profile '{profile_name}' added to group '{profile_type}'")
        elif os.path.exists(profile_folder):
            settings = toml.load(settings_file)
            settings = SimulationSettings.model_validate(settings)
            a = ProfileManager(None, settings)
            for _, _, files in os.walk(profile_folder):
                for file in files:
                    if file.endswith(".csv"):
                        filename = file.replace(".csv", "")
                        if "__" in filename:
                            p_type, p_name = filename.split("__")
                            a.add_profiles_from_csv(
                                csv_file=os.path.join(profile_folder, file),
                                name=p_name,
                                p_type=p_type,
                                start_time=dt.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S.%f"),
                                resolution_sec=profile_res,
                                info=profile_info,
                            )
                            msg = f"Profile '{p_name}'' added to group '{p_type}'"
                            logger.info(msg)
        else:
            msg = "Value for either -f or -p flag has to be passed"
            raise Exception(msg)
    else:
        msg = f"{project_path} is not a valid pypsse project"
        raise Exception(msg)
