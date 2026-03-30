from pathlib import Path
from uuid import uuid4
import tempfile
import shutil
import os

import toml

from pypsse.project import Project


TESTING_BASE_FOLDER = Path(__file__).parent

PROJECT_CREATION_SETTINGS = {
    "simulation_file": None,
    "export_settings_file": None,
    "psse_project_folder": "./tests/data/psse_project",
    "profile_store": "./tests/data/profiles/Profiles.hdf5",
    "profile_mapping": "./tests/data/profiles/Profile_mapping.toml",
    "overwrite": True,
    "autofill": True,
}


def build_temp_project():
    TEMPPATH = tempfile.gettempdir()
    TMP_FOLDER = os.path.join(TEMPPATH, str(uuid4()))
    PROJECT_NAME = str(uuid4())
    
    project_path = Path(TMP_FOLDER) / PROJECT_NAME
    
    if os.path.exists(TMP_FOLDER):
        shutil.rmtree(TMP_FOLDER)

    if not os.path.exists(TMP_FOLDER):
        os.mkdir(TMP_FOLDER)
    settings = PROJECT_CREATION_SETTINGS
    
    s_settings = toml.load(settings["simulation_file"]) if settings["simulation_file"] else {}
    e_settings = toml.load(settings["export_settings_file"]) if settings["export_settings_file"] else {}
    # TODO: Validate settings
    a = Project()
    a.create(
        TMP_FOLDER,
        PROJECT_NAME,
        settings["psse_project_folder"],
        s_settings,
        e_settings,
        settings["profile_store"],
        settings["profile_mapping"],
        settings["overwrite"],
        settings["autofill"],
    )
    return project_path
    
    
def remove_temp_project(project_path):
    if os.path.exists(project_path):
        print('----', project_path)
        shutil.rmtree(project_path)
