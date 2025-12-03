"""Defines all standard shared varibales in PyPSSE"""

VALUE_UPDATE_BOUND = 1e6
MAX_PSSE_BUSSYSTEMS = 12

CASESTUDY_FOLDER = "case_study"
SETTINGS_FOLDER = "settings"
EXPORTS_FOLDER = "exports"
LOGS_FOLDER = "logs"
PROFILES_FOLDER = "profiles"
DEFAULTS_FOLDER = "defaults"
EXAMPLES_FOLDER = "examples"

SIMULATION_SETTINGS_FILENAME = "simulation_settings.toml"
EXPORTS_SETTINGS_FILENAME = "export_settings.toml"
DEFAULT_SUBSCRIPTION_FILENAME = "subscriptions.csv"
DEFAULT_PROFILE_MAPPING_FILENAME = "profile_mapping.toml"
DEFAULT_PROFILE_STORE_FILENAME = "profiles.hdf5"
DEFAULT_RESULTS_FILENAME = "simulation_results.hdf5"

DEFAULT_OUT_FILE = "results.out"
DEFAULT_OUTX_FILE = "results.outx"
DEFAULT_EXCEL_FILE = "results.xls"
DEFAULT_LOG_FILE = "psse.log"
DEFAULT_GRAPH_FILE = "network.gpickle"
DEFAULT_COORDINATES_FILE = "coordinates.csv"
DEFAULT_PROFILE_EXPORT_FILE = "profiles.csv"

MAPPED_CLASS_NAMES = {
    "buses": "Buses",
    "areas": "Areas",
    "zones": "Zones",
    "loads": "Loads",
    "branches": "Branches",
    "machines": "Machines",
    "stations": "Stations",
    "transformers": "Transformers",
    "fixed_shunts": "Fixed_shunts",
    "switched_shunts": "Switched_shunts",
    "induction_generators": "Induction_generators",
    "dctransmissionlines": "DCtransmissionlines",
}

MACHINE_CHANNELS = {
    "ANGLE": 1,
    "PELEC": 2,
    "QELEC": 3,
    "ETERM": 4,
    "EFD": 5,
    "PMECH": 6,
    "SPEED": 7,
    "XADIFD": 8,
    "ECOMP": 9,
    "VOTHSG": 10,
    "VREF": 11,
    "VUEL": 12,
    "VOEL": 13,
    "GREF": 14,
    "LCREF": 15,
    "WVLCTY": 16,
    "WTRBSP": 17,
    "WPITCH": 18,
    "WAEROT": 19,
    "WROTRV": 20,
    "WROTRI": 21,
    "WPCMND": 22,
    "WQCMND": 23,
    "WAUXSG": 24,
}
