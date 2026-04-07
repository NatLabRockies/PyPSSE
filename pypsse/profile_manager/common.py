from enum import Enum

class ProfileTypes(str, Enum):
    LOAD = 'Load'
    INDUCTION_MACHINE = 'Induction_machine'
    MACHINE = 'Machine'
    PLANT = "Plant"
    LOAD_STATUS = "Load_status"
    LINE_STATUS = "Line_status"
    MACHINE_STATUS = "Machine_status"

PROFILE_VALIDATION = {
    ProfileTypes.LOAD: ["PL", "QL", "IP", "IQ", "YP", "YQ", "PG", "QG"],
    ProfileTypes.INDUCTION_MACHINE: [
        "MBASE",
        "RATEKVPSET",
        "H",
        "A",
        "B",
        "D",
        "E",
        "RA",
        "XA",
        "XM",
        "R1",
        "X1",
        "R2",
        "X2",
        "X3",
        "E1",
        "SE1",
        "E2",
        "SE2",
        "IA1",
        "IA2",
        "XAMULT",
    ],
    ProfileTypes.MACHINE: [
        "PG",
        "QG",
        "QT",
        "QB",
        "PT",
        "PB",
        "MBASE",
        "ZR",
        "ZX",
        "RT",
        "XT",
        "GTAP",
        "F1",
        "F2",
        "F3",
        "F4",
        "WPF",
    ],
    ProfileTypes.PLANT: ["VS", "RMPCT"],
    ProfileTypes.LOAD_STATUS: ["STATUS"],
    ProfileTypes.LINE_STATUS: ["STATUS"],
    ProfileTypes.MACHINE_STATUS: ["STATUS"],
}

DEFAULT_PROFILE_NAME = "Default"
DEFAULT_START_TIME = "2020-01-01 00:00:00.00"
DEFAULT_PROFILE_TYPE = "Load"
DEFAULT_PROFILE_RESOLUTION = 1.0