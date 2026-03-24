from enum import Enum, IntEnum


class ProjectFolders(str, Enum):
    "Defines valid the project folder"
    CASESTUDY = "case_study"
    EXPORTS = "exports"
    GISDATA = "gis_data"
    LOGS = "logs"
    PROFILES = "profiles"


class BulkWriteModes(str, Enum):
    "Supported bulk writers"
    CSV = "csv"
    PKL = "pkl"


class StreamedWriteModes(str, Enum):
    "Supported stream writers"
    H5 = "h5"


class SimulationModes(str, Enum):
    "Valid PyPSSE simulation modes"
    PCM = "PCM"
    SNAP = "Snap"
    STATIC = "Steady-state"
    DYNAMIC = "Dynamic"

class GenerationLevel(str, Enum):
    "Valid generation level setting modes"
    TRANSMISSION = "transmission"
    DISTRIBUTION = "distribution"

class HelicsCoreTypes(str, Enum):
    "HELICS core types"
    ZMQ = "zmq"
    TCP_SS="tcp_ss"
    TCP="tcp"


class WritableModelTypes(str, Enum):
    "Writable data ty[es]"
    LOAD = "Load"
    PLANT = "Plant"
    MACHINE = "Machine"
    GENERATOR = "Induction_machine"
    LOAD_STATUS = "Load_status"
    LINE_STATUS = "Line_status"
    MACHINE_STATUS = "Machine_status"

class ModelTypes(str, Enum):
    "Supported asset tpyes in PyPSSE"
    BUSES = "Buses"
    BRANCHES = "Branches"
    LOADS = "Loads"
    GENERATORS = "Induction_generators"
    MACHINES = "Machines"
    FIXED_SHUNTS = "Fixed_shunts"
    SWITCHED_SHUNTS = "Switched_shunts"
    TRANSFORMERS = "Transformers"
    AREAS = "Areas"
    ZONES = "Zones"
    DC_LINES = "DCtransmissionlines"
    STATIONS = "Stations"


class ModelProperties(str, Enum):
    "Support model properties"
    PU = "PU"
    FREQ = "FREQ"
    ANGLE = "ANGLE"
    ANGLED = "ANGLED"


class LoggingLevels(str, Enum):
    "logging level setting options"
    NOTSET = "INFO"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SubscriptionFileRequiredColumns(str, Enum):
    "Subscription file requirements"
    bus_subsystem_id = "bus_subsystem_id"
    element_type = "element_type"
    element_id = "element_id"
    element_property = "element_property"
    sub_tag = "sub_tag"
    scaler = "scaler"
    bus = "bus"


class BusProperties(str, Enum):
    "Valid bus properties"

    BASE = "BASE"
    FREQ = "FREQ"
    PU = "PU"
    KV = "KV"
    ANGLE = "ANGLE"
    ANGLED = "ANGLED"
    NVLMHI = "NVLMHI"
    NVLMLO = "NVLMLO"
    EVLMHI = "EVLMHI"
    EVLMLO = "EVLMLO"


class AreaProperties(str, Enum):
    "Valid area properties"

    LOAD = "LOAD"
    LOADID = "LOADID"
    LGDN = "LGDN"
    LDGNLD = "LDGNLD"
    GEN = "GEN"
    AREANAME = "AREANAME"
    AREANUMBER = "AREANUMBER"


class ZoneProperties(str, Enum):
    "Valid zone properties"

    LOAD = "LOAD"
    LOADID = "LOADID"
    LGDN = "LGDN"
    LDGNLD = "LDGNLD"
    GEN = "GEN"
    ZONENAME = "ZONENAME"
    ZONENUMBER = "ZONENUMBER"


class StationProperties(str, Enum):
    "Valid station properties"

    SUBNAME = "SUBNAME"
    SUBNUMBER = "SUBNUMBER"
    BUSES = "BUSES"
    GENERATORS = "GENERATORS"
    TRANSFORMERS = "TRANSFORMERS"
    NOMKV = "NOMKV"
    LOADMW = "LOADMW"
    GENMW = "GENMW"


class DCLineProperties(str, Enum):
    "Valid DC line properties"

    DCLINENAME = "DCLINENAME"
    MDC = "MDC"
    RECT = "RECT"
    INV = "INV"
    METER = "METER"
    NBR = "NBR"
    NBI = "NBI"
    ICR = "ICR"
    ICI = "ICI"
    NDR = "NDR"
    NDI = "NDI"


class LoadProperties(str, Enum):
    "Valid load properties"

    MVA = "MVA"
    IL = "IL"
    YL = "YL"
    TOTAL = "TOTAL"
    YNEG = "YNEG"
    YZERO = "YZERO"
    FmA = "FmA"
    FmB = "FmB"
    FmC = "FmC"
    FmD = "FmD"
    Fel = "Fel"
    PFel = "PFel"
    TD = "TD"
    TC = "TC"


class FixedShuntProperties(str, Enum):
    "Valid fixed shunt properties"

    ACT = "ACT"
    O_ACT = "O_ACT"
    NOM = "NOM"
    O_NOM = "O_NOM"
    PQZERO = "PQZERO"
    PQZ = "PQZ"
    O_PQZ = "O_PQZ"


class SwitchedShuntProperties(str, Enum):
    "Valid switched shunt properties"

    VSWHI = "VSWHI"
    VSWLO = "VSWLO"
    RMPCT = "RMPCT"
    BINIT = "BINIT"
    O_BINIT = "O_BINIT"


class TransformerProperties(str, Enum):
    "Valid transformer properties"

    RATIO = "RATIO"
    RATIO2 = "RATIO2"
    ANGLE = "ANGLE"
    RMAX = "RMAX"
    RMIN = "RMIN"
    VMAX = "VMAX"
    VMIN = "VMIN"
    STEP = "STEP"
    CR = "CR"
    CX = "CX"
    CNXANG = "CNXANG"
    SBASE1 = "SBASE1"
    NOMV1 = "NOMV1"
    NOMV2 = "NOMV2"
    GMAGNT = "GMAGNT"
    BMAGNT = "BMAGNT"
    RG1 = "RG1"
    XG1 = "XG1"
    R01 = "R01"
    X01 = "X01"
    RG2 = "RG2"
    XG2 = "XG2"
    R02 = "R02"
    X02 = "X02"
    RNUTRL = "RNUTRL"
    XNUTRL = "XNUTRL"
    RX1_2 = "RX1-2"
    RX2_3 = "RX2-3"
    RX3_1 = "RX3-1"
    YMAGNT = "YMAGNT"
    ZG1 = "ZG1"
    Z01 = "Z01"
    ZG2 = "ZG2"
    Z02 = "Z02"
    ZG3 = "ZG3"
    Z03 = "Z03"
    ZNUTRL = "ZNUTRL"


class BranchProperties(str, Enum):
    "Valid branch properties"

    RATEn = "RATEn"
    RATEA = "RATEA"
    RATEB = "RATEB"
    RATEC = "RATEC"
    RATE = "RATE"
    LENGTH = "LENGTH"
    CHARG = "CHARG"
    CHARGZ = "CHARGZ"
    FRACT1 = "FRACT1"
    FRACT2 = "FRACT2"
    FRACT3 = "FRACT3"
    FRACT4 = "FRACT4"
    RX = "RX"
    ISHNT = "ISHNT"
    JSHNT = "JSHNT"
    RXZ = "RXZ"
    ISHNTZ = "ISHNTZ"
    JSHNTZ = "JSHNTZ"
    LOSSES = "LOSSES"
    O_LOSSES = "O_LOSSES"
    MVA = "MVA"
    AMPS = "AMPS"
    PUCUR = "PUCUR"
    CURANG = "CURANG"
    P = "P"
    O_P = "O_P"
    Q = "Q"
    O_Q = "O_Q"
    PLOS = "PLOS"
    O_PLOS = "O_PLOS"
    QLOS = "QLOS"
    O_QLOS = "O_QLOS"


class InductionGeneratorProperties(str, Enum):
    "Valid induction generator properties"

    MBASE = "MBASE"
    RATEKV = "RATEKV"
    PSET = "PSET"
    RA = "RA"
    XA = "XA"
    R1 = "R1"
    X1 = "X1"
    R2 = "R2"
    X2 = "X2"
    X3 = "X3"
    E1 = "E1"
    SE1 = "SE1"
    E2 = "E2"
    SE2 = "SE2"
    IA1 = "IA1"
    IA2 = "IA2"
    XAMULT = "XAMULT"
    TRQA = "TRQA"
    TRQB = "TRQB"
    TRQD = "TRQD"
    TRQE = "TRQE"
    H = "H"
    IRATIO = "IRATIO"
    ROVERX = "ROVERX"
    RZERO = "RZERO"
    XZERO = "XZERO"
    RGRND = "RGRND"
    XGRND = "XGRND"
    P = "P"
    O_P = "O_P"
    Q = "Q"
    O_Q = "O_Q"
    MVA = "MVA"
    O_MVA = "O_MVA"
    SLIP = "SLIP"
    ZA = "ZA"
    Z1 = "Z1"
    Z2 = "Z2"
    ZZERO = "ZZERO"
    ZGRND = "ZGRND"
    PQ = "PQ"
    O_PQ = "O_PQ"


class MachinesProperties(str, Enum):
    "Valid machine properties"

    QMAX = "QMAX"
    O_QMAX = "O_QMAX"
    QMIN = "QMIN"
    O_QMIN = "O_QMIN"
    PMAX = "PMAX"
    O_PMAX = "O_PMAX"
    PMIN = "PMIN"
    O_PMIN = "O_PMIN"
    MBASE = "MBASE"
    MVA = "MVA"
    O_MVA = "O_MVA"
    P = "P"
    O_P = "O_P"
    Q = "Q"
    O_Q = "O_Q"
    PERCENT = "PERCENT"
    GENTAP = "GENTAP"
    VSCHED = "VSCHED"
    WPF = "WPF"
    RMPCT = "RMPCT"
    RPOS = "RPOS"
    XSUBTR = "XSUBTR"
    XTRANS = "XTRANS"
    XSYNCH = "XSYNCH"
    PQ = "PQ"
    O_PQ = "O_PQ"
    ZSORCE = "ZSORCE"
    XTRAN = "XTRAN"
    ZPOS = "ZPOS"
    ZNEG = "ZNEG"
    ZZERO = "ZZERO"
    ZGRND = "ZGRND"


Channels = {
    "ANGLE = false machine relative rotor angle (degrees).": False,
    "PELEC = false machine electrical power (pu on SBASE).": False,
    "QELEC = false machine reactive power.": False,
    "ETERM = false machine terminal voltage (pu).": False,
    "EFD = false generator main field voltage (pu).": False,
    "PMECH = false turbine mechanical power (pu on MBASE).": False,
    "SPEED = false machine speed deviation from nominal (pu).": False,
    "XADIFD = false machine field current (pu).": False,
    "ECOMP = false voltage regulator compensated voltage (pu).": False,
    "VOTHSG = false stabilizer output signal (pu).": False,
    "VREF = false voltage regulator voltage setpoint (pu).": False,
    "BSFREQ = false bus pu frequency deviations.": False,
    "VOLT = false bus pu voltages (complex).": False,
    "voltage and angle.": False,
    "flow (P).": False,
    "flow (P and Q).": False,
    "flow (MVA).": False,
    "apparent impedance (R and X).": False,
    "ITERM.": False,
    "machine apparent impedance.": False,
    "VUEL = false minimum excitation limiter output signal (pu).": False,
    "VOEL = false maximum excitation limiter output signal (pu).": False,
    "PLOAD.": False,
    "QLOAD.": False,
    "GREF = false turbine governor reference.": False,
    "LCREF = false turbine load control reference.": False,
    "WVLCTY = false wind velocity (m/s).": False,
    "WTRBSP = false wind turbine rotor speed deviation (pu).": False,
    "WPITCH = false pitch angle (degrees).": False,
    "WAEROT = false aerodynamic torque (pu on MBASE).": False,
    "WROTRV = false rotor voltage (pu on MBASE).": False,
    "WROTRI = false rotor current (pu on MBASE).": False,
    "WPCMND = false active power command from wind control (pu on MBASE).": False,
    "WQCMND = false reactive power command from wind control (pu on MBASE).": False,
}


class ExportModes(str, Enum):
    "Valid export modes"

    JSON = "json"
    CSV = "csv"
    H5 = "h5"


class ChannelTypes(str, Enum):
    "Valid channel types"

    BUSES = "buses"
    LOADS = "loads"
    MACHINES = "machines"


class UseModes(str, Enum):
    "Valid use modes"

    REGEX = "regex"
    LIST = "list"
    ALL = "all"


class ApiCommands(str, Enum):
    RUN_SIMUALTION = "run_simulation"
    STATUS = "status"
    SOLVE_STEP = "run_step"
    OPEN_CASE = "open_case"
    CLOSE_CASE = "close_case"
    UDPATE_MODEL = "update_model"
    UPDATE_PARAMETERS = "update_settings"
    QUERY_ALL = "query_all"
    QUERY_BY_ID = "query_by_asset"
    QUERY_BY_PPTY = "query_by_ppty"
    QUERY_ASSET_LIST = "query_asset_list"


class SimulationStatus(str, Enum):
    NOT_INITIALIZED = "Instance not initialized"
    STARTING_INSTANCE = "Starting PyPSSE instance"
    INITIALIZATION_COMPLETE = "PyPSSE initialization complete"
    RUNNING_SIMULATION = "Running simulation"
    SIMULATION_COMPLETE = "Simulation complete"
    STARTING_RESULT_EXPORT = "Starting exports"
    RESULT_EXPORT_COMPLETE = "Export coplete"

class PSSE_VERSIONS(str, Enum):
    PSSE34 = "psse34"
    PSSE35 = "psse35"
    PSSE36 = "psse36"