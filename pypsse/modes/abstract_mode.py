# Standard imports

import numpy as np
from loguru import logger
import os

from pypsse.common import CASESTUDY_FOLDER, VALUE_UPDATE_BOUND
from pypsse.enumerations import ModelTypes, WritableModelTypes
from pypsse.models import ExportSettings, SimulationModes, SimulationSettings, ExportFileOptions
from pypsse.modes.constants import converter


class AbstractMode:
    def __init__(
        self,
        psse,
        dyntools,
        settings: SimulationSettings,
        export_settings: ExportFileOptions,
        subsystem_buses,
        raw_data,
    ):
        self.psse = psse
        self.raw_data = raw_data
        self.bus_freq_channels = {}

        from psspy import _f, _i, _o, _s

        self._i = _i
        self._f = _f
        self._s = _s
        self._o = _o
        self.bus_freq_channels = {}

        self.sub_buses = subsystem_buses
        self.dyntools = dyntools
        self.settings = settings
        self.export_settings = export_settings
        self.func_options = {
            ModelTypes.BUSES.value: {
                "busdat": ["BASE", "PU", "KV", "ANGLE", "ANGLED", "NVLMHI", "NVLMLO", "EVLMHI", "EVLMLO"],
                "busdt2": ["TOTAL", "FX_TOTAL", "SC_TOTAL", "YS", "YSW"],  # requires string2 input
                "busint": ["STATION", "TYPE", "AREA", "ZONE", "OWNER", "DUMMY"],
                "arenam": ["AREANAME"],
                "stadat": ["LATI", "LONG"],
                "gendat": ["GENPOWER"],
                "notona": ["NAME"],
                "busexs": ["STATUS"],
                "busnofunc": ["NUMBER", "ISLOADBUS"],
                "frequency": ["FREQ"],
            },
            ModelTypes.STATIONS.value: {
                "stanofunc": ["SUBNAME", "SUBNUMBER", "BUSES", "GENERATORS", "TRANSFORMERS", "NOMKV", "LOADMW", "GENMW"]
            },
            ModelTypes.AREAS.value: {
                "arenofunc": ["AREANAME", "AREANUMBER"],
                "ardat": ["LOAD", "LOADLD", "LDGN", "LDGNLD", "GEN"],
            },
            ModelTypes.ZONES.value: {
                "zonnofunc": ["ZONENAME", "ZONENUMBER"],
                "zndat": ["LOAD", "LOADID", "LDGN", "LDGNLD", "GEN", "LOSS"],
            },
            ModelTypes.DC_LINES.value: {
                "dctnofunc": ["DCLINENAME"],
                "dc2int_2": ["MDC", "RECT", "INV", "METER", "NBR", "NBI", "ICR", "ICI", "NDR", "NDI"],
            },
            ModelTypes.BRANCHES.value: {
                "brndat": [
                    "RATEn",
                    "RATEA",
                    "RATEB",
                    "RATEC",
                    "RATE",
                    "LENGTH",
                    "CHARG",
                    "CHARGZ",
                    "FRACT1",
                    "FRACT2",
                    "FRACT3",
                    "FRACT4",
                ],
                "brndt2": ["RX", "ISHNT", "JSHNT", "RXZ", "ISHNTZ", "JSHNTZ", "LOSSES", "O_LOSSES", "RX"],
                "brnmsc": [
                    "MVA",
                    "AMPS",
                    "PUCUR",
                    "CURANG",
                    "P",
                    "O_P",
                    "Q",
                    "O_Q",
                    "PLOS",
                    "O_PLOS",
                    "QLOS",
                    "O_QLOS",
                    "PCTRTA",
                ],
                "brnint": [
                    "STATUS",
                    "METER",
                    "NMETR",
                    "OWNERS",
                    "OWN1",
                    "OWN2",
                    "OWN3",
                    "OWN4",
                    "STATION_I",
                    "STATION_J",
                    "SECTION_I",
                    "SECTION_J",
                    "NODE_I",
                    "NODE_J",
                    "SCTYPE",
                ],
                "brnnofunc": [
                    "FROMBUSNUM",
                    "TOBUSNUM",
                    "FROMBUSNAME",
                    "TOBUSNAME",
                    "CIRCUIT",
                    "SUBNUMBERTO",
                    "SUBNUMBERFROM",
                    "NOMKVFROM",
                    "NOMKVTO",
                    "BY",
                ],
            },
            ModelTypes.GENERATORS.value: {
                "inddt1": [
                    "MBASE",
                    "RATEKV",
                    "PSET",
                    "RA",
                    "XA",
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
                    "TRQA",
                    "TRQB",
                    "TRQD",
                    "TRQE",
                    "H",
                    "IRATIO",
                    "ROVERX",
                    "RZERO",
                    "XZERO",
                    "RGRND",
                    "XGRND",
                    "P",
                    "O_P",
                    "Q",
                    "O_Q",
                    "MVA",
                    "O_MVA",
                    "SLIP",
                ],
                "inddt2": ["ZA", "Z1", "Z2", "ZZERO", "ZGRND", "PQ", "O_PQ"],
                "indnofunc": ["INDID", "BUSNUM", "BUSNAME"],
            },
            ModelTypes.LOADS.value: {
                "loddt2": ["MVA", "IL", "YL", "TOTAL", "YNEG", "YZERO"],  # required string 2 input
                "lodnofunc": ["LOADID", "BUSNUM", "BUSNAME"],
                "lodint": ["STATION", "SECTION", "STATUS", "AREA", "ZONE", "OWNER", "SCALE", "CGR"],
            },
            ModelTypes.MACHINES.value: {
                "macdat": [
                    "QMAX",
                    "O_QMAX",
                    "QMIN",
                    "O_QMIN",
                    "PMAX",
                    "O_PMAX",
                    "PMIN",
                    "O_PMIN",
                    "MBASE",
                    "MVA",
                    "O_MVA",
                    "P",
                    "O_P",
                    "Q",
                    "O_Q",
                    "PERCENT",
                    "GENTAP",
                    "VSCHED",
                    "WPF",
                    "RMPCT",
                    "RPOS",
                    "XSUBTR",
                    "XTRANS",
                    "XSYNCH",
                ],
                "macdt2": ["PQ", "O_PQ", "ZSORCE", "XTRAN", "ZPOS", "ZNEG", "ZZERO", "ZGRND"],
                "macnofunc": ["MACID", "BUSNUM", "BUSNAME", "SUBNUMBER", "SUBLATITUDE", "SUBLONGITUDE", "AREANUMBER"],
                "macint": ["STATION", "SECTION", "STATUS", "IREG", "NREG", "OWNERS", "WMOD", "PERCENT", "CZG"],
            },
            ModelTypes.FIXED_SHUNTS.value: {
                "fxsdt2": ["ACT", "O_ACT", "NOM", "O_NOM", "PQZERO", "PQZ", "O_PQZ"],
                "fxsnofunc": ["FXSHID", "BUSNUM", "BUSNAME"],
            },
            ModelTypes.SWITCHED_SHUNTS.value: {
                "swsdt1": ["VSWHI", "VSWLO", "RMPCT", "BINIT", "O_BINIT"],
                "swsnofunc": ["BUSNUM", "BUSNAME"],
            },
            ModelTypes.TRANSFORMERS.value: {
                "xfrdat": [
                    "RATIO",
                    "RATIO2",
                    "ANGLE",
                    "RMAX",
                    "RMIN",
                    "VMAX",
                    "VMIN",
                    "STEP",
                    "CR",
                    "CX",
                    "CNXANG",
                    "SBASE1",
                    "NOMV1",
                    "NOMV2",
                    "NOMV2",
                    "GMAGNT",
                    "BMAGNT",
                    "RG1",
                    "XG1",
                    "R01",
                    "X01",
                    "RG2",
                    "XG2",
                    "R02",
                    "X02",
                    "RNUTRL",
                    "XNUTRL",
                ],
                "tr3dt2": ["RX1-2", "RX2-3", "RX3-1", "YMAGNT", "ZG1", "Z01", "ZG2", "Z02", "ZG3", "Z03", "ZNUTRL"],
                "trfnofunc": [
                    "FROMBUSNUM_3WDG",
                    "FROMBUSNUM_2WDG",
                    "TOBUSNUM_3WDG",
                    "TOBUSNUM_2WDG",
                    "TOBUS2NUM_3WDG",
                    "FROMBUSNAME_3WDG",
                    "FROMBUSNAME_2WDG",
                    "TOBUSNAME_3WDG",
                    "TOBUSNAME_2WDG",
                    "TOBUS2NAME_3WDG",
                    "CIRCUIT_2WDG",
                    "CIRCUIT_3WDG",
                ],
            },
        }
        self.initialization_complete = False

    def save_model(self):
        export_path = self.settings.simulation.project_path / CASESTUDY_FOLDER

        i = 0
        file_path = export_path / f"modified_steady_state_{i}.sav"
        while file_path.exists():
            file_path = export_path / f"modified_steady_state_{i}.sav"
            i += 1

        savfile = export_path / f"modified_steady_state_{i}.sav"
        rawfile = export_path / f"modified_steady_state_{i}.raw"
        self.psse.save(str(savfile))
        logger.info(f"file saved - {savfile}")
        self.psse.rawd_2(0, 1, [1, 1, 1, 0, 0, 0, 0], 0, str(rawfile))
        logger.info(f"file saved - {rawfile}")
        if self.settings.simulation.simulation_mode in [SimulationModes.DYNAMIC, SimulationModes.SNAP]:
            snpfile = export_path / f"modified_dynamic_{i}.snp"
            self.psse.snap([-1, -1, -1, -1, -1], str(snpfile))
            logger.info(f"file saved - {snpfile}")
            return [savfile, rawfile, snpfile]

        else:
            return [savfile, rawfile]

    def init(self, bus_subsystems=None):
        assert bus_subsystems is not None
        if self.settings.log.disable_psse_logging:
            self.disable_logging()

    def step(self):
        return

    def resolve_step(self):
        return

    def export(self):
        try:
            logger.info("Starting export process. Can take a few minutes for large files")
            
            outx_file  = str(self.settings.export.outx_file).split("\\")
            outx_file[-1] = self.export_settings.filename_prefix + "_" + outx_file[-1]
            outx_file = "\\".join(outx_file)
            
            achnf = self.dyntools.CHNF(outx_file)
            
            excel_file  = str(self.settings.export.excel_file).split("\\")
            excel_file[-1] = self.export_settings.filename_prefix + "_" + excel_file[-1]
            excel_file = "\\".join(excel_file)
            
            achnf.xlsout(
                channels="",
                show=False,
                xlsfile=excel_file,
                outfile="",
                sheet="Sheet1",
                overwritesheet=True,
            )
            logger.debug(f"Exported {self.settings.export.excel_file}")
        except Exception as e:
            raise e

    def load_user_defined_models(self):
        for mdl in self.settings.simulation.user_models:
            self.psse.addmodellibrary(str(mdl))
            logger.debug(f"User defined library added: {mdl}")

    def load_setup_files(self):
        for f in self.settings.simulation.setup_files:
            ierr = self.psse.runrspnsfile(str(f))
            if ierr:
                msg = f'Error running setup file "{f}"'
                raise Exception(msg)
            else:
                logger.debug(f"Setup file {f} sucessfully run")

    def disable_logging(self):
        self.psse.progress_output(islct=6, filarg="", options=[0, 0])
        self.psse.prompt_output(islct=6, filarg="", options=[0, 0])
        self.psse.report_output(islct=6, filarg="", options=[0, 0])

    def setup_channels(self):
        from pypsse import channel_map

        for channel, add in self.export_settings["Channels"].items():
            if add:
                channel_id = channel_map.channel_map[channel]
                logger.debug(f'"{channel}" added to the channel file')
                self.psse.chsb(0, 1, [-1, -1, -1, 1, channel_id, 0])

    def get_area_numbers(self, subsystem_buses):
        area_numbers = []
        for b in subsystem_buses:
            ierr, val = self.psse.busint(int(b), "AREA")

            if val:
                area_numbers.append(val)
        return list(set(area_numbers))

    def get_substation_numbers(self, subsystem_buses):
        substation_numbers = []
        for b in subsystem_buses:
            irr, val = self.psse.busint(int(b), "STATION")
            if val:
                substation_numbers.append(val)
        return list(set(substation_numbers))

    def get_zone_numbers(self, subsystem_buses):
        zone_numbers = []
        for b in subsystem_buses:
            irr, val = self.psse.busint(int(b), "ZONE")
            if val:
                zone_numbers.append(val)
        return list(set(zone_numbers))

    def get_dctr_line_names(self, subsystem_buses):
        dc_lines = []
        ierr = self.psse.ini2dc()
        if ierr:
            logger.info("No DC line in the model.")
        else:
            ierr, dc_line_name = self.psse.nxtmdc()
            while dc_line_name is not None:
                ierr, rectbus = self.psse.dc2int_2(dc_line_name, "RECT")
                ierr, invbus = self.psse.dc2int_2(dc_line_name, "INV")
                if rectbus in subsystem_buses and invbus in subsystem_buses:
                    dc_lines.append(dc_line_name)

                ierr, dc_line_name = self.psse.nxt2dc()
        return list(set(dc_lines))

    def get_a_list_of_buses_in_substation(self, sub_number, subsystem_buses):
        bus_numbers = []
        for b in subsystem_buses:
            ierr, val = self.psse.busint(int(b), "STATION")

            if val == sub_number:
                bus_numbers.append(int(b))
        return list(set(bus_numbers))

    def get_a_list_of_nomkv_in_substation(self, sub_number, subsystem_buses):
        nom_kvs = []
        for b in subsystem_buses:
            ierr, val = self.psse.busint(int(b), "STATION")

            if val == sub_number:
                ierr, val = self.psse.busdat(int(b), "BASE")

                nom_kvs.append(val)
        return list(set(nom_kvs))

    def get_a_loadmw_in_substation(self, sub_number, subsystem_buses):
        load_mw = 0
        for b in subsystem_buses:
            ierr, val = self.psse.busint(int(b), "STATION")

            if val == sub_number:
                ierr = self.psse.inilod(int(b))

                ierr, load_id = self.psse.nxtlod(int(b))

                while ierr == 0:
                    ierr, val = self.psse.loddt2(int(b), load_id, "TOTAL", "ACT")
                    if isinstance(val, complex):
                        load_mw += val.real
                    ierr, load_id = self.psse.nxtlod(int(b))

        return load_mw

    def get_a_genmw_in_substation(self, sub_number, subsystem_buses):
        gen_mw = 0
        for b in subsystem_buses:
            ierr, val = self.psse.busint(int(b), "STATION")

            if val == sub_number:
                ierr, val = self.psse.gendat(int(b))

                if isinstance(val, complex):
                    gen_mw += val.real
        return gen_mw

    def get_a_list_of_generators_in_substation(self, sub_number, subsystem_buses):
        generators = []
        for b in subsystem_buses:
            ierr, val = self.psse.busint(int(b), "STATION")

            if val == sub_number:
                ierr = self.psse.inimac(int(b))

                ierr, mach_id = self.psse.nxtmac(int(b))

                while ierr == 0:
                    gen_name = f"{b}_{mach_id}"
                    ierr, mach_id = self.psse.nxtmac(int(b))

                    generators.append(gen_name)

        return list(set(generators))

    def get_a_list_of_transformers_in_substation(self, sub_number, subsystem_buses):
        transformers = []
        for b in subsystem_buses:
            ierr, val = self.psse.busint(int(b), "STATION")

            if val == sub_number:
                # Get two winding transformers
                ierr = self.psse.inibrx(int(b), 1)

                ierr, b1, ickt = self.psse.nxtbrn(int(b))

                ickt_string = str(ickt)
                while ierr == 0:
                    ierr, val = self.psse.xfrdat(int(b), int(b1), ickt, "RATIO")

                    if ierr == 0:
                        t_name = f"{b}_{b1}_{ickt_string}"
                        transformers.append(t_name)
                    ierr, b1, ickt = self.psse.nxtbrn(int(b))

                # Get three winding transformers
                ierr = self.psse.inibrx(int(b), 1)

                ierr, b1, b2, ickt = self.psse.nxtbrn3(int(b))

                ickt_string = str(ickt)
                while ierr == 0:
                    ierr, val = self.psse.tr3dt2(int(b), int(b1), int(b2), ickt, "Z01")

                    if ierr == 0:
                        t_name = f"{b!s}_{b1!s}_{b2!s}_{ickt_string}"
                        transformers.append(t_name)
                    ierr, b1, b2, ickt = self.psse.nxtbrn3(int(b))

        return list(set(transformers))

    def check_for_loadbus(self, b):
        ierr = self.psse.inilod(int(b))

        if ierr == 0:
            ierr = self.psse.inimac(int(b))
            if ierr:
                return 1

        return 0

    @converter
    def read_subsystems(self, quantities, subsystem_buses, ext_string2_info=None, mapping_dict=None):
        if ext_string2_info is None:
            ext_string2_info = {}
        if mapping_dict is None:
            mapping_dict = {}
        # logger.debug(f" subsystem_buses : {subsystem_buses}") 
        results = {}
        area_numbers = self.get_area_numbers(subsystem_buses)
        zone_numbers = self.get_zone_numbers(subsystem_buses)
        dctr_lines = self.get_dctr_line_names(subsystem_buses)
        substation_numbers = self.get_substation_numbers(subsystem_buses)

        for class_name, var_list in quantities.items():
            # logger.debug(f"class_name : {class_name}, var_list : {var_list}") 
            # logger.debug(f"self.func_options : {self.func_options}") 
            if class_name in self.func_options:
                funcs = self.func_options[class_name]
                # logger.debug(f"funcs : {funcs}") 
                for id_, v in enumerate(var_list):
                    for func_name, settinsgs in funcs.items():
                        # logger.debug(f"func_name : {func_name}, settinsgs : {settinsgs}") 
                        if v in settinsgs:
                            q = f"{class_name}_{v}"
                            # logger.debug(f"q : {q}") 
                            # os.system("PAUSE")

                            if len(mapping_dict) != 0:
                                if class_name in mapping_dict:
                                    new_v = mapping_dict[class_name][id_]
                                    q = f"{class_name}_{new_v}"

                            if class_name == ModelTypes.AREAS.value:
                                for arr_num in area_numbers:
                                    if func_name in ["arenofunc"]:
                                        if v == "AREANUMBER":
                                            results = self.add_result(results, q, int(arr_num), arr_num)
                                        elif v == "AREANAME":
                                            ierr, val = self.psse.arenam(int(arr_num))

                                            results = self.add_result(results, q, val, arr_num)

                            elif class_name == ModelTypes.STATIONS.value:
                                for sub_num in substation_numbers:
                                    if func_name in ["stanofunc"]:
                                        if v == "SUBNUMBER":
                                            results = self.add_result(results, q, int(sub_num), sub_num)
                                        elif v == "SUBNAME":
                                            ierr, val = self.psse.staname(int(sub_num))

                                            results = self.add_result(results, q, val, sub_num)
                                        elif v == "BUSES":
                                            val = self.get_a_list_of_buses_in_substation(sub_num, subsystem_buses)
                                            results = self.add_result(results, q, val, sub_num)
                                        elif v == "GENERATORS":
                                            val = self.get_a_list_of_generators_in_substation(sub_num, subsystem_buses)
                                            results = self.add_result(results, q, val, sub_num)
                                        elif v == "TRANSFORMERS":
                                            val = self.get_a_list_of_transformers_in_substation(
                                                sub_num, subsystem_buses
                                            )
                                            results = self.add_result(results, q, val, sub_num)
                                        elif v == "NOMKV":
                                            val = self.get_a_list_of_nomkv_in_substation(sub_num, subsystem_buses)
                                            results = self.add_result(results, q, max(val), sub_num)
                                        elif v == "LOADMW":
                                            val = self.get_a_loadmw_in_substation(sub_num, subsystem_buses)
                                            results = self.add_result(results, q, val, sub_num)
                                        elif v == "GENMW":
                                            val = self.get_a_genmw_in_substation(sub_num, subsystem_buses)
                                            results = self.add_result(results, q, val, sub_num)

                            elif class_name == ModelTypes.ZONES.value:
                                for zn_num in zone_numbers:
                                    if func_name in ["zonnofunc"]:
                                        if v == "ZONENUMBER":
                                            results = self.add_result(results, q, int(zn_num), zn_num)
                                        elif v == "ZONENAME":
                                            ierr, val = self.psse.zonnam(int(zn_num))

                                            results = self.add_result(results, q, val, zn_num)

                            elif class_name == ModelTypes.DC_LINES.value:
                                for dcline in dctr_lines:
                                    if func_name == "dctnofunc":
                                        if v == "DCLINENAME":
                                            results = self.add_result(results, q, dcline, dcline)
                                    elif func_name == "dc2int_2":
                                        ierr, val = getattr(self.psse, func_name)(dcline, v)

                                        results = self.add_result(results, q, val, dcline)

                            else:
                                for b in subsystem_buses:
                                    # logger.debug(f"bus : {b}") 
                                    if func_name in [
                                        "busdat",
                                        "busdt2",
                                        "busint",
                                        "arenam",
                                        "notona",
                                        "busexs",
                                        "gendat",
                                        "busnofunc",
                                        "frequency",
                                    ]:
                                        if func_name == "busnofunc":
                                            if v == "NUMBER":
                                                results = self.add_result(results, q, int(b), b)
                                            if v == "ISLOADBUS":
                                                val = self.check_for_loadbus(b)
                                                results = self.add_result(results, q, val, b)

                                        if func_name == "frequency":
                                            if v == "FREQ":
                                                if b in self.bus_freq_channels:
                                                    ierr, val = self.psse.chnval(self.bus_freq_channels[b])

                                                    if not val:
                                                        val = np.NaN
                                                    results = self.add_result(results, q, val * 60.0, int(b))
                                                else:
                                                    results = self.add_result(results, q, 0, int(b))

                                        if func_name in ["busdat", "busint"]:
                                            # if v == 'FREQ':
                                            ierr, val = getattr(self.psse, func_name)(int(b), v)

                                            results = self.add_result(results, q, val, b)

                                        elif func_name == "busdt2":
                                            string2 = "ACT"
                                            if class_name in ext_string2_info:
                                                if v in ext_string2_info[class_name]:
                                                    string2 = ext_string2_info[class_name][v]

                                            ierr, val = getattr(self.psse, func_name)(int(b), v, string2)

                                            results = self.add_result(results, q, val, b)

                                        elif func_name in ["notona", "gendat"]:
                                            ierr, val = getattr(self.psse, func_name)(int(b))

                                            results = self.add_result(results, q, val, b)

                                        elif func_name == "busexs":
                                            val = getattr(self.psse, func_name)(int(b))
                                            val = not val  # PSSE is weird, 0 means bus exist so had to negate
                                            results = self.add_result(results, q, val, b)

                                        elif func_name == "arenam":
                                            ierr, val = self.psse.busint(int(b), "AREA")

                                            if val:
                                                irr, val = getattr(self.psse, func_name)(val)
                                            results = self.add_result(results, q, val, b)

                                    elif func_name == "stadat":
                                        ierr, val = self.psse.busint(int(b), "STATION")

                                        if val:
                                            irr, val = getattr(self.psse, func_name)(val, v)
                                        results = self.add_result(results, q, val, b)

                                    elif func_name in ["inddt1", "inddt2", "indnofunc"]:
                                        ierr = self.psse.iniind(int(b))
                                        if ierr:
                                            # os.system("PAUSE")
                                            logger.info("No induction machine in the model")
                                        else:
                                            ierr, ind_id = self.psse.nxtind(int(b))
                                            while ind_id is not None:
                                                if func_name == "indnofunc":
                                                    if v in ["BUSNUM", "INDID"]:
                                                        val = {"INDID": ind_id, "BUSNUM": int(b)}[v]
                                                        results = self.add_result(results, q, val, f"{b}_{ind_id}")
                                                    elif v == "BUSNAME":
                                                        irr, val = self.psse.notona(int(b))
                                                        results = self.add_result(results, q, val, f"{b}_{ind_id}")
                                                else:
                                                    ierr, val = getattr(self.psse, func_name)(int(b), ind_id, v)
                                                    results = self.add_result(results, q, val, f"{b}_{ind_id}")
                                                ierr, ind_id = self.psse.nxtind(int(b))

                                    elif func_name in ["loddt2", "lodnofunc", "lodint"]:
                                        ierr = self.psse.inilod(int(b))
                                        # logger.debug(f"func_name : {func_name}")
                                        # logger.debug(f"bus : {b}")
                                        ierr, load_id = self.psse.nxtlod(int(b))
                                        # logger.debug(f"first load_id : {load_id}")

                                        while load_id is not None:
                                            if func_name == "lodnofunc":
                                                if v in ["BUSNUM", "LOADID"]:
                                                    val = {"LOADID": load_id, "BUSNUM": int(b)}[v]
                                                    results = self.add_result(results, q, val, f"{b}_{load_id}")
                                                elif v == "BUSNAME":
                                                    ierr, val = self.psse.notona(int(b))
                                                    
                                                    results = self.add_result(results, q, val, f"{b}_{load_id}")
                                            elif func_name == "loddt2":
                                                ierr, val = getattr(self.psse, func_name)(int(b), load_id, v, "ACT")
                                                results = self.add_result(results, q, val, f"{b}_{load_id}")
                                                # if int(b) == 10446 or int(b) == 10466 or int(b) == 10493:
                                                #     logger.debug(f"loddt2 -> int(b): {int(b)}, load_id: {load_id}, v: {v}, ierr : {ierr}, val : {val}")
                                                #     ierr, state_index = self.psse.lmodind(int(b), load_id, "CHARAC", "STATE")
                                                #     assert ierr == 0, f"error={ierr}"
                                                #     if state_index is not None:
                                                #         act_state_index = state_index + 20
                                                #         ierr, value = self.psse.dsrval("STATE", act_state_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Motor D: non-restartable compressor motor A temperature - Load: {int(b)}-{load_id} -> index: {act_state_index}, value:{value}")
                                                #         act_state_index = state_index + 21
                                                #         ierr, value = self.psse.dsrval("STATE", act_state_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Motor D: restartable compressor motor B temperature - Load: {int(b)}-{load_id} -> index: {act_state_index}, value:{value}")
                                                #     ierr, var_index = self.psse.lmodind(int(b), load_id, "CHARAC", "VAR")
                                                #     assert ierr == 0, f"error={ierr}"
                                                #     if var_index is not None:
                                                #         act_var_index = var_index + 0
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Load MVA base - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 12
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Transformer tap - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 13
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Transformer low side voltage real part - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 14
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Transformer low side voltage imaginary part - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 15
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Load bus voltage real part - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 16
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Load bus voltage imaginary part - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 20
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Sum of substation shunt admittance and feeder compensation admittance at substation end - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 21
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Substation shunt admittance - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 24
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Transformer reactance (pu on system MVA base) - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 25
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Static load real part (pu on system MVA base) - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 26
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Static load reactive part (pu on system MVA base) - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 29
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Load bus voltage magnitude - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 30
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Low side bus voltage magnitude - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 37
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Motor D P (pu on system MVA base) - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 38
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Motor D Q (pu on system MVA base) - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 53
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Initial value of Motor D P (pu on system MVA base) - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 54
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Initial value of Motor D Q (pu on system MVA base) - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 55
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Feeder compensation admittance at substation end - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 56
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Feeder compensation admittance at far end - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 60
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Calculated Vstallbrk - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 66
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Initial value of Motor D MVA base - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 117
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Motor D: Bus voltage (pu) - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 118
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Motor D: Bus frequency (pu) - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 119
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Motor D: Aggregated AC unit real power (MW) - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 120
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Motor D: Aggregated AC unit reactive power (MVAr) - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 121
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Motor D: Aggregated AC unit current (pu on system MVA base) - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 124
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Terminal current component on network real axis on system MVA base (pu) - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 125
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Terminal current comp on network imaginary axis on system MVA base (pu) - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 126
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Motor D: non-restartable motor A and restartable motor B Initial Temperature - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 131
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Motor D: KthA non-restartable compressor motor A fraction not tripped by thermal protection - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 133
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Motor D: non-restartable Motor A run / stall state (run=1/stall=0) - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 134
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Motor D: restartable Motor B run / stall state (run=1/stall=0) - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 135
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Motor D: KthB restartable compressor motor B fraction not tripped by thermal protection - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 136
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Internal variable used for determining non-restartable motor A temperature - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 137
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Internal variable used for determining restartable motor B temperature - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 138
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Motor D: Real component of voltage at pervious time step (pu) - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 139
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Motor D: Reactive component of voltage at previous time step (pu) - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 141
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Motor D: Internal variable, P0 for active power at 1.0 pu voltage - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 142
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Motor D: Internal variable, Q0 for reactive power at 1.0 pu voltage - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 143
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Motor D: Computed motor MVA base - Load: {int(b)}-{load_id} -> index: {act_var_index}, value:{value}")

                                                # os.system("PAUSE")
                                            elif func_name == "lodint":
                                                ierr, val = getattr(self.psse, func_name)(int(b), load_id, v)

                                                results = self.add_result(results, q, val, f"{b}_{load_id}")
                                        #     logger.debug(f"val : {val}")
                                        #     logger.debug(f"v : {v}")
                                            ierr, load_id = self.psse.nxtlod(int(b))
                                        #     logger.debug(f"next load_id : {load_id}")
                                        # logger.debug(f"results : {results}")
                                        # os.system("PAUSE")

                                    elif func_name in ["macdat", "macdt2", "macnofunc", "macint"]:
                                        ierr = self.psse.inimac(int(b))
                                        ierr, mach_id = self.psse.nxtmac(int(b))
                                        while mach_id is not None:
                                            if func_name == "macnofunc":
                                                if v in ["BUSNUM", "MACID"]:
                                                    val = {"BUSNUM": int(b), "MACID": mach_id}[v]
                                                    results = self.add_result(results, q, val, f"{b}_{mach_id}")
                                                elif v == "BUSNAME":
                                                    ierr, val = self.psse.notona(int(b))
                                                    results = self.add_result(results, q, val, f"{b}_{mach_id}")
                                                elif v == "SUBNUMBER":
                                                    ierr, val = self.psse.busint(int(b), "STATION")

                                                    results = self.add_result(results, q, val, f"{b}_{mach_id}")

                                                elif v == "AREANUMBER":
                                                    ierr, val = self.psse.busint(int(b), "AREA")

                                                    results = self.add_result(results, q, val, f"{b}_{mach_id}")

                                                elif v in ["SUBLATITUDE", "SUBLONGITUDE"]:
                                                    sub_dict = {"SUBLATITUDE": "LATI", "SUBLONGITUDE": "LONG"}
                                                    ierr, val = self.psse.busint(int(b), "STATION")

                                                    if val:
                                                        ierr, val = self.psse.stadat(val, sub_dict[v])

                                                    results = self.add_result(results, q, val, b)
                                            else:
                                                logger.debug(f"machine -> int(b): {int(b)}, mach_id: {mach_id}, func_name: {func_name}, v: {v}")
                                                # if mach_id == "PV":
                                                #     ierr, state_index = self.psse.windmind(int(b), mach_id, 'WGEN', 'STATE')
                                                #     assert ierr == 0, f"error={ierr}"
                                                #     if state_index is not None:
                                                #         act_state_index = state_index + 0
                                                #         ierr, value = self.psse.dsrval("STATE", act_state_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"voltage measurement lag: {int(b)}-{mach_id} -> index: {act_state_index}, value:{value}")
                                                #         act_state_index = state_index + 1
                                                #         ierr, value = self.psse.dsrval("STATE", act_state_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"generator power measurement lag: {int(b)}-{mach_id} -> index: {act_state_index}, value:{value}")
                                                #         act_state_index = state_index + 2
                                                #         ierr, value = self.psse.dsrval("STATE", act_state_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"first order lag for reactive current: {int(b)}-{mach_id} -> index: {act_state_index}, value:{value}")
                                                #         act_state_index = state_index + 3
                                                #         ierr, value = self.psse.dsrval("STATE", act_state_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"lag block representing inner current loop for iq: {int(b)}-{mach_id} -> index: {act_state_index}, value:{value}")
                                                #         act_state_index = state_index + 4
                                                #         ierr, value = self.psse.dsrval("STATE", act_state_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"lag block for output of multiplier block: {int(b)}-{mach_id} -> index: {act_state_index}, value:{value}")
                                                #         act_state_index = state_index + 8
                                                #         ierr, value = self.psse.dsrval("STATE", act_state_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"first order lag for Pord: {int(b)}-{mach_id} -> index: {act_state_index}, value:{value}")
                                                #         act_state_index = state_index + 9
                                                #         ierr, value = self.psse.dsrval("STATE", act_state_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"lag block representing inner current loop for id: {int(b)}-{mach_id} -> index: {act_state_index}, value:{value}")
                                                #     ierr, var_index = self.psse.windmind(int(b), mach_id, 'WGEN', 'VAR')
                                                #     assert ierr == 0, f"error={ierr}"
                                                #     if var_index is not None:
                                                #         act_var_index = var_index + 0
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"bus voltage reference (Vref0): {int(b)}-{mach_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 1
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"power factor angle reference (pfaref), radians: {int(b)}-{mach_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 3
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"active power reference (Pref): {int(b)}-{mach_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 4
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"reactive power reference (Qref): {int(b)}-{mach_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 7
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Ipcmd: {int(b)}-{mach_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 8
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"Iqcmd: {int(b)}-{mach_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 11
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"output of undervoltage voltage multiplier block: {int(b)}-{mach_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 12
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"output of overvoltage voltage multiplier block: {int(b)}-{mach_id} -> index: {act_var_index}, value:{value}")
                                                #         act_var_index = var_index + 22
                                                #         ierr, value = self.psse.dsrval("VAR", act_var_index)
                                                #         assert ierr == 0, f"error={ierr}"
                                                #         logger.debug(f"reactive current injection, Iqinj: {int(b)}-{mach_id} -> index: {act_var_index}, value:{value}")
                                                # # os.system("PAUSE")
                                                ierr, val = getattr(self.psse, func_name)(int(b), mach_id, v)

                                                results = self.add_result(results, q, val, f"{b}_{mach_id}")
                                            ierr, mach_id = self.psse.nxtmac(int(b))

                                    elif func_name in ["fxsdt2", "fxsnofunc"]:
                                        ierr = self.psse.inifxs(int(b))

                                        ierr, fx_id = self.psse.nxtfxs(int(b))

                                        while fx_id is not None:
                                            if func_name == "fxsnofunc":
                                                if v in ["BUSNUM", "FXSHID"]:
                                                    val = {"BUSNUM": int(b), "FXSHID": fx_id}[v]
                                                    results = self.add_result(results, q, val, f"{b}_{fx_id}")
                                                elif v == "BUSNAME":
                                                    ierr, val = self.psse.notona(int(b))

                                                    results = self.add_result(results, q, val, f"{b}_{fx_id}")
                                            else:
                                                ierr, val = getattr(self.psse, func_name)(int(b), fx_id, v)

                                                results = self.add_result(results, q, val, f"{b}_{fx_id}")
                                            ierr, fx_id = self.psse.nxtfxs(int(b))

                                    elif func_name in ["swsdt1", "swsnofunc"]:
                                        if func_name == "swsnofunc":
                                            ierr, val = self.psse.swsint(int(b), "STATUS")

                                        else:
                                            ierr, val = getattr(self.psse, func_name)(int(b), v)

                                        if ierr == 0:
                                            if func_name == "nofunc":
                                                if v == "BUSNUM":
                                                    val = int(b)
                                                elif v == "BUSNAME":
                                                    ierr, val = self.psse.notona(int(b))

                                            results = self.add_result(results, q, val, b)

                                    elif func_name in ["brndat", "brndt2", "brnmsc", "brnint", "brnnofunc"]:
                                        ierr = self.psse.inibrx(int(b), 1)

                                        ierr, b1, ickt = self.psse.nxtbrn(int(b))

                                        ickt_string = str(ickt)
                                        while ierr == 0:
                                            if func_name == "brnnofunc":
                                                if v in ["FROMBUSNUM", "TOBUSNUM", "CIRCUIT"]:
                                                    val = {
                                                        "FROMBUSNUM": int(b),
                                                        "TOBUSNUM": int(b1),
                                                        "CIRCUIT": ickt_string,
                                                    }[v]
                                                    results = self.add_result(
                                                        results, q, val, f"{b!s}_{b1!s}_{ickt_string}"
                                                    )
                                                elif v == "FROMBUSNAME":
                                                    ierr, val = self.psse.notona(int(b))

                                                    results = self.add_result(
                                                        results, q, val, f"{b!s}_{b1!s}_{ickt_string}"
                                                    )
                                                elif v == "TOBUSNAME":
                                                    ierr, val = self.psse.notona(int(b1))

                                                    results = self.add_result(
                                                        results, q, val, f"{b!s}_{b1!s}_{ickt_string}"
                                                    )
                                                elif v in ["SUBNUMBERTO", "SUBNUMBERFROM"]:
                                                    sub_dict = {"SUBNUMBERFROM": int(b), "SUBNUMBERTO": int(b1)}
                                                    ierr, val = self.psse.busint(sub_dict[v], "STATION")

                                                    results = self.add_result(
                                                        results, q, val, f"{b!s}_{b1!s}_{ickt_string}"
                                                    )
                                                elif v == "NOMKVFROM":
                                                    ierr, val = self.psse.busdat(int(b), "BASE")

                                                    results = self.add_result(
                                                        results, q, val, f"{b!s}_{b1!s}_{ickt_string}"
                                                    )
                                                elif v == "NOMKVTO":
                                                    ierr, val = self.psse.busdat(int(b1), "BASE")

                                                    results = self.add_result(
                                                        results, q, val, f"{b!s}_{b1!s}_{ickt_string}"
                                                    )
                                                elif v == "BY":
                                                    ierr, vali = self.psse.brndt2(int(b), int(b1), "ISHNT", v)

                                                    ierr, valj = self.psse.brndt2(int(b), int(b1), "JSHNT", v)

                                                    if isinstance(vali, complex) and isinstance(valj, complex):
                                                        val = 0.5 * vali.imag + 0.5 * valj.imag
                                                    else:
                                                        val = None
                                                    results = self.add_result(
                                                        results, q, val, f"{b!s}_{b1!s}_{ickt_string}"
                                                    )

                                            else:
                                                ierr, val = getattr(self.psse, func_name)(int(b), int(b1), str(ickt), v)

                                                if ierr == 0:
                                                    results = self.add_result(
                                                        results, q, val, f"{b!s}_{b1!s}_{ickt_string}"
                                                    )
                                            ierr, b1, ickt = self.psse.nxtbrn(int(b))

                                    elif func_name in ["xfrdat", "tr3dt2", "trfnofunc"]:
                                        ierr = self.psse.inibrx(int(b), 1)

                                        ierr, b1, ickt = self.psse.nxtbrn(int(b))

                                        ickt_string = str(ickt)
                                        while ierr == 0:
                                            if func_name == "xfrdat":
                                                ierr, val = getattr(self.psse, func_name)(int(b), int(b1), ickt, v)

                                                if ierr == 0:
                                                    results = self.add_result(
                                                        results, q, val, f"{b!s}_{b1!s}_{ickt_string}"
                                                    )
                                            elif func_name == "trfnofunc":
                                                if v.split("_")[1] == "2WDG":
                                                    if v in ["FROMBUSNUM_2WDG", "TOBUSNUM_2WDG", "CIRCUIT_2WDG"]:
                                                        val = {
                                                            "FROMBUSNUM_2WDG": int(b),
                                                            "TOBUSNUM_2WDG": int(b1),
                                                            "CIRCUIT_2WDG": ickt,
                                                        }[v]
                                                        results = self.add_result(
                                                            results,
                                                            q,
                                                            val,
                                                            f"{b!s}_{b1!s}_{ickt_string}",
                                                        )
                                                    elif v == "FROMBUSNAME_2WDG":
                                                        ierr, val = self.psse.notona(int(b))

                                                        results = self.add_result(
                                                            results,
                                                            q,
                                                            val,
                                                            f"{b!s}_{b1!s}_{ickt_string}",
                                                        )
                                                    elif v == "TOBUSNAME_2WDG":
                                                        ierr, val = self.psse.notona(int(b1))

                                                        results = self.add_result(
                                                            results,
                                                            q,
                                                            val,
                                                            f"{b!s}_{b1!s}_{ickt_string}",
                                                        )

                                            ierr, b1, ickt = self.psse.nxtbrn(int(b))

                                        ierr = self.psse.inibrx(int(b), 1)

                                        ierr, b1, b2, ickt = self.psse.nxtbrn3(int(b))

                                        ickt_string = str(ickt)
                                        while ierr == 0:
                                            if func_name == "tr3dt2":
                                                ierr, val = getattr(self.psse, func_name)(
                                                    int(b), int(b1), int(b2), ickt, v
                                                )

                                                if ierr == 0:
                                                    results = self.add_result(
                                                        results,
                                                        q,
                                                        val,
                                                        f"{b!s}_{b1!s}_{b2!s}_{ickt_string}",
                                                    )
                                            elif func_name == "trfnofunc":
                                                if v.split("_")[1] == "3WDG":
                                                    if v in [
                                                        "FROMBUSNUM_3WDG",
                                                        "TOBUSNUM_3WDG",
                                                        "TOBUS2NUM_3WDG",
                                                        "CIRCUIT_3WDG",
                                                    ]:
                                                        val = {
                                                            "FROMBUSNUM_3WDG": int(b),
                                                            "TOBUSNUM_3WDG": int(b1),
                                                            "CIRCUIT_3WDG": ickt,
                                                            "TOBUS2NUM_3WDG": int(b2),
                                                        }[v]
                                                        results = self.add_result(
                                                            results,
                                                            q,
                                                            val,
                                                            f"{b!s}_{b1!s}_{ickt_string}",
                                                        )
                                                    elif v == "FROMBUSNAME_3WDG":
                                                        ierr, val = self.psse.notona(int(b))

                                                        results = self.add_result(
                                                            results,
                                                            q,
                                                            val,
                                                            f"{b!s}_{b1!s}_{ickt_string}",
                                                        )
                                                    elif v == "TOBUSNAME_3WDG":
                                                        ierr, val = self.psse.notona(int(b1))

                                                        results = self.add_result(
                                                            results,
                                                            q,
                                                            val,
                                                            f"{b!s}_{b1!s}_{ickt_string}",
                                                        )
                                                    elif v == "TOBUS2NAME_3WDG":
                                                        ierr, val = self.psse.notona(int(b1))

                                                        results = self.add_result(
                                                            results,
                                                            q,
                                                            val,
                                                            f"{b!s}_{b1!s}_{ickt_string}",
                                                        )
                                            ierr, b1, b2, ickt = self.psse.nxtbrn3(int(b))

        return results

    def add_result(self, results_dict, class_name, value, label):
        if value or value == 0:
            if class_name not in results_dict:
                results_dict[class_name] = {}
            results_dict[class_name][label] = value
        else:
            if class_name not in results_dict:
                results_dict[class_name] = {}
            results_dict[class_name][label] = None
        return results_dict

    def convert_load(self, bus_subsystem=None):
        if self.settings.loads.convert:
            p1 = self.settings.loads.active_load.constant_current_percentage
            p2 = self.settings.loads.active_load.constant_admittance_percentage
            q1 = self.settings.loads.reactive_load.constant_current_percentage
            q2 = self.settings.loads.reactive_load.constant_admittance_percentage
            if bus_subsystem:
                self.psse.conl(bus_subsystem, 0, 1, [0, 0], [p1, p2, q1, q2])  # initialize for load conversion.
                self.psse.conl(bus_subsystem, 0, 2, [0, 0], [p1, p2, q1, q2])  # convert loads.
                self.psse.conl(bus_subsystem, 0, 3, [0, 0], [p1, p2, q1, q2])  # postprocessing housekeeping.
            else:
                self.psse.conl(0, 1, 1, [0, 0], [p1, p2, q1, q2])  # initialize for load conversion.
                self.psse.conl(0, 1, 2, [0, 0], [p1, p2, q1, q2])  # convert loads.
                self.psse.conl(0, 1, 3, [0, 0], [p1, p2, q1, q2])  # postprocessing housekeeping.

    def update_object(self, dtype, bus, element_id, values: dict):
        val = sum(list(values.values()))
        if val > -VALUE_UPDATE_BOUND and val < VALUE_UPDATE_BOUND:
            if dtype == WritableModelTypes.LOAD.value:
                ierr = self.psse.load_chng_5(ibus=int(bus), id=element_id, **values)
                # ierr, cmpval = self.psse.loddt2(int(bus),element_id,'MVA','ACT')
                # if ierr == 0:
                #     logger.info(f"recheck load value: {cmpval}")
                #     os.system("PAUSE")
            elif dtype == WritableModelTypes.GENERATOR.value:
                ierr = self.psse.induction_machine_data(ibus=int(bus), id=element_id, **values)
            elif dtype == WritableModelTypes.MACHINE.value:
                ierr = self.psse.machine_data_2(i=int(bus), id=element_id, **values)
            elif dtype == WritableModelTypes.PLANT.value:
                ierr = self.psse.plant_data_4(ibus=int(bus), inode=element_id, **values)
            else:
                ierr = 1

            if ierr == 0:
                logger.info(f"Profile Manager: {dtype} '{element_id}' on bus '{bus}' has been updated. {values}")
            else:
                logger.error(f"Profile Manager: Error updating {dtype} '{element_id}' on bus '{bus}'.")

    def has_converged(self):
        return self.psse.solved()
