# Handle complex and nominal this way
"Constant definations for the simulation modes"

import copy

dyn_only_options = {
    "Loads": {
        "lmodind": {
            "TD": 14,
            "TC": 15,
            "FmA": 18,
            "FmB": 19,
            "FmC": 20,
            "FmD": 21,
            "Fel": 22,
            "PFel": 23,
            "P1c": 28,
            "P2c": 30,
            "Q1c": 33,
            "Q2c": 35,
        }
    },
}

DYNAMIC_ONLY_PPTY = {
    "Loads": {
        "FmA": ["FmA"],
        "FmB": ["FmB"],
        "FmC": ["FmC"],
        "FmD": ["FmD"],
        "Fel": ["Fel"],
        "PFel": ["PFel"],
        "P1c": ["P1c"],
        "P2c": ["P2c"],
        "Q1c": ["Q1c"],
        "Q2c": ["Q2c"],
        "TD": ["TD"],
        "TC": ["TC"],
    }
}

STANDARD_FORMAT = {
    "Buses": {
        "Number": ["NUMBER"],
        "BusNum": ["NUMBER"],
        "NomkV": ["BASE"],
        "kV": ["KV"],
        "Vpu": ["PU"],
        "Vangle": ["ANGLE"],  # not sure whether degree or radian is expected in NAERM
        "AreaNumber": ["AREA"],
        "ZoneNumber": ["ZONE"],
        "AreaName": ["AREANAME"],
        "SubLongitude": ["LONG"],
        "SubLatitude": ["LATI"],
        "LoadMW": ["TOTAL", "REAL"],
        "LoadMvar": ["TOTAL", "IMAG"],
        "NomB": ["YS", "IMAG", "NOM"],
        "NomG": ["YS", "REAL", "NOM"],
        "GenMW": ["GENPOWER", "REAL"],
        "GenMvar": ["GENPOWER", "IMAG"],
        "SubNumber": ["STATION"],
        "ShuntMVAR": ["YS", "IMAG", "ACT"],
        "Name": ["NAME"],
        "Status": ["STATUS"],
        "IsFeederEligible": ["ISLOADBUS"],
        "BusName": ["NAME"],
        "BusNomVolt": ["BASE"],
        "BusPUVolt": ["PU"],
    },
    "Branches": {
        "LimitMVAA": ["RATEA"],
        "LimitMVAB": ["RATEB"],
        "LimitMVAC": ["RATEC"],
        "Status": ["STATUS"],
        "B": ["BY"],  # BY for susceptance using Star model
        "BusNameFrom": ["FROMBUSNAME"],
        "BusNumFrom": ["FROMBUSNUM"],
        "BusNameTo": ["TOBUSNAME"],
        "BusNumTo": ["TOBUSNUM"],
        "Circuit": ["CIRCUIT"],
        "SubNumberTo": ["SUBNUMBERTO"],
        "SubNumberFrom": ["SUBNUMBERFROM"],
        "BusNum": ["FROMBUSNUM"],
        "BusNum:1": ["TOBUSNUM"],
        "LineCircuit": ["CIRCUIT"],
        "LineStatus": ["STATUS"],
        "LineMaxPercentAmp": ["PCTRTA"],
        "NomkVFrom": ["NOMKVFROM"],
        "NomKVTo": ["NOMKVTO"],
        "R": ["RX", "REAL"],
        "X": ["RX", "IMAG"],
    },
    "Machines": {
        "BusName": ["BUSNAME"],
        "BusNum": ["BUSNUM"],
        "ID": ["MACID"],
        "MWMin": ["PMIN"],
        "MWMax": ["PMAX"],
        "MvarMin": ["QMIN"],
        "MvarMax": ["QMAX"],
        "MW": ["P"],
        "Mvar": ["Q"],
        "Status": ["STATUS"],
        "SubNumber": ["SUBNUMBER"],
        "SubLatitude": ["SUBLATITUDE"],
        "SubLongitude": ["SUBLONGITUDE"],
        "MachineID": ["MACID"],
        "AreaNumber": ["AREANUMBER"],
    },
    "Stations": {
        "SubName": ["SUBNAME"],
        "SubNum": ["SubNum"],
        "Buses": ["BUSES"],
        "Gens": ["GENERATORS"],
        "Trans": ["TRANSFORMERS"],
        "NomkV": ["NOMKV"],
        "LoadMW": ["LOADMW"],
        "GenMW": ["GENMW"],
    },
    "Areas": {"TotalGenMW": ["GEN", "REAL"], "TotalGenMvar": ["GEN", "IMAG"], "AreaNum": ["AREANUMBER"]},
    "Zones": {"TotalGenMW": ["GEN", "REAL"], "TotalGenMvar": ["GEN", "IMAG"], "ZoneName": ["ZONENAME"]},
    "Fixed_shunts": {"ShuntMW": ["ACT", "REAL"], "ShuntMvar": ["ACT", "IMAG"], "FXShuntID": ["FXSHID"]},
    "Loads": {
        "LoadMW": ["MVA", "REAL"],
        "LoadID": ["LOADID"],
        "MW": ["TOTAL", "REAL"],
        "Mvar": ["TOTAL", "IMAG"],
        "SMW": ["MVA", "REAL"],
        "SMvar": ["MVA", "IMAG"],
        "ZMW": ["YL", "REAL"],
        "ZMvar": ["YL", "IMAG"],
        "IMW": ["IL", "REAL"],
        "IMvar": ["IL", "IMAG"],
        "ID": ["LOADID"],
        "BusNum": ["BUSNUM"],
        "Status": ["STATUS"],
    },
    "Induction_generators": {"IndID": ["INDID"]},
    "Switched_shunts": {"BusNum": ["BUSNUM"]},
    "Transformers": {
        "FromBus": ["FROMBUSNUM_3WDG"],
        "ToBus": ["TOBUSNUM_3WDG"],
        "ToBus2": ["TOBUS2NUM_3WDG"],
        "TransCircuit": ["CIRCUIT_3WDG"],
    },
    "DCtransmissionlines": {"BusNum": ["RECT"], "BusNum:1": ["INV"], "DCLID": ["DCLINENAME"]},
}


def converter(func):
    def wrapper(*args, **kwargs):
        new_args = list(args)

        def return_data_type(data_list, cls_type):
            for data in data_list:
                if isinstance(data, cls_type):
                    return data

        if "quantities" in kwargs:
            quantities = kwargs["quantities"]
        else:
            quantities = return_data_type(new_args, dict)

        if "subsystem_buses" in kwargs:
            buses = kwargs["subsystem_buses"]
        else:
            buses = return_data_type(new_args, list)

        new_args = [new_args[0], quantities, buses]

        """ Map NAERM keys to PyPSSE keys """
        ext_string2_info = {}
        complex_conversion_dict = {}
        mapping_dict = copy.deepcopy(quantities)
        for class_name, var_list in quantities.items():
            if class_name in STANDARD_FORMAT:
                new_vars = copy.deepcopy(var_list)
                for v in var_list:
                    if v in STANDARD_FORMAT[class_name]:
                        naerm_element_array = STANDARD_FORMAT[class_name][v]
                        new_vars = [naerm_element_array[0] if el == v else el for el in new_vars]

                        if len(naerm_element_array) == 3:  # noqa: PLR2004
                            ext_string2_info[naerm_element_array[0]] = naerm_element_array[2]
                        if len(naerm_element_array) > 1:
                            if class_name not in complex_conversion_dict:
                                complex_conversion_dict[class_name] = {}
                            complex_conversion_dict[class_name][v] = naerm_element_array[1]
                quantities[class_name] = new_vars

        """ Pass extra string info if a variable requires it """

        if "ext_string2_info" not in kwargs:
            kwargs["ext_string2_info"] = {}
        elif kwargs["ext_string2_info"] is None:
            kwargs["ext_string2_info"] = {}

        kwargs["ext_string2_info"].update(ext_string2_info)
        kwargs["mapping_dict"] = mapping_dict

        result_dict = func(*args, **kwargs)

        """ Convert complex values """

        for class_name_, sub_dict in result_dict.items():
            class_name, param = class_name_.split("_")[0], class_name_.split("_")[1]
            class_name_with_ = [keys for keys in STANDARD_FORMAT.keys() if "_" in keys and keys in class_name_]
            if len(class_name_with_) != 0:
                class_name = class_name_with_[0]
                param = class_name_.split("_")[-1]

            # convert status to string
            status_translation = {1: "connected", 0: "notconnected"}
            if param.lower() == "status":
                for key, value in sub_dict.items():
                    if value == "connected" or "notconnected":
                        sub_dict[key] = value
                    else:
                        sub_dict[key] = status_translation[value]

            if class_name in complex_conversion_dict:
                conv_dict = complex_conversion_dict[class_name]
                if param in conv_dict:
                    for key, value in sub_dict.items():
                        if isinstance(value, complex):
                            if conv_dict[param] == "REAL":
                                sub_dict[key] = value.real
                            elif conv_dict[param] == "IMAG":
                                sub_dict[key] = value.imag

        return result_dict

    return wrapper
