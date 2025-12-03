import ast
import os
import time
import helics as h
import pandas as pd
from loguru import logger

from pypsse.common import MAPPED_CLASS_NAMES, VALUE_UPDATE_BOUND
from pypsse.models import ExportFileOptions, SimulationSettings
from pypsse.modes.abstract_mode import AbstractMode
from pypsse.profile_manager.common import PROFILE_VALIDATION


class HelicsInterface:
    "Implements the HEILCS interface for PyPSSE"
    all_sub_results = {}
    all_pub_results = {}

    dynamic_iter_const = 1000.0
    n_states = 5
    init_state = 1
    dynamic_params = ("FmA", "FmB", "FmC", "FmD", "Fel")

    def __init__(
        self,
        psse: object,
        sim: AbstractMode,
        settings: SimulationSettings,
        export_settings: ExportFileOptions,
        bus_subsystems: dict,
    ):
        """Sets up the co-simulation federate

        Args:
            psse (object): simulator instance
            sim (AbstractMode): simulator controller inatance
            settings (SimulationSettings): simulation settings
            export_settings (ExportFileOptions): export settings
            bus_subsystems (dict): bus subsystem to bus mapping
        """
        self.bus_pubs = ["bus_id", "bus_Vmag", "bus_Vang", "bus_dev"]
        self.psse = psse
        self.settings = settings
        self.export_settings = export_settings
        self.bus_subsystems = bus_subsystems
        self.sim = sim
        self.c_seconds = 0
        self.c_seconds_old = -1

        self._co_convergance_error_tolerance = settings.helics.error_tolerance
        self._co_convergance_max_iterations = settings.helics.max_coiterations
        self.create_federate()
        self.subsystem_info = []
        self.publications = {}
        self.subscriptions = {}
        self.load_fault = False
        #################################################################
        # FX: add these hardcored load value to better matching
        self.load_power = {
            tid: {
            "transmission_loads_at_fault": at_fault,
            "transmission_loads_clear_fault": clear_fault
            }
            for tid, at_fault, clear_fault in zip(
                settings.simulation.transmission_ids,
                settings.simulation.transmission_loads_at_fault,
                settings.simulation.transmission_loads_clear_fault
            )
        }
        #################################################################

    def enter_execution_mode(self):
        """Enables federate to enter execution mode"""
        itr = 0
        itr_flag = h.helics_iteration_request_iterate_if_needed
        while True:
            itr_status = h.helicsFederateEnterExecutingModeIterative(self.psse_federate, itr_flag)
            logger.debug(f"--- Iter {itr}: Iteration Status = {itr_status}, Passed Iteration Requestion = {itr_flag}")
            if itr_status == h.helics_iteration_result_next_step:
                break

    def create_federate(self):
        """Creates a HELICS co-simulation federate"""
        self.fedinfo = h.helicsCreateFederateInfo()
        # logger.debug(f"self.fedinfo: {self.fedinfo}")
        h.helicsFederateInfoSetCoreName(self.fedinfo, self.settings.helics.federate_name)
        h.helicsFederateInfoSetCoreTypeFromString(self.fedinfo, self.settings.helics.core_type.value)
        h.helicsFederateInfoSetCoreInitString(self.fedinfo, "--federates=1")
        h.helicsFederateInfoSetBroker(self.fedinfo, str(self.settings.helics.broker_ip))
        h.helicsFederateInfoSetBrokerPort(self.fedinfo, self.settings.helics.broker_port)
        IP = self.settings.helics.broker_ip
        Port = self.settings.helics.broker_port
        logger.info("Connecting to broker @ {}".format(f"{IP}:{Port}" if Port else IP))
        logger.info(f"Connecting to broker @ {self.settings.helics}")

        if self.settings.helics.iterative_mode:
            h.helicsFederateInfoSetTimeProperty(
                self.fedinfo,
                h.helics_property_time_delta,
                self.settings.simulation.simulation_step_resolution.total_seconds() / self.dynamic_iter_const,
            )
        else:
            h.helicsFederateInfoSetTimeProperty(
                self.fedinfo,
                h.helics_property_time_delta,
                self.settings.simulation.simulation_step_resolution.total_seconds(),
            )
        self.psse_federate = h.helicsCreateValueFederate(self.settings.helics.federate_name, self.fedinfo)

    def register_publications(self, bus_subsystems: dict):
        """Creates a HELICS publications

        Args:
            bus_subsystems (dict): bus subsystem to bus mapping

        Raises:
            LookupError: returned if invalid subsystem passed
            LookupError: returned if invalid class named passed
            LookupError: returned if property fields do not match class names
        """
        self.publications = {}
        self.pub_struc = []
        for publication_dict in self.settings.helics.publications:
            # logger.debug(f"publication_dict: {publication_dict}")
            bus_subsystem_ids = publication_dict.bus_subsystems
            if not set(bus_subsystem_ids).issubset(self.bus_subsystems):
                msg = f"One or more invalid bus subsystem ID pass in {bus_subsystem_ids}."
                f"Valid subsystem IDs are  '{list(self.bus_subsystems.keys())}'."
                raise LookupError(msg)

            elm_class = publication_dict.asset_type.value

            if not hasattr(self.export_settings, elm_class.lower()):
                msg = f"'{elm_class.lower()}' is not a valid class of elements. "
                f"Valid fields are: {list(self.export_settings.dict().keys())}"
                raise LookupError(msg)

            managed_properties = [ptpy.value for ptpy in getattr(self.export_settings, elm_class.lower())]
            properties = [p.value for p in publication_dict.asset_properties]
            # logger.debug(f"managed_properties: {managed_properties}")
            # logger.debug(f"properties: {properties}")

            if not set(properties).issubset(managed_properties):
                msg = f"One or more publication property defined for class '{elm_class}' is invalid."
                " Properties defined {properties}"
                f"Valid properties for class '{elm_class.lower()}' are '{managed_properties}'"

                raise Exception(msg)

            bus_cluster = []
            for bus_subsystem_id in bus_subsystem_ids:
                bus_cluster.extend([str(x) for x in bus_subsystems[bus_subsystem_id]])

            if elm_class.lower() in MAPPED_CLASS_NAMES:
                elm_class = MAPPED_CLASS_NAMES[elm_class.lower()]

            self.pub_struc.append([{elm_class: properties}, bus_cluster])
            temp_res = self.sim.read_subsystems({elm_class: properties}, bus_cluster)
            temp_res = self.get_restructured_results(temp_res)

            for c_name, elm_info in temp_res.items():
                for name, v_info in elm_info.items():
                    for p_name, val in v_info.items():
                        pub_tag = f"{self.settings.helics.federate_name}.{c_name}.{name}.{p_name}"
                        dtype_matched = True
                        if isinstance(val, float):
                            self.publications[pub_tag] = h.helicsFederateRegisterGlobalTypePublication(
                                self.psse_federate, pub_tag, "double", ""
                            )
                        elif isinstance(val, complex):
                            self.publications[pub_tag] = h.helicsFederateRegisterGlobalTypePublication(
                                self.psse_federate, pub_tag, "complex", ""
                            )
                        elif isinstance(val, int):
                            self.publications[pub_tag] = h.helicsFederateRegisterGlobalTypePublication(
                                self.psse_federate, pub_tag, "integer", ""
                            )
                        elif isinstance(val, list):
                            self.publications[pub_tag] = h.helicsFederateRegisterGlobalTypePublication(
                                self.psse_federate, pub_tag, "vector", ""
                            )
                        elif isinstance(val, str):
                            self.publications[pub_tag] = h.helicsFederateRegisterGlobalTypePublication(
                                self.psse_federate, pub_tag, "string", ""
                            )
                        else:
                            dtype_matched = False
                            logger.warning(f"Publication {pub_tag} could not be registered. Data type not found")
                        if dtype_matched:
                            logger.debug(f"Publication registered: {pub_tag}")

    def register_subscriptions(self):
        """Creates a HELICS subscriptions

        Raises:
            Exception: raised if invalid element type passed
            Exception: raised if property do not match class
            Exception: rasied if invalid property
        """

        self.subscriptions = {}
        assert (
            self.settings.simulation.subscriptions_file
        ), "HELICS co-simulations requires a subscriptions_file property populated"
        sub_data = pd.read_csv(self.settings.simulation.subscriptions_file)
        self.psse_dict = {}
        for _, row in sub_data.iterrows():
            try:
                row["element_property"] = ast.literal_eval(row["element_property"])
            except Exception as e:
                logger.debug(str(e))
            try:
                row["scaler"] = ast.literal_eval(row["scaler"])
            except Exception as e:
                logger.debug(str(e))

            if row["element_type"] not in PROFILE_VALIDATION:
                msg = f"Subscription file error: {row['element_type']} not a valid element_type."
                f"Valid element_type are: {list(PROFILE_VALIDATION.keys())}"
                raise Exception(msg)

            if (
                isinstance(row["element_property"], str)
                and row["element_property"] not in PROFILE_VALIDATION[row["element_type"]]
            ):
                msg = f"Subscription file error: {row['property']} is not valid. "
                f"Valid subtypes for '{row['element_type']}' are: {PROFILE_VALIDATION[row['element_type']]}"
                raise Exception(msg)

            valid_set = set(PROFILE_VALIDATION[row["element_type"]])
            s_set = set(row["element_property"])
            if isinstance(row["element_property"], list) and valid_set.issubset(s_set):
                msg = f"Subscription file error: {row['element_property']} is not a valid subset. "
                f"Valid subtypes for '{row['element_type']}' are: {PROFILE_VALIDATION[row['element_type']]}"
                raise Exception(msg)

            element_id = str(row["element_id"])

            self.subscriptions[row["sub_tag"]] = {
                "bus": row["bus"],
                "element_id": element_id,
                "element_type": row["element_type"],
                "property": row["element_property"],
                "scaler": row["scaler"],
                "dStates": [self.init_state] * self.n_states,
                "subscription": h.helicsFederateRegisterSubscription(self.psse_federate, row["sub_tag"], ""),
            }

            logger.info(
                "{} property of element {}.{} at bus {} has subscribed to {}".format(
                    row["element_property"], row["element_type"], row["element_id"], row["bus"], row["sub_tag"]
                )
            )

            if row["bus"] not in self.psse_dict:
                self.psse_dict[row["bus"]] = {}
            if row["element_type"] not in self.psse_dict[row["bus"]]:
                self.psse_dict[row["bus"]][row["element_type"]] = {}
            if element_id not in self.psse_dict[row["bus"]][row["element_type"]]:
                self.psse_dict[row["bus"]][row["element_type"]][element_id] = {}
            if isinstance(row["element_property"], str):
                if row["element_property"] not in self.psse_dict[row["bus"]][row["element_type"]][element_id]:
                    # FX: modify this so that multiple feeders can be connected to a Transmission Bus
                    self.psse_dict[row["bus"]][row["element_type"]][element_id][row["element_property"]] = []
            elif isinstance(row["element_property"], list):
                for r in row["element_property"]:
                    if r not in self.psse_dict[row["bus"]][row["element_type"]][element_id]:
                        self.psse_dict[row["bus"]][row["element_type"]][element_id][r] = []

    def request_time(self, _) -> (bool, float):
        """Enables time increment of the federate ina  co-simulation."
        Works for both loosely ans tightly coupled co-simulations

        Args:
            self (_type_): _description_
            float (_type_): _description_

        Returns:
            bool: flag for iteration requrirement (rerun same time step)
            float: current helics time in seconds
        """
        
        r_seconds = self.sim.get_total_seconds()  # - self._dss_solver.GetStepResolutionSeconds()
        logger.info(f"Time requested: {r_seconds}")

        if self.sim.get_time() not in self.all_sub_results:
            self.all_sub_results[self.sim.get_time()] = {}
            self.all_pub_results[self.sim.get_time()] = {}

        if not self.settings.helics.iterative_mode:
            while self.c_seconds < r_seconds:
                self.c_seconds = h.helicsFederateRequestTime(self.psse_federate, r_seconds)
            logger.info(f"Time requested: {r_seconds} - time granted: {self.c_seconds} ")
            return True, self.c_seconds
        else:
            itr = 0

            while True:
                self.c_seconds, itr_state = h.helicsFederateRequestTimeIterative(
                    self.psse_federate, r_seconds, h.helics_iteration_request_iterate_if_needed
                )
                if itr_state == h.helics_iteration_result_next_step:
                    logger.debug("\tIteration complete!")
                    break

                error = max([abs(x["dStates"][0] - x["dStates"][1]) for k, x in self.subscriptions.items()])

                subscriptions = self.subscribe()
                for sub_name, sub_value in subscriptions.items():
                    if sub_name not in self.all_sub_results[self.sim.get_time()]:
                        self.all_sub_results[self.sim.get_time()][sub_name] = []
                    self.all_sub_results[self.sim.get_time()][sub_name].append(sub_value)

                self.sim.resolveStep(r_seconds)

                publications = self.publish()
                for pub_name, pub_value in publications.items():
                    if pub_name not in self.all_pub_results[self.sim.get_time()]:
                        self.all_pub_results[self.sim.get_time()][pub_name] = []
                    self.all_pub_results[self.sim.get_time()][pub_name].append(pub_value)

                itr += 1
                logger.debug(f"\titr = {itr}")

                if itr > self.settings.helics.max_coiterations or error < self.settings.helics.error_tolerance:
                    self.c_seconds, itr_state = h.helicsFederateRequestTimeIterative(
                        self.psse_federate, r_seconds, h.helics_iteration_request_no_iteration
                    )
                else:
                    pass

            return True, self.c_seconds

    def get_restructured_results(self, results: dict) -> dict:
        """result restructure for helics interface

        Args:
            results (dict): simulation results from result container

        Returns:
            dict: restructured result
        """

        results_dict = {}
        for k, d in results.items():
            c, p = k.split("_")
            if c not in results_dict:
                results_dict[c] = {}
            for n_raw, v in d.items():
                if isinstance(n_raw, str):
                    n = n_raw.replace(" ", "")
                else:
                    n = str(n_raw)
                if n not in results_dict[c]:
                    results_dict[c][n] = {}
                results_dict[c][n].update({p: v})
        return results_dict

    def publish(self) -> dict:
        """Publishes updated reslts each iteration

        Returns:
            dict: mapping of publication key to result
        """

        pub_results = {}
        for quantities, subsystem_buses in self.pub_struc:
            temp_res = self.sim.read_subsystems(quantities, subsystem_buses)
            temp_res = self.get_restructured_results(temp_res)
            for c_name, elm_info in temp_res.items():
                for name, v_info in elm_info.items():
                    for p_name, val in v_info.items():
                        pub_tag = f"{self.settings.helics.federate_name}.{c_name}.{name}.{p_name}"
                        pub_tag_reduced = f"{c_name}.{name}.{p_name}"
                        pub = self.publications[pub_tag]
                        dtype_matched = True
                        pub_results[pub_tag_reduced] = val
                        if isinstance(val, float):
                            h.helicsPublicationPublishDouble(pub, val)
                        elif isinstance(val, complex):
                            h.helicsPublicationPublishComplex(pub, val.real, val.imag)
                        elif isinstance(val, int):
                            h.helicsPublicationPublishInteger(pub, val)
                        elif isinstance(val, list):
                            h.helicsPublicationPublishVector(pub, val)
                        elif isinstance(val, str):
                            h.helicsPublicationPublishString(pub, val)
                        else:
                            dtype_matched = False
                            logger.warning(f"Publication {pub_tag} not updated")
                        if dtype_matched:
                            logger.debug(f"Publication {pub_tag} published: {val}")
        return pub_results

    def subscribe(self) -> dict:
        """Subscribes results each iteration and updates PSSE objects accordingly

        Returns:
            dict: mapping of subscription key to updated result
        """

        "Subscribes results each iteration and updates PSSE objects accordingly"

        # logger.debug(f"self.psse_dict is {self.psse_dict}")
        for sub_tag, sub_data in self.subscriptions.items():
            if isinstance(sub_data["property"], str):
                sub_data["value"] = h.helicsInputGetDouble(sub_data["subscription"])
                # logger.debug(f"sub_data is {sub_data}")
                self.psse_dict[sub_data["bus"]][sub_data["element_type"]][sub_data["element_id"]][
                    sub_data["property"]
                ].append((sub_data["value"], sub_data["scaler"]))
            elif isinstance(sub_data["property"], list):
                sub_data["value"] = h.helicsInputGetVector(sub_data["subscription"])
                # logger.debug(f"sub_data is {sub_data}")
                if isinstance(sub_data["value"], list) and len(sub_data["value"]) == len(sub_data["property"]):
                    for i, p in enumerate(sub_data["property"]):
                        self.psse_dict[sub_data["bus"]][sub_data["element_type"]][sub_data["element_id"]][p].append((
                            sub_data["value"][i],
                            sub_data["scaler"][i],
                        ))

            logger.debug("Data received {} for tag {}".format(sub_data["value"], sub_tag))
            if self.settings.helics.iterative_mode:
                if self.c_seconds != self.c_seconds_old:
                    sub_data["dStates"] = [self.init_state] * self.n_states
                else:
                    sub_data["dStates"].insert(0, sub_data["dStates"].pop())
        all_values = {}
        for b, b_info in self.psse_dict.items():
            for t, t_info in b_info.items():
                for i, v_dict in t_info.items():
                    values = {}
                    j = 0
                    for p, v_raws in v_dict.items():
                        if isinstance(v_raws, list):
                            if isinstance(p, str):
                                ppty = f"realar{PROFILE_VALIDATION[t].index(p) + 1}"
                                values[ppty] = 0
                                for v_raw in v_raws:
                                    v, scale = v_raw
                                    if isinstance(scale, (float, int)):
                                        values[ppty] += v * scale
                                    else:
                                        values[ppty] += v
                                all_values[f"{t}.{b}.{i}.{p}"] = values[ppty]
                            elif isinstance(p, list):
                                for _, ppt in enumerate(p):
                                    ppty = f"realar{PROFILE_VALIDATION[t].index(ppt) + 1}"
                                    values[ppty] = 0
                                for v_raw in v_raws:
                                    v, scale = v_raw
                                    if isinstance(scale, (float, int)):
                                        values[ppty] += v * scale
                                    else:
                                        values[ppty] += v
                                all_values[f"{t}.{b}.{i}.{p}"] = values[ppty]
                        j += 1
                    is_empty = [0 if not vx else 1 for vx in values.values()]
                    logger.debug(f"{t}.{b}.{i} = {values}")
                    logger.debug(f"current HELICE time: {self.c_seconds}")
                    ######################################################
                    ## FX: add this for better transit matching
                    if round(self.c_seconds, 3) == 0.1 and self.settings.simulation.transmission_loads_markup:
                        logger.debug("the moment of fault")
                        logger.debug(f"old {t}.{b}.{i} = {values}")
                        logger.debug(self.load_power)
                        load_data = self.load_power[b]['transmission_loads_at_fault']
                        values['realar1'] = load_data[0]
                        values['realar2'] = load_data[1]
                        # os.system("PAUSE")
                    if round(self.c_seconds, 3) == 0.175 and self.settings.simulation.transmission_loads_markup:
                        logger.debug("the moment of clearing fault")
                        logger.debug(f"old {t}.{b}.{i} = {values}")
                        load_data = self.load_power[b]['transmission_loads_clear_fault']
                        values['realar1'] = load_data[0]
                        values['realar2'] = load_data[1]
                        # os.system("PAUSE")
                    ######################################################
                    if (
                        sum(is_empty) != 0
                        and sum(values.values()) < VALUE_UPDATE_BOUND
                        and sum(values.values()) > -VALUE_UPDATE_BOUND
                        and self.c_seconds > 0.02
                    ):
                        self.sim.update_object(t, b, i, values)
                        logger.debug(f"{t}.{b}.{i} = {values}")

                    else:
                        logger.debug("write failed")
        
        ######################################################
        ## FX: clear the result list
        for sub_tag, sub_data in self.subscriptions.items():
            if isinstance(sub_data["property"], str):
                self.psse_dict[sub_data["bus"]][sub_data["element_type"]][sub_data["element_id"]][sub_data["property"]] = []
            elif isinstance(sub_data["property"], list):
                if isinstance(sub_data["value"], list) and len(sub_data["value"]) == len(sub_data["property"]):
                    for i, p in enumerate(sub_data["property"]):
                        self.psse_dict[sub_data["bus"]][sub_data["element_type"]][sub_data["element_id"]][p] = []
        logger.debug(f"clear self.psse_dict is {self.psse_dict}")
        ######################################################

        # os.system("PAUSE")
        self.c_seconds_old = self.c_seconds
        return all_values

    def fill_missing_values(self, value):
        "fixes values before model dispatch"
        idx = [f"realar{PROFILE_VALIDATION[self.d_type].index(c) + 1}" for c in self.Columns]
        x = dict(zip(idx, list(value)))
        return x

    def __del__(self):
        "frees helics resources"
        h.helicsFederateFinalize(self.psse_federate)
        h.helicsFederateGetState(self.psse_federate)
        h.helicsFederateInfoFree(self.fedinfo)
        h.helicsFederateFree(self.psse_federate)
        logger.info("HELICS federate for PyDSS destroyed")
