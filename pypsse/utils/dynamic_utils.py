import os
from typing import ClassVar, List

import pandas as pd
from loguru import logger

from pypsse.common import MACHINE_CHANNELS
from pypsse.modes.constants import dyn_only_options
import re
import toml
from pypsse.models import REGCA1_data_model, REECA1_data_model, REPCA1_data_model

class DynamicUtils:
    "Utility functions for dynamic simulations"

    dynamic_params: ClassVar[List[str]] = ["FmA", "FmB", "FmC", "FmD", "Fel"]

    def disable_generation_for_coupled_buses(self):
        """Disables generation of coupled buses (co-simulation mode only)"""
        if ((self.settings.helics and self.settings.helics.cosimulation_mode and self.settings.simulation.disable_generation_on_coupled_buses) # cosim mode, load in distribution level
            or (self.settings.simulation.disable_generation_on_coupled_buses and self.settings.simulation.generation_model_level.lower() == "transmission")): 
            sub_data = pd.read_csv(self.settings.simulation.subscriptions_file)
            sub_data = sub_data[sub_data["element_type"] == "Load"]
            generators = {}
            generator_list = {}

            for gen_bus, gen_id in self.raw_data.generators:
                if gen_bus not in generator_list:
                    generator_list[gen_bus] = []
                generator_list[gen_bus].append(gen_id)

            for _, row in sub_data.iterrows():
                bus = row["bus"]
                generators[bus] = generator_list[bus]

            for bus_id, machines in generators.items():
                for machine in machines:
                    intgar = [0, self._i, self._i, self._i, self._i, self._i]
                    realar = [
                        self._f,
                        self._f,
                        self._f,
                        self._f,
                        self._f,
                        self._f,
                        self._f,
                        self._f,
                        self._f,
                        self._f,
                        self._f,
                        self._f,
                        self._f,
                        self._f,
                        self._f,
                        self._f,
                        self._f,
                    ]
                    self.psse.machine_chng_2(bus_id, machine, intgar, realar)
                    logger.info(f"Machine disabled: {bus_id}_{machine}")
            
            if self.settings.simulation.generation_model_level.lower() == "transmission" and len(self.settings.simulation.transmission_ibrs) > 0:
                intgar = [0, self._i, self._i, self._i, self._i, self._i]
                realar = [
                    self._f,
                    self._f,
                    self._f,
                    self._f,
                    self._f,
                    self._f,
                    self._f,
                    self._f,
                    self._f,
                    self._f,
                    self._f,
                    self._f,
                    self._f,
                    self._f,
                    self._f,
                    self._f,
                    self._f,
                ]
                transmission_ibrs_id = self.settings.simulation.transmission_ibrs
                transmission_ibrs_std = self.settings.simulation.transmission_ibrs_std
                assert len(transmission_ibrs_id)==len(transmission_ibrs_std), f"Input dimension mismatch"
                # added by FX for IBR temporarily
                ibr_setting_path = r"C:\Users\FXIE\Documents\GitHub\NEARM_TnD\dynamic_cosim_test_cases-model\WECC\Transmission\pypsse_model\Scenarios\IEEE2800\IBR.toml"
                ibr_settings = toml.load(ibr_setting_path)
                for ibr, std in zip(transmission_ibrs_id, transmission_ibrs_std):
                    # generation is still in transmission level but use different modeling method
                    if std.lower() == "ieee2800":
                        logger.debug(f"Added IBR with IEEE STD 2800 for generator {ibr}")
                        bus_id, machine = ibr.split("_")
                        ierr, machine_pg = self.psse.macdat(int(bus_id),machine,'P')
                        ierr, machine_qg = self.psse.macdat(int(bus_id),machine,'Q')
                        ierr, machine_base = self.psse.macdat(int(bus_id),machine,'MBASE')
                        ierr, machine_status = self.psse.macint(int(bus_id),machine,'STATUS')
                        if ibr in ibr_settings.keys():
                            ibr_setting = ibr_settings[ibr]
                            logger.debug(f"REPCA1: {ibr_setting['REPCA1']}")
                            param_repca1 = REPCA1_data_model(**ibr_setting["REPCA1"])
                            param_reeca1 = REECA1_data_model(**ibr_setting["REECA1"])
                            param_regca1 = REGCA1_data_model(**ibr_setting["REGCA1"])
                        else:
                            param_repca1 = None
                            param_reeca1 = None
                            param_regca1 = None
                        if ierr == 0 and machine_status == 1:
                            self.psse.machine_chng_2(int(bus_id), machine, intgar, realar)
                            ibr_id = f"ir"
                            logger.info(f"IBR added: {bus_id}_{machine} (pg={machine_pg},qg={machine_qg},base={machine_base})")
                            ibr_dt = self.settings.simulation.simulation_step_resolution.total_seconds()
                            self.der.add_ibr([int(bus_id)],[machine_pg],[machine_qg],[ibr_id],ibr_dt,param_repca1,param_reeca1,param_regca1)
                        else:
                            logger.warning(f"Can't add IBR: {bus_id}_{machine})")
                    else:
                        # todo
                        raise Exception("Standard has not been implemented")
                logger.debug(f"self.der.ibr['model']: {self.der.ibr['model']}")
        
        # 
        # os.system("PAUSE")

    def update_loadchannel_asset_list(self,assset_type, asset):
        for n in range(len(self.export_settings.channel_setup)):
            channel = self.export_settings.channel_setup[n]
            method_type = channel.asset_type
            if method_type == assset_type:
                logger.info(channel)
                asset_list = channel.asset_list
                if asset not in asset_list:
                    asset_list.append(asset)
                self.export_settings.channel_setup[n].asset_list = asset_list


    def disable_load_models_for_coupled_buses(self):
        """Disables loads of coupled buses (co-simulation mode only)"""
        if self.settings.helics and self.settings.helics.cosimulation_mode:
            sub_data = pd.read_csv(self.settings.simulation.subscriptions_file)
            sub_data = sub_data[sub_data["element_type"] == "Load"]
            ### add by Aadil
            logger.debug("Implementing load brek logic for dynamic cosimulations")
            self.break_loads_for_dynamic_cosimulations(loads=None, components_to_replace=["FmD"])

            self.psse_dict = {}
            for _, row in sub_data.iterrows():
                bus = row["bus"]
                load = row["element_id"]
                ierr = self.psse.ldmod_status(0, int(bus), str(load), 1, 0)
                if ierr == 0:
                    logger.info(f"Dynamic model for load {load} connected to bus {bus} has been disabled")
                elif ierr == 5:
                    logger.error(f"No dynamic model found for load {load} connected to bus {bus}")
                else:
                    raise Exception(f"error={ierr}")

    def break_loads_for_dynamic_cosimulations(self, loads: list = None, components_to_replace: List[str] = []):
        """Implements the load split logic

        Args:
            loads (list, optional): list of coupled loads. Defaults to None.
            components_to_replace (List[str], optional): components to be simulated on distribution side. Defaults to [].
        """
        logger.info("##################### break_loads #########################")
        components_to_stay = [x for x in self.dynamic_params if x not in components_to_replace]
        logger.info(f"components_to_stay: {components_to_stay}")
        logger.info(f"components_to_replace: {components_to_replace}")
        # os.system("PAUSE")
        if loads is None:
            loads = self._get_coupled_loads()
        logger.debug("Fetching static data for loads")
        loads = self._get_load_static_data(loads)
        # os.system("PAUSE")
        logger.debug("Fetching dynamic data for loads")
        loads = self._get_load_dynamic_data(loads)
        logger.debug("Creating dummy loads for coupled buses")
        loads = self._replicate_coupled_load(loads, components_to_replace)
        # os.system("PAUSE")
        logger.debug("Updating dynamic parameters for load models")
        self._update_dynamic_parameters(loads, components_to_stay, components_to_replace)
        # os.system("PAUSE")

    def _update_dynamic_parameters(self, loads: dict, components_to_stay: list, components_to_replace: list):
        """Updates dynamic parameters of composite old / replicated load models

        Args:
            loads (dict): load dictionary
            components_to_stay (list): components to be simulated on transmission side
            components_to_replace (list): components to be simulated on distribution side
        """
        logger.info(f"_update_dynamic_parameters")
        new_percentages = {}
        for load in loads:
            count = 0
            for comp in components_to_stay:
                count += load[comp]
            for comp in components_to_stay:
                if count == 0:
                    new_percentages[comp] = 0
                else:
                    new_percentages[comp] = load[comp] / count
            for comp in components_to_replace:
                new_percentages[comp] = 0.0

            settings = self._get_load_dynamic_properties(load)

            for k, v in new_percentages.items():
                idx = dyn_only_options["Loads"]["lmodind"][k]
                settings[idx] = v
                logger.debug(f"Dynamic model parameters for load {load['bus']}/{load['id']} at bus XX --> index{idx}, value{v}.")
                # self.psse.change_ldmod_con(load['bus'], 'XX' ,r"""CMLDBLU2""" ,idx ,v)
            values = list(settings.values())
            self.psse.add_load_model(load["bus"], "XX", 0, 1, r"""CMLDBLU2""", 2, [0, 0], ["", ""], 133, values)
            # self.psse.add_load_model(load["bus"], "A", 0, 1, r"""CMLDBLU2""", 2, [0, 0], ["", ""], 133, values)
            # self.psse.add_load_model(load["bus"], "B", 0, 1, r"""CMLDBLU2""", 2, [0, 0], ["", ""], 133, values)
            # self.psse.add_load_model(load["bus"], "C", 0, 1, r"""CMLDBLU2""", 2, [0, 0], ["", ""], 133, values)
            # self.psse.add_load_model(load["bus"], "D", 0, 1, r"""CMLDBLU2""", 2, [0, 0], ["", ""], 133, values)
            # self.psse.add_load_model(load["bus"], "F", 0, 1, r"""CMLDBLU2""", 2, [0, 0], ["", ""], 133, values)
            # self.psse.add_load_model(load["bus"], "S", 0, 1, r"""CMLDBLU2""", 2, [0, 0], ["", ""], 133, values)
            logger.info(f"Dynamic model parameters for load {load['id']} at bus 'XX' changed. New settings are {values}.")


    def _get_load_dynamic_properties(self, load):
        "Returns dynamic parameters of composite load models"
        settings = {}
        for i in range(133):
            ierr, con_index = self.psse.lmodind(load["bus"], str(load["id"]), "CHARAC", "CON")
            if con_index is not None:
                act_con_index = con_index + i
                ierr, value = self.psse.dsrval("CON", act_con_index)
                assert ierr == 0, f"error={ierr}"
                settings[i] = value
        return settings
    
    def _get_bus_generation(self, bus):
        "Returns the total generation on an certain bus"
        generator_list = {}
        generators = {}
        for gen_bus, gen_id in self.raw_data.generators:
            if gen_bus not in generator_list:
                generator_list[gen_bus] = []
            generator_list[gen_bus].append(gen_id)
        try:
            generators[bus] = generator_list[bus]
        except:
            raise Exception(f"Can't get generator on bus {bus}")
            # logger.warning(f"Can't get generator on bus {bus}")
        bus_total_p = 0
        bus_total_q = 0
        for bus_id, machines in generators.items():
            for machine in machines:
                ierr, ival = self.psse.macint(bus_id, machine, 'STATUS')
                if ival == 1:
                    ierr, cmpval = self.psse.macdt2(bus_id, machine, 'PQ')
                    assert ierr == 0, f"error={ierr}"
                    logger.debug(f"{bus_id}.{machine} generation: {cmpval}")
                else:
                    cmpval = complex(0,0)
                    logger.debug(f"{bus_id}.{machine} offline generation: {cmpval}")
                bus_total_p += cmpval.real
                bus_total_q += cmpval.imag
        return bus_total_p, bus_total_q
    
    def _replicate_coupled_load(self, loads: dict, components_to_replace: list):
        """create a replica of composite load model

        Args:
            loads (dict): load dictionary
            components_to_replace (list): composite load models to replace on distribution side

        Returns:
            dict: updated load dictionary
        """

        for load in loads:
            logger.info(f"load : {load}")
            dynamic_percentage = load["FmA"] + load["FmB"] + load["FmC"] + load["FmD"] + load["Fel"]
            static_percentage = 1.0 - dynamic_percentage
            logger.debug(f"Static properties - Load: Static ->  value:{static_percentage}")
            for comp in components_to_replace:
                static_percentage += load[comp]
            remaining_load = 1 - static_percentage
            total_load = load["MVA"]
            total_distribution_load = total_load * static_percentage
            total_transmission_load = total_load * remaining_load
            # ceate new load
            self.psse.load_data_5(
                load["bus"],
                "XX",
                realar=[total_transmission_load.real, total_transmission_load.imag, 0.0, 0.0, 0.0, 0.0],
                # lodtyp="replica",
            )
            if (self.settings.helics and self.settings.helics.cosimulation_mode and self.settings.helics.disable_generation_on_coupled_buses 
                and self.settings.helics.generation_model_level == 'distribution'):
                total_bus_generation_p, total_bus_generation_q = self._get_bus_generation(load['bus'])
                logger.info(f"Generation is modeled in distribution level so transmission load is substituted the generation")
                self.psse.load_data_5(
                    load["bus"],
                    str(load["id"]),
                    realar=[total_distribution_load.real-total_bus_generation_p, total_distribution_load.imag-total_bus_generation_q, 0.0, 0.0, 0.0, 0.0],
                    # lodtyp="original",
                )
                logger.info(f"Original load {load['id']} @ bus {load['bus']}: {total_load}")
                logger.info(f"New load 'XX' @ bus {load['bus']} created successfully: {total_transmission_load}")
                logger.info(f"Load {load['id']} @ bus {load['bus']} updated : ({total_distribution_load.real-total_bus_generation_p},{total_distribution_load.imag-total_bus_generation_q})")
                load["distribution"] = complex(total_distribution_load.real-total_bus_generation_p, total_distribution_load.imag-total_bus_generation_q)
                load["transmission"] = total_transmission_load
            else:
                self.psse.load_data_5(
                    load["bus"],
                    str(load["id"]),
                    realar=[total_distribution_load.real, total_distribution_load.imag, 0.0, 0.0, 0.0, 0.0],
                    # lodtyp="original",
                )
                logger.info(f"Original load {load['id']} @ bus {load['bus']}: {total_load}")
                logger.info(f"New load 'XX' @ bus {load['bus']} created successfully: {total_transmission_load}")
                logger.info(f"Load {load['id']} @ bus {load['bus']} updated : {total_distribution_load}")
                load["distribution"] = total_distribution_load
                load["transmission"] = total_transmission_load
        logger.info(f"{loads}")
        return loads

    def _get_coupled_loads(self) -> list:
        """Returns a list of all coupled loads in a give simualtion

        Returns:
            list: list of coupled loads
        """
        
        # sub_data = pd.read_csv(
        #     os.path.join(
        #         self.settings["Simulation"]["Project Path"], "Settings", self.settings["HELICS"]["Subscriptions file"]
        #     ) # C:\Users\FXIE\Documents\GitHub\NEARM_TnD\dynamic_cosim_test_cases-model\WECC\Transmission\pypsse_model\Subscriptions.csv
        # )
        sub_data = pd.read_csv(self.settings.simulation.subscriptions_file)
        load = []
        logger.info(f"_get_coupled_loads")
        for _, row in sub_data.iterrows():
            if row["element_type"] == "Load":
                load.append(
                    {
                        "type": row["element_type"],
                        "id": row["element_id"],
                        "bus": row["bus"],
                    }
                )
        logger.info(f"load is {load}")
        return load

    def _get_load_static_data(self, loads: list) -> dict:
        """Returns static data for load models

        Args:
            loads (list): list of load names

        Returns:
            dict: mapping load to static values
        """
        logger.info(f"_get_load_static_data")
        values = ["MVA", "IL", "YL", "TOTAL"]
        for load in loads:
            for v in values:
                ierr, cmpval = self.psse.loddt2(load["bus"], str(load["id"]), v, "ACT")
                load[v] = cmpval
                logger.debug(f"Static properties - Load: {v} -> {cmpval}")
        return loads

    def _get_load_dynamic_data(self, loads: list) -> dict:
        """Returns dynamic data for load models

        Args:
            loads (list): list of load names

        Returns:
            dict: mapping load to dynamic values
        """
        logger.info(f"_get_load_dynamic_data")
        values = dyn_only_options["Loads"]["lmodind"]
        for load in loads:
            for v, con_ind in values.items():
                ierr = self.psse.inilod(load["bus"])
                assert ierr == 0, f"error={ierr}"
                ierr, ld_id = self.psse.nxtlod(load["bus"])
                assert ierr == 0, f"error={ierr}"
                if ld_id is not None:
                    ierr, con_index = self.psse.lmodind(load["bus"], ld_id, "CHARAC", "CON")
                    assert ierr == 0, f"error={ierr}"
                    if con_index is not None:
                        act_con_index = con_index + con_ind
                        ierr, value = self.psse.dsrval("CON", act_con_index)
                        assert ierr == 0, f"error={ierr}"
                        load[v] = value
                        logger.debug(f"Dynamic properties - Load: {v} -> index: {act_con_index}, value:{value}")
        return loads

    def setup_machine_channels(self, machines: dict, properties: list):
        """sets up machine channels

        Args:
            machines (dict): mapping machine to connected bus
            properties (list): list of machine properties
        """

        for _, qty in enumerate(properties):
            if qty not in self.channel_map:
                nqty = f"MACHINE_{qty}"
                self.channel_map[nqty] = {}
            for mch, b in machines:
                if qty in MACHINE_CHANNELS:
                    self.channel_map[nqty][f"{b}_{mch}"] = [self.chnl_idx]
                    chnl_id = MACHINE_CHANNELS[qty]
                    # logger.info(f"{qty} for machine {b}_{mch} added to channel {self.chnl_idx}")
                    self.psse.machine_array_channel([self.chnl_idx, chnl_id, int(b)], mch, "")
                    self.chnl_idx += 1

    def setup_load_channels(self, loads: list):
        """Sets up load channels

        Args:
            loads (list): list of loads
        """

        if "LOAD_P" not in self.channel_map:
            self.channel_map["LOAD_P"] = {}
            self.channel_map["LOAD_Q"] = {}

        for ld, b in loads:
            self.channel_map["LOAD_P"][f"{b}_{ld}"] = [self.chnl_idx]
            self.channel_map["LOAD_Q"][f"{b}_{ld}"] = [self.chnl_idx + 1]
            self.psse.load_array_channel([self.chnl_idx, 1, int(b)], ld, "")
            self.psse.load_array_channel([self.chnl_idx + 1, 2, int(b)], ld, "")
            logger.info(f"P and Q for load {b}_{ld} added to channel {self.chnl_idx} and {self.chnl_idx + 1}")
            self.chnl_idx += 2

    def setup_bus_channels(self, buses: list, properties: list):
        """Sets up bus channels

        Args:
            buses (list): list of buses
            properties (dict): list of bus properties
        """

        for _, qty in enumerate(properties):
            if qty not in self.channel_map:
                self.channel_map[qty] = {}
            for _, b in enumerate(buses):
                if qty == "frequency":
                    self.channel_map[qty][b] = [self.chnl_idx]
                    self.psse.bus_frequency_channel([self.chnl_idx, int(b)], "")
                    # logger.info(f"Frequency for bus {b} added to channel { self.chnl_idx}")
                    self.chnl_idx += 1
                elif qty == "voltage_and_angle":
                    self.channel_map[qty][b] = [self.chnl_idx, self.chnl_idx + 1]
                    self.psse.voltage_and_angle_channel([self.chnl_idx, -1, -1, int(b)], "")
                    # logger.info(f"Voltage and angle for bus {b} added to channel {self.chnl_idx} and {self.chnl_idx+1}")
                    self.chnl_idx += 2

    def poll_channels(self) -> dict:
        """Polls all channels adde during the setup process

        Returns:
            dict: mapping of polled channels to values
        """

        results = {}
        for ppty, b_dict in self.channel_map.items():
            ppty_new = ppty.split("_and_")
            for b, indices in b_dict.items():
                for n, idx in zip(ppty_new, indices):
                    if "_" not in n:
                        n_name = f"BUS_{n}"
                    else:
                        n_name = n
                    if n_name not in results:
                        results[n_name] = {}
                    ierr, value = self.psse.chnval(idx)
                    assert ierr == 0, f"error={ierr}"
                    if value is None:
                        value = -1
                    results[n_name][b] = value
        return results

    def setup_all_channels(self):
        """Sets up all user-defined channels for a project"""

        self.channel_map = {}
        self.chnl_idx = 1
        if not self.export_settings.channel_setup:
            return

        for channel in self.export_settings.channel_setup:
            method_type = channel.asset_type
            if method_type == "buses":
                self.setup_bus_channels(channel.asset_list, channel.asset_properties)
            elif method_type == "loads":
                load_list = [[x, int(y)] for x, y in channel.asset_list]
                self.setup_load_channels(load_list)
                # logger.debug(f"load_list: {load_list}")
                # os.system("PAUSE")
            elif method_type == "machines":
                machine_list = [[x, int(y)] for x, y in channel.asset_list]
                self.setup_machine_channels(machine_list, channel.asset_properties)
