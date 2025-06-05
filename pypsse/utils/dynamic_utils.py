import os
from typing import ClassVar, List

import pandas as pd
from loguru import logger

from pypsse.common import MACHINE_CHANNELS
from pypsse.modes.constants import dyn_only_options


class DynamicUtils:
    "Utility functions for dynamic simulations"

    dynamic_params: ClassVar[List[str]] = ["FmA", "FmB", "FmC", "FmD", "Fel"]

    def disable_generation_for_coupled_buses(self):
        """Disables generation of coupled buses (co-simulation mode only)"""
        if (
            self.settings.helics
            and self.settings.helics.cosimulation_mode
            and self.settings.helics.disable_generation_on_coupled_buses
        ):
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
            # os.system("PAUSE")

    def disable_load_models_for_coupled_buses(self):
        """Disables loads of coupled buses (co-simulation mode only)"""
        if self.settings.helics and self.settings.helics.cosimulation_mode:
            sub_data = pd.read_csv(self.settings.simulation.subscriptions_file)
            sub_data = sub_data[sub_data["element_type"] == "Load"]
            logger.info(f"break the load")
            self.break_loads(None, ['FmD'])
            # self.psse_dict = {}
            # for _, row in sub_data.iterrows():
            #     bus = row["bus"]
            #     load = row["element_id"]
            #     ierr = self.psse.ldmod_status(0, int(bus), str(load), 1, 0)
            #     if ierr == 0:
            #         logger.info(f"Dynamic model for load {load} connected to bus {bus} has been disabled")
            #     elif ierr == 5:
            #         logger.error(f"No dynamic model found for load {load} connected to bus {bus}")
            #     else:
            #         raise Exception(f"error={ierr}") 
                # os.system("PAUSE")
                
    def break_loads(self, loads: list = None, components_to_replace: List[str] = []):
        """Implements the load split logic

        Args:
            loads (list, optional): list of coupled loads. Defaults to None.
            components_to_replace (List[str], optional): components to be simulated on distribution side. Defaults to [].
        """
        logger.info("##################### break_loads #########################")
        os.system("PAUSE")
        components_to_stay = [x for x in self.dynamic_params if x not in components_to_replace]
        logger.info(f"components_to_stay: {components_to_stay}")
        logger.info(f"components_to_replace: {components_to_replace}")
        os.system("PAUSE")
        if loads is None:
            loads = self._get_coupled_loads()
        os.system("PAUSE")
        loads = self._get_load_static_data(loads)
        os.system("PAUSE")
        loads = self._get_load_dynamic_data(loads)
        os.system("PAUSE")
        loads = self._replicate_coupled_load(loads, components_to_replace)
        os.system("PAUSE")
        self._update_dynamic_parameters(loads, components_to_stay, components_to_replace)
        os.system("PAUSE")

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
                new_percentages[comp] = load[comp] / count
            for comp in components_to_replace:
                new_percentages[comp] = 0.0

            settings = self._get_load_dynamic_properties(load)
            #
            for k, v in new_percentages.items():
                idx = dyn_only_options["Loads"]["lmodind"][k]
                settings[idx] = v
                # self.psse.change_ldmod_con(load['bus'], 'XX' ,r"""CMLDBLU2""" ,idx ,v)
            values = list(settings.values())
            self.psse.add_load_model(load["bus"], "XX", 0, 1, r"""CMLDBLU2""", 2, [0, 0], ["", ""], 133, values)
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

    def _replicate_coupled_load(self, loads: dict, components_to_replace: list):
        """create a replica of composite load model

        Args:
            loads (dict): load dictionary
            components_to_replace (list): composite load models to replace on distribution side

        Returns:
            dict: updated load dictionary
        """
        logger.info(f"2222222222222222222222222")
        for load in loads:
            dynamic_percentage = load["FmA"] + load["FmB"] + load["FmC"] + load["FmD"] + load["Fel"]
            static_percentage = 1.0 - dynamic_percentage
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
            # ierr, cmpval = self.psse.loddt2(load["bus"], "XX" ,"MVA" , "ACT")
            # modify old load
            self.psse.load_data_5(
                load["bus"],
                str(load["id"]),
                realar=[total_distribution_load.real, total_distribution_load.imag, 0.0, 0.0, 0.0, 0.0],
                # lodtyp="original",
            )
            # ierr, cmpval = self.psse.loddt2(load["bus"], load["id"] ,"MVA" , "ACT")
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
        logger.info(f"{loads}")
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
        logger.info(f"{loads}")
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
            elif method_type == "machines":
                machine_list = [[x, int(y)] for x, y in channel.asset_list]
                self.setup_machine_channels(machine_list, channel.asset_properties)
