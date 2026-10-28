from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from pypsse.common import SIMULATION_SETTINGS_FILENAME
from pypsse.enumerations import SimulationModes
from pypsse.modes.constants import dyn_only_options
from pypsse.simulator import Simulator
from pypsse.utils.dynamic_utils import DynamicUtils
from pypsse.utils.utils import load_settings

from utils import build_temp_project, remove_temp_project

def run_sim_dynamic():
    project_path = build_temp_project()
    file_Path = project_path / SIMULATION_SETTINGS_FILENAME

    if file_Path.exists():
        settings = load_settings(file_Path, project_path)
        settings.simulation.simulation_mode = SimulationModes.DYNAMIC
        settings.simulation.use_profile_manager = False
        settings.helics.cosimulation_mode = False
        x = Simulator(settings)
        yield x
        x.init()
        x.run()
        del x
        remove_temp_project(project_path)
    else:
        msg = f"'{file_Path}' is not a valid path."
        raise Exception(msg)

def test_run_sim_dynamic_save_model():
    simulators = run_sim_dynamic()
    simulator = next(simulators)
    files = simulator.sim.save_model()
    for file in files:
        assert file.exists(), f"{file}"

def test_disable_generation():
    simulators = run_sim_dynamic()
    simulator = next(simulators)
    


def test_disbale_load_model():
    simulators = run_sim_dynamic()
    simulator = next(simulators)


def test_channel_setup():
    simulators = run_sim_dynamic()
    simulator = next(simulators)


def test_load_break():
    simulators = run_sim_dynamic()
    simulator = next(simulators)


@pytest.mark.parametrize("has_abc", [None, False, True])
def test_load_split_selects_distribution_motor_components(tmp_path, has_abc):
    row = {"bus": 244, "element_id": "1", "element_type": "Load"}
    if has_abc is not None:
        row["replace_motor_abc"] = has_abc
    path = tmp_path / "Subscriptions.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    utils = DynamicUtils()
    utils.settings = SimpleNamespace(
        helics=SimpleNamespace(cosimulation_mode=True),
        simulation=SimpleNamespace(subscriptions_file=path),
    )
    utils.psse = SimpleNamespace(ldmod_status=Mock(return_value=0))
    utils.break_loads_for_dynamic_cosimulations = Mock()
    utils.disable_load_models_for_coupled_buses()
    expected = ["FmA", "FmB", "FmC", "FmD"] if has_abc else ["FmD"]
    utils.break_loads_for_dynamic_cosimulations.assert_called_once_with(
        loads=[{"type": "Load", "id": "1", "bus": 244}], components_to_replace=expected,
    )


@pytest.mark.parametrize("replace", [["FmD"], ["FmA", "FmB", "FmC", "FmD"]])
@pytest.mark.parametrize("generation_on_dx", [False, True])
def test_load_split_conserves_power_and_updates_motor_fractions(replace, generation_on_dx):
    utils = DynamicUtils()
    utils.settings = SimpleNamespace(
        helics=SimpleNamespace(cosimulation_mode=True, generation_model_level="distribution"),
        simulation=SimpleNamespace(disable_generation_on_coupled_buses=generation_on_dx),
    )
    utils.psse = SimpleNamespace(load_data_5=Mock(), add_load_model=Mock())
    utils._get_bus_generation = Mock(return_value=(13, 0))
    utils._get_load_dynamic_properties = Mock(return_value=dict.fromkeys(range(133), 0.0))
    load = {
        "bus": 244, "id": "1", "TOTAL": complex(31, 7),
        "FmA": 0.05, "FmB": 0.05, "FmC": 0.05, "FmD": 0.3, "Fel": 0.0,
    }
    loads = utils._replicate_coupled_load([load], replace)
    generation = 13 if generation_on_dx else 0
    assert load["distribution"] + load["transmission"] + generation == pytest.approx(complex(31, 7))
    assert load["transmission"].real == pytest.approx(0 if len(replace) == 4 else 4.65)
    stay = [name for name in utils.dynamic_params if name not in replace]
    utils._update_dynamic_parameters(loads, stay, replace)
    values = utils.psse.add_load_model.call_args.args[-1]
    for name in replace:
        assert values[dyn_only_options["Loads"]["lmodind"][name]] == 0


@pytest.mark.parametrize("placements", [[True, True], [True, False], [None, None], ["bad", "bad"]])
def test_load_split_validates_duplicate_subscriptions(tmp_path, placements):
    path = tmp_path / "Subscriptions.csv"
    pd.DataFrame([
        {"bus": 244, "element_id": "1", "element_type": "Load", "replace_motor_abc": value}
        for value in placements
    ]).to_csv(path, index=False)
    utils = DynamicUtils()
    utils.settings = SimpleNamespace(
        helics=SimpleNamespace(cosimulation_mode=True),
        simulation=SimpleNamespace(subscriptions_file=path),
    )
    utils.psse = SimpleNamespace(ldmod_status=Mock(return_value=0))
    utils.break_loads_for_dynamic_cosimulations = Mock()
    if placements == [True, True]:
        utils.disable_load_models_for_coupled_buses()
        utils.break_loads_for_dynamic_cosimulations.assert_called_once()
        utils.psse.ldmod_status.assert_called_once()
    else:
        with pytest.raises(ValueError, match="replace_motor_abc"):
            utils.disable_load_models_for_coupled_buses()
        utils.break_loads_for_dynamic_cosimulations.assert_not_called()
