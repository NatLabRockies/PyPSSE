from pathlib import Path
import os
from loguru import logger
import pandas as pd
import click
import toml

from pypsse.common import EXPORTS_SETTINGS_FILENAME, SIMULATION_SETTINGS_FILENAME
from pypsse.models import ExportFileOptions, SimulationSettings
from pypsse.simulator import Simulator

@click.argument(
    "project-path",
)
@click.option(
    "-s",
    "--simulations-file",
    required=False,
    default=SIMULATION_SETTINGS_FILENAME,
    show_default=True,
    help="scenario toml file to run (over rides default)",
)
@click.option(
    "-e",
    "--export-file-path",
    required=False,
    default="./filtered_results.csv",
    show_default=True,
    help="path for exporting filtered results",
)
@click.option(
    "-l",
    "--load-filter",
    help="applies load filters if set to true",
    is_flag=True,
    default=False,
    show_default=True,
)
@click.option(
    '--load/--no-load',
    required=False,
    show_default=True,
    help="filter by load [bool]",
)
@click.option(
    "-g",
    "--gen-filter",
    help="applies generator filters if set to true",
    is_flag=True,
    default=False,
    show_default=True,
)
@click.option(
    '--generation/--no-generation',
    required=False,
    show_default=True,
    help="filter by generation [bool]",
)
@click.option(
    '--comp-load/--no-comp-load',
    required=False,
    show_default=True,
    help="filter by composite load models [bool]",
)
@click.option(
    "-b",
    "--apply-bounds",
    help="applies load and generation limit bounds if set to true",
    is_flag=True,
    default=False,
    show_default=True,
)
@click.option(
    '--load-bounds',
    required=False,
    show_default=True,
    default= "0/10000",
    help="bounds for load [example 10/100]",
)
@click.option(
    '--gen-bounds',
    required=False,
    show_default=True,
    default= "0/10000",
    help="bounds for generation ",
)
@click.command()
def explore(project_path, simulations_file, export_file_path, load_filter, load, gen_filter, generation, apply_bounds, comp_load, load_bounds, gen_bounds):
    """Runs a valid PyPSSE simulation."""
    
    export_file_path = Path(export_file_path)
    
    file_path = Path(project_path) / simulations_file
    msg = "Simulation file not found. Use -s to choose a valid settings file"
    "if its name differs from the default file name."
    assert file_path.exists(), msg
    
    simulation_settings = toml.load(file_path)
    simulation_settings = SimulationSettings(**simulation_settings)
    simulation_settings.helics.cosimulation_mode = False
    x = Simulator(simulation_settings)
    buses = set(x.raw_data.buses)
    quantities =  {
        'Loads': ['MVA', "IL", "YL", 'FmA', 'FmB', 'FmC', 'FmD', 'Fel', 'PFel'], 
        'Induction_generators': ['MVA'], 
        'Machines': ['MVA', 'PERCENT'], 
        }
    results = x.sim.read_subsystems(quantities,  buses)

    # print(results.keys())
    # print(results["LOAD_P"])
    # quit()

    had_comp_models = False
    if "Loads_FmA" in results:
        had_comp_models = True
    
    load_dict = {}
    bus_load_real = {}
    bus_load_imag = {}
    is_comp_load = {}
    for bus, ld_id in x.raw_data.loads:
        if bus not in load_dict:
            load_dict[bus] = []
            bus_load_real[bus] = 0
            bus_load_imag[bus] = 0
        
        load_bus_id = f'{bus}_{ld_id}'
        if had_comp_models:
            is_comp = True if f'{bus}_{ld_id}' in results["Loads_FmA"] else False
        else:
            is_comp = False
            
        is_comp_load[bus] = is_comp
        load_dict[bus].append(ld_id)
        key = f"{bus}_{ld_id}"
        key2 =  f"{bus}_{ld_id}".replace(" ", "") 
        load_p  = max(
            results["Loads_MVA"][key].real + results["Loads_IL"][key].real + results["Loads_YL"][key].real, 
            results["LOAD_P"][key2] *100 if key2 in results["LOAD_P"] else 0
            )
        load_q  = max(
            results["Loads_MVA"][key].imag + results["Loads_IL"][key].imag + results["Loads_YL"][key].imag, 
            results["LOAD_Q"][key2] *100 if key2 in results["LOAD_Q"] else 0
            )
        
        bus_load_real[bus] += load_p
        bus_load_imag[bus] += load_q
        
    generator_dict = {}
    bus_gen = {}
    for bus, gen_id in x.raw_data.generators:
        if bus not in generator_dict:
            generator_dict[bus] = []
            bus_gen[bus] = 0
        key = f"{bus}_{gen_id}"
        generator_dict[bus].append(gen_id)
        bus_gen[bus] += results["Machines_MVA"][key]

    results = {
        "bus" : [],
        "has load" : [],
        "is load comp" : [],
        "total P load [MW]" : [],
        "total Q load [MVar]" : [],
        "has generation" : [],
        "total generation [MVA]" : [],
    }
    for bus in x.raw_data.buses:
        results["bus"].append(bus)
        results["has load"].append(True if bus in load_dict else False)
        results["is load comp"].append(is_comp_load[bus] if bus in is_comp_load else False)
        results["total P load [MW]"].append(bus_load_real[bus] if bus in bus_load_real else 0)
        results["total Q load [MVar]"].append(bus_load_imag[bus] if bus in bus_load_imag else 0)
        results["has generation"].append(True if (bus in generator_dict and bus_gen[bus] > 0)else False)
        results["total generation [MVA]"].append(bus_gen[bus] if bus in bus_gen else 0)
    
    
    results = pd.DataFrame(results)
    
    results["total P load [MW]"] = results["total P load [MW]"] 
    results["total Q load [MVar]"] = results["total Q load [MVar]"]
    
    if load_filter:
        if load:
            results=results[results["has load"] == True]
        elif not load:
            results=results[results["has load"] == False]
        
        if comp_load:
            results=results[results["is load comp"] == True]
        elif not comp_load:
            results=results[results["is load comp"] == False]
        
    if gen_filter:
        if generation:
            results=results[results["has generation"] == True]
        elif not generation:
            results=results[results["has generation"] == False]    
    
    load_lower, load_upper = [float(x) for x in load_bounds.split("/")] 
    gen_lower, gen_upper = [float(x) for x in gen_bounds.split("/")] 
    if apply_bounds:
        results=results[(results["total P load [MW]"] >= load_lower) & (results["total P load [MW]"] <= load_upper)]
        results=results[(results["total generation [MVA]"] >= gen_lower) & (results["total generation [MVA]"] <= gen_upper)]

    # print(results)    
    results.to_csv(export_file_path)
    logger.info(f"Results exported to {export_file_path.absolute()}")
    

        
    
    