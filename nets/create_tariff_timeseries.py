import os
import pandas as pd

from scenario_setup import tariff_scenarios
from nets.network_tariffs import run_model_multiple_households

res_dir = r"H:\Tariffs\Timeseries"
group_dir = r"H:\Tariffs\Consumer_Groups"
prices_dir = \
    r"C:\Users\aheider\Documents\Software\project_Ade\IndustrialDSMFinland\prices.csv"
check_existence = True

# import timeseries of components
base_components = {
    "Inflexible": ["loads_active_power"],
    "Heat_Pumps": ["heat_pumps", "heat_demand", "cop"],
    "Electric_Vehicles": ["upper_power", "lower_energy", "upper_energy"],
    "Photovoltaic": ["generators_active_power"],
    "Battery_Storage": ["storage_units"]
}

data = {}
for comp_dir, data_names in base_components.items():
    for data_name in data_names:
        data[data_name] = \
            pd.read_csv(os.path.join(group_dir, comp_dir, f"{data_name}.csv"),
                        index_col=0, parse_dates=True)
# adapt names to household loads
names = data["loads_active_power"].columns
for comp in ["heat_pumps", "storage_units"]:
    tmp = data[comp].iloc[:len(names)]
    tmp.index = names
    data[comp] = tmp
for ts in ["heat_demand", "cop", "upper_power", "lower_energy", "upper_energy",
           "generators_active_power"]:
    tmp = data[ts].iloc[:, :len(names)]
    tmp.columns = names
    data[ts] = tmp

# extract information relevant for optimisation
kwargs = {
    "ev_upper_power": data["upper_power"],
    "ev_lower_energy": data["lower_energy"],
    "ev_upper_energy": data["upper_energy"],
    "hp_p_nom_el": data["heat_pumps"]["p_set"],
    "hp_capacity_tes": data["heat_pumps"]["tes_capacity"],
    "hp_fixed_soc_tes": pd.Series(index=data["heat_pumps"].index, data=0.5),
    "gen_ts": data["generators_active_power"],
    "bess_capacity": data["storage_units"].capacity,
    "bess_ratio_power_to_energy":
        data["storage_units"].p_nom/data["storage_units"].capacity,
    "bess_fixed_soc": pd.Series(index=data["storage_units"].index, data=0.5)
}

# create consumer groups
consumer_groups = ["HH", "EV", "HP", "PV", "EV_HP", "EV_PV", "HP_PV", "PV_BESS",
                   "EV_HP_PV", "EV_HP_PV_BESS"]
# import scenarios for tariffs
scenarios = tariff_scenarios(
    timesteps=data["loads_active_power"].index,
    dir_time_varying_suppliers_costs=prices_dir,
)

for consumer_group in consumer_groups:
    has_ev = ("EV" in consumer_group)
    has_hp = ("HP" in consumer_group)
    has_pv = ("PV" in consumer_group)
    has_bess = ("BESS" in consumer_group)
    for tariff, tariff_data in scenarios.items():
        res_dir_tmp = os.path.join(res_dir, consumer_group, tariff)
        os.makedirs(res_dir_tmp, exist_ok=True)
        if check_existence:
            if os.path.isfile(os.path.join(res_dir, tariff, "hp_ts.csv")):
                print(f"{tariff} already solved. Skipping.")
                continue
        results_ts, results_scalar_tmp = run_model_multiple_households(
            load_ts=data["loads_active_power"],
            add_ev=has_ev,
            add_hp=has_hp,
            add_pv=has_pv,
            add_bess=has_bess,
            feedin_tariff=62.4,
            **kwargs, **tariff_data
        )
        results_scalar_tmp.to_csv(os.path.join(res_dir_tmp, f"scalars.csv"))
