import os
import pandas as pd
from edisgo.edisgo import import_edisgo_from_files

from nets.network_tariffs import run_model
from scenario_setup import tariff_scenarios

grid_id = 1056
grid_dir = r"C:\Users\aheider\Documents\Grids\100_percent_dfo\{}".format(grid_id)
ev_dir = r"C:\Users\aheider\Documents\Grids\{}".format(grid_id)
prices_dir = \
    r"C:\Users\aheider\Documents\Software\project_Ade\IndustrialDSMFinland\prices.csv"
res_dir = "results/test"
check_existence = False

# import grid
edisgo_obj = import_edisgo_from_files(
    grid_dir,
    import_timeseries=True,
    import_heat_pump=True
)
# import scenarios for tariffs
scenarios = tariff_scenarios(
    timesteps=edisgo_obj.timeseries.timeindex,
    dir_time_varying_suppliers_costs=prices_dir,
)
# import EV data
cp_mapping = pd.read_csv(os.path.join(grid_dir, "cp_mapping.csv"),
                         index_col=0).set_index("name_new")
upper_powers = pd.read_csv(os.path.join(ev_dir, "upper_power_home.csv"), index_col=0,
                           parse_dates=True)
upper_energies = pd.read_csv(os.path.join(ev_dir, "upper_energy_home.csv"), index_col=0,
                             parse_dates=True)
lower_energies = pd.read_csv(os.path.join(ev_dir, "lower_energy_home.csv"), index_col=0,
                             parse_dates=True)
# resample to 1h resolution
upper_powers = upper_powers.resample("1h").mean().loc[edisgo_obj.timeseries.timeindex]
upper_energies = \
    upper_energies.resample("1h").max().loc[edisgo_obj.timeseries.timeindex]
lower_energies = \
    lower_energies.resample("1h").max().loc[edisgo_obj.timeseries.timeindex]
# extract households
households = edisgo_obj.topology.loads_df.loc[
    edisgo_obj.topology.loads_df.sector == "residential"
]
heat_pumps = edisgo_obj.topology.loads_df.loc[
    edisgo_obj.topology.loads_df.type == "heat_pump"
]
charging_points = edisgo_obj.topology.loads_df.loc[
    edisgo_obj.topology.loads_df.type == "charging_point"
]
pv_power_plants = edisgo_obj.topology.generators_df.loc[
    edisgo_obj.topology.generators_df.type == "solar"
]
households["has_hp"] = households.bus.isin(heat_pumps.bus)
households["has_ev"] = households.bus.isin(charging_points.bus)
households["has_pv"] = households.bus.isin(pv_power_plants.bus)
households["has_bess"] = households.bus.isin(edisgo_obj.topology.storage_units_df.bus)
examples = households[["has_ev", "has_hp", "has_pv", "has_bess"]].drop_duplicates(
    keep="first").index
for tariff, tariff_data in scenarios.items():
    os.makedirs(os.path.join(res_dir, tariff), exist_ok=True)
    if check_existence:
        if len(os.listdir(os.path.join(res_dir, tariff))) >= 8:
            print(f"{tariff} already solved. Skipping.")
            continue
    for name, data in households.loc[examples].iterrows():
        kwargs = {}
        connected_elements = edisgo_obj.topology.get_connected_components_from_bus(data.bus)
        has_ev, has_hp, has_pv, has_bess = False, False, False, False
        load_ts = pd.Series()
        for comp_type in ["loads", "generators", "storage_units"]:
            comps = connected_elements[comp_type]
            if comp_type == "loads":
                # get load time series
                residential_load = comps.loc[comps.sector == "residential"]
                load_ts = edisgo_obj.timeseries.loads_active_power[
                    residential_load.index].sum(axis=1)
                # handle heat pumps if present
                heat_pumps = comps.loc[comps.type == "heat_pump"]
                has_hp = (len(heat_pumps) > 0)
                if has_hp:
                    kwargs["hp_p_nom_el"] = heat_pumps.p_set.sum()
                    kwargs["hp_capacity_tes"] = \
                        edisgo_obj.heat_pump.thermal_storage_units_df.loc[
                            heat_pumps.index, "capacity"].sum()
                    kwargs["hp_heat_demand"] = \
                        edisgo_obj.heat_pump.heat_demand_df[heat_pumps.index].sum(axis=1)
                    kwargs["hp_cop"] = \
                        edisgo_obj.heat_pump.cop_df[heat_pumps.index].sum(axis=1)
                # handle electric vehicles if present
                charging_points = comps.loc[comps.type == "charging_point"]
                has_ev = (len(charging_points) > 0)
                if has_ev:
                    names_cps = cp_mapping.loc[charging_points.index, "name_orig"]
                    kwargs["ev_upper_power"] = upper_powers[names_cps].sum(axis=1)
                    kwargs["ev_upper_energy"] = upper_energies[names_cps].sum(axis=1)
                    kwargs["ev_lower_energy"] = lower_energies[names_cps].sum(axis=1)
            if comp_type == "generators":
                has_pv = (len(comps) > 0)
                if has_pv:
                    if not (comps.type == "solar").all():
                        raise NotImplementedError
                    kwargs["gen_ts"] = edisgo_obj.timeseries.generators_active_power[
                        comps.index].sum(axis=1)
            if comp_type == "storage_units":
                has_bess = (len(comps) > 0)
                if has_bess:
                    kwargs["bess_capacity"] = comps.capacity.sum()
                    kwargs["bess_ratio_power_to_energy"] = \
                        comps.p_nom.sum()/comps.capacity.sum()
        results_ts, results_scalar = run_model(
            load_ts=load_ts,
            add_ev=has_ev,
            add_hp=has_hp,
            add_pv=has_pv,
            add_bess=has_bess,
            feedin_tariff=62.4,
            **kwargs, **tariff_data
        )
        name_res = has_ev*"EV_" + has_hp*"HP_" + has_pv*"PV_" + has_bess*"BESS_"
        results_ts.to_csv(os.path.join(res_dir, tariff, f"ts_{name_res}.csv"))
        results_scalar.to_csv(os.path.join(res_dir, tariff, f"scalars_{name_res}.csv"))
print("Success")
