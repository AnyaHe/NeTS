import os
import pandas as pd
from edisgo.edisgo import import_edisgo_from_files

from nets.network_tariffs import run_model_multiple_households
from scenario_setup import tariff_scenarios

grid_id = 1056
feeder_id = 2
grid_dir = r"C:\Users\aheider\Documents\Grids\100_percent_dfo\{}\feeder\{}".format(
    grid_id, feeder_id)
mapping_dir = r"C:\Users\aheider\Documents\Grids\100_percent_dfo\{}".format(grid_id)
ev_dir = r"C:\Users\aheider\Documents\Grids\{}".format(grid_id)
prices_dir = \
    r"C:\Users\aheider\Documents\Software\project_Ade\IndustrialDSMFinland\prices.csv"
res_dir = "results/test_update"
check_existence = True

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
cp_mapping = pd.read_csv(os.path.join(mapping_dir, "cp_mapping.csv"),
                         index_col=0).set_index("name_new")
upper_power = pd.read_csv(os.path.join(ev_dir, "upper_power_home.csv"), index_col=0,
                          parse_dates=True)
upper_energy = pd.read_csv(os.path.join(ev_dir, "upper_energy_home.csv"), index_col=0,
                           parse_dates=True)
lower_energy = pd.read_csv(os.path.join(ev_dir, "lower_energy_home.csv"), index_col=0,
                           parse_dates=True)
reference_charging = pd.read_csv(os.path.join(ev_dir, "dumb", "timeseries",
                                              "charging_points_active_power.csv"),
                                 index_col=0, parse_dates=True)
# resample to 1h resolution
upper_power = upper_power.resample("1h").mean().loc[edisgo_obj.timeseries.timeindex]
upper_energy = \
    upper_energy.resample("1h").max().loc[edisgo_obj.timeseries.timeindex]
lower_energy = \
    lower_energy.resample("1h").max().loc[edisgo_obj.timeseries.timeindex]
reference_charging = \
    reference_charging.resample("1h").mean().loc[edisgo_obj.timeseries.timeindex]
# extract households
households = edisgo_obj.topology.loads_df.loc[
    edisgo_obj.topology.loads_df.sector == "residential"
]
# handle heat pumps
hp = edisgo_obj.topology.loads_df.loc[
    (edisgo_obj.topology.loads_df.type == "heat_pump") &
    (edisgo_obj.topology.loads_df.bus.isin(households.bus))
]
hp["tes_capacity"] = \
    edisgo_obj.heat_pump.thermal_storage_units_df.loc[hp.index, "capacity"]
hp_grouped = hp.groupby("bus").agg({
    "p_set": sum, "annual_consumption": sum, "number_households": sum,
    "tes_capacity": sum,
    "building_id": "first", "type": "first", "sector": "first",
    "voltage_level": "first"})
hp_ts = {}
for ts in ["heat_demand", "cop"]:
    ts_tmp = getattr(edisgo_obj.heat_pump, f"{ts}_df")[hp.index]
    ts_tmp.columns = hp.bus
    hp_ts[ts] = ts_tmp.groupby(ts_tmp.columns, axis=1).sum()
# handle charging points
ev = edisgo_obj.topology.loads_df.loc[
    (edisgo_obj.topology.loads_df.type == "charging_point") &
    (edisgo_obj.topology.loads_df.bus.isin(households.bus))
]
evs_grouped = ev.groupby("bus").agg({
    "p_set": sum, "type": "first", "sector": "first"
})
names_cps = cp_mapping.loc[ev.index, "name_orig"]
cp_ts = {}
for ts in ["upper_power", "upper_energy", "lower_energy"]:
    ts_tmp = locals()[ts][names_cps]
    ts_tmp.columns = ev.bus
    cp_ts[ts] = ts_tmp.groupby(ts_tmp.columns, axis=1).sum()
# handle pv plants
pv = edisgo_obj.topology.generators_df.loc[
    (edisgo_obj.topology.generators_df.type == "solar") &
    (edisgo_obj.topology.generators_df.bus.isin(households.bus))
]
pv_grouped = pv.groupby("bus").agg({
    "p_nom": sum, "id": "mean", "type": "first", "control": "first",
    "weather_cell_id": "first", "subtype": "first", "voltage_level": "first"
})
ts_gen_group = edisgo_obj.timeseries.generators_active_power[pv.index]
ts_gen_group.columns = pv.bus
ts_gen_group = ts_gen_group.groupby(ts_gen_group.columns, axis=1).sum()
# handle battery systems
bess = edisgo_obj.topology.storage_units_df.loc[
    edisgo_obj.topology.storage_units_df.bus.isin(households.bus)]
bess_grouped = bess.groupby("bus").agg({
    "capacity": sum, "p_nom": sum, "control": "first"
})
# build household groups with same der constellation
households["has_hp"] = households.bus.isin(hp.bus)
households["has_ev"] = households.bus.isin(ev.bus)
households["has_pv"] = households.bus.isin(pv.bus)
households["has_bess"] = households.bus.isin(edisgo_obj.topology.storage_units_df.bus)

# add time series for non-residential cps
non_res_ev = edisgo_obj.topology.loads_df.loc[
    (edisgo_obj.topology.loads_df.type == "charging_point") &
    ~(edisgo_obj.topology.loads_df.bus.isin(households.bus))
    ]
ts_non_res_ev = reference_charging[cp_mapping.loc[non_res_ev.index, "name_orig"]]
ts_non_res_ev.columns = non_res_ev.index
edisgo_obj.timeseries.loads_active_power = \
    pd.concat([edisgo_obj.timeseries.loads_active_power, ts_non_res_ev], axis=1)


def adapt_der_to_households(der_group, household_grp, ders_new):
    der_group_tmp = der_group.loc[household_grp.bus]
    der_group_tmp = der_group_tmp.reset_index()
    der_group_tmp.index = household_grp.index
    ders_new = pd.concat([ders_new, der_group_tmp])
    return der_group_tmp, ders_new


def update_edisgo_dfs(der_type, ders_new, ders_ts_new, ders_orig, comps_edisgo,
                      comps_ts_edisgo):
    # rename components
    names_new_ders = [f"{der_type.upper()}_" + idx for idx in ders_new.index]
    ders_new.index = names_new_ders
    ders_ts_new.columns = names_new_ders
    # if first time object is updated, drop existing elements,
    # otherwise just update frames
    if ders_orig.index.isin(comps_edisgo.index).any():
        # drop original components
        comps_edisgo.drop(ders_orig.index, inplace=True)
        if ders_orig.index.isin(comps_ts_edisgo.columns).any():
            comps_ts_edisgo.drop(ders_orig.index, axis=1, inplace=True)
        # add new components
        comps_edisgo = pd.concat([comps_edisgo, ders_new])
        comps_ts_edisgo = pd.concat([comps_ts_edisgo, ders_ts_new], axis=1)
    else:
        comps_edisgo.update(ders_new)
        comps_ts_edisgo.update(comps_ts_edisgo)
    return comps_edisgo, comps_ts_edisgo


def update_edisgo_obj(edisgo_obj):
    def _update_edisgo_obj():
        comps = getattr(edisgo_obj.topology, f"{comp_type}_df")
        comps_ts_tmp = getattr(edisgo_obj.timeseries, f"{comp_type}_active_power")
        for der in ders:
            comps, comps_ts_tmp = \
                update_edisgo_dfs(der_type=der,
                                  ders_new=globals()[f"{der}_new"],
                                  ders_ts_new=globals()[f"{der}_ts_new"],
                                  ders_orig=globals()[f"{der}"],
                                  comps_edisgo=comps,
                                  comps_ts_edisgo=comps_ts_tmp)
        setattr(edisgo_obj.topology, f"{comp_type}_df", comps)
        setattr(edisgo_obj.timeseries, f"{comp_type}_active_power", comps_ts_tmp)

    components = {
        "loads": ["hp", "ev"],
        "generators": ["pv"],
        "storage_units": ["bess"]
    }
    for comp_type, ders in components.items():
        _update_edisgo_obj()
    return edisgo_obj


for tariff, tariff_data in scenarios.items():
    os.makedirs(os.path.join(res_dir, tariff), exist_ok=True)
    if check_existence:
        if len(os.listdir(os.path.join(res_dir, tariff))) >= 6:
            print(f"{tariff} already solved. Skipping.")
            continue
    # initialise objects to update edisgo_obj
    hp_new = pd.DataFrame()
    hp_ts_new = pd.DataFrame()
    ev_new = pd.DataFrame()
    ev_ts_new = pd.DataFrame()
    pv_new = pd.DataFrame()
    pv_ts_new = pd.DataFrame()
    bess_new = pd.DataFrame()
    bess_ts_new = pd.DataFrame()
    results_scalar = pd.DataFrame()
    # get all households with the same combination of ders
    for (has_ev, has_hp, has_pv, has_bess), household_group in households.groupby(
            ["has_ev", "has_hp", "has_pv", "has_bess"]):
        kwargs = {}
        loads_ts = edisgo_obj.timeseries.loads_active_power[
            household_group.index]
        if has_hp:
            hp_group_tmp, hp_new = \
                adapt_der_to_households(hp_grouped, household_group, hp_new)
            kwargs["hp_p_nom_el"] = hp_group_tmp["p_set"]
            kwargs["hp_capacity_tes"] = hp_group_tmp["tes_capacity"]
            kwargs["hp_fixed_soc_tes"] = pd.Series(index=hp_group_tmp.index, data=0.5)
            for ts in ["heat_demand", "cop"]:
                ts_tmp = hp_ts[ts][household_group.bus]
                ts_tmp.columns = household_group.index
                kwargs[f"hp_{ts}"] = ts_tmp
        if has_ev:
            _, ev_new = adapt_der_to_households(evs_grouped, household_group, ev_new)
            for ts in ["upper_power", "upper_energy", "lower_energy"]:
                ts_tmp = cp_ts[ts][household_group.bus]
                ts_tmp.columns = household_group.index
                kwargs[f"ev_{ts}"] = ts_tmp
        if has_pv:
            _, pv_new = adapt_der_to_households(pv_grouped, household_group, pv_new)
            ts_tmp = ts_gen_group[household_group.bus]
            ts_tmp.columns = household_group.index
            kwargs["gen_ts"] = ts_tmp
        if has_bess:
            bess_group_tmp, bess_new = \
                adapt_der_to_households(bess_grouped, household_group, bess_new)
            kwargs["bess_capacity"] = bess_group_tmp.capacity
            kwargs["bess_ratio_power_to_energy"] = \
                bess_group_tmp.p_nom / bess_group_tmp.capacity
            kwargs["bess_fixed_soc"] = pd.Series(index=bess_group_tmp.index, data=0.5)

        results_ts, results_scalar_tmp = run_model_multiple_households(
            load_ts=loads_ts,
            add_ev=has_ev,
            add_hp=has_hp,
            add_pv=has_pv,
            add_bess=has_bess,
            feedin_tariff=62.4,
            **kwargs, **tariff_data
        )
        # save ts and new components
        if has_hp:
            hp_ts_new = pd.concat([hp_ts_new, results_ts["hp"]], axis=1)
        if has_ev:
            ev_ts_new = pd.concat([ev_ts_new, results_ts["ev"]], axis=1)
        if has_pv:
            pv_ts_new = pd.concat([pv_ts_new, results_ts["pv"]], axis=1)
        if has_bess:
            bess_ts_new = pd.concat([bess_ts_new, results_ts["bess"]], axis=1)
        results_scalar = pd.concat([results_scalar, results_scalar_tmp])
    # write results back to edisgo_obj
    edisgo_obj = update_edisgo_obj(edisgo_obj)
    # get reactive power
    edisgo_obj.set_time_series_reactive_power_control()
    edisgo_obj.check_integrity()
    edisgo_obj.save(os.path.join(res_dir, tariff))
    results_scalar.to_csv(os.path.join(res_dir, tariff, f"scalars.csv"))
print("Success")
