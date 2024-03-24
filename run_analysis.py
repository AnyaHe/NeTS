import os
import pandas as pd
from edisgo.edisgo import import_edisgo_from_files
from edisgo.opf.timeseries_reduction import get_steps_reinforcement
import traceback

import logging
logger = logging.getLogger("pypsa")
logger.setLevel(logging.ERROR)

from copy import deepcopy

grid_id = 1056
feeder_id = 2
penetration = 0.1
grid_dir = r"H:\Tariffs\Grids\{}".format(grid_id)
mapping_dir = r"C:\Users\aheider\Documents\Grids\100_percent_dfo\{}".format(grid_id)
ev_dir = r"C:\Users\aheider\Documents\Grids\{}".format(grid_id)
prices_dir = \
    r"C:\Users\aheider\Documents\Software\project_Ade\IndustrialDSMFinland\prices.csv"
res_dir = r"H:\Tariffs\Results"
ts_dir = r"H:\Tariffs\Timeseries"
check_existence = False
calculate_reinforcement = True
tariffs = ['Ec', 'Cl', 'Ec_Cl', 'Ec_Clf', 'Ec_Clf_Sv', 'Ec_Cl_Sv', 'Ec_Sv',
           'Edn', 'Edn_Cl', 'Edn_Clf']


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
    # change convention of bess
    if der_type == "bess":
        ders_ts_new *= -1
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
        comps_ts_edisgo.update(ders_ts_new)
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


if __name__ == "__main__":
    # import grid
    edisgo_obj = import_edisgo_from_files(
        grid_dir,
        import_timeseries=True,
        import_heat_pump=True
    )
    households = edisgo_obj.topology.loads_df.loc[
        edisgo_obj.topology.loads_df.sector == "residential"
        ]
    # add ders and determine household groups
    households[["has_ev", "has_hp", "has_pv", "has_bess"]] = False
    for der in ["ev", "hp", "pv"]:
        hh_tmp = households.sample(frac=penetration)
        households.loc[hh_tmp.index, f"has_{der}"] = True
        if der == "pv":
            households.loc[hh_tmp.index, f"has_bess"] = True

    for tariff in tariffs:
        os.makedirs(os.path.join(res_dir, tariff), exist_ok=True)
        if check_existence:
            if os.path.isfile(os.path.join(res_dir, tariff, "scalars.csv")):
                print(f"{tariff} already solved. Skipping.")
                continue
        # update timeseries of household groups
        loads_ts = edisgo_obj.timeseries.loads_active_power.copy()
        for (has_ev, has_hp, has_pv, has_bess), household_group in households.groupby(
                ["has_ev", "has_hp", "has_pv", "has_bess"]):
            # determine group of time series
            customer_group = has_ev*"EV" + has_hp*"HP" + has_ev*"PV" + has_hp*"BESS"
            if customer_group == "":
                customer_group = "HH"
            # load time series
            ts_customer_group = pd.read_csv(
                os.path.join(ts_dir, customer_group, tariff, "timeseries.csv"),
                index_col=0, parse_dates=True)
            # sample number of group members
            replace = (len(household_group) > len(ts_customer_group.columns))
            ts_loads_customer_group = ts_customer_group.sample(
                    n=len(household_group), replace=replace, axis=1)
            # update load timeseries
            loads_ts.loc[:, household_group.index] = ts_loads_customer_group.values
        # write results back to edisgo_obj
        edisgo_obj.timeseries.loads_active_power = loads_ts
        # get reactive power
        edisgo_obj.set_time_series_reactive_power_control()
        edisgo_obj.check_integrity()
        edisgo_obj.save(os.path.join(res_dir, tariff))
        # determine grid issues and grid reinforcement costs
        if calculate_reinforcement:
            try:
                edisgo_tmp = deepcopy(edisgo_obj)
                edisgo_tmp.analyze()
                edisgo_tmp.results.to_csv(
                    f"{res_dir}/{tariff}/results_before_reinforcement",
                    parameters={"powerflow_results": ["i_res", "v_res"]})
                timesteps_reinforcement = get_steps_reinforcement(edisgo_tmp)
                edisgo_tmp.reinforce(timesteps_pfa=timesteps_reinforcement)
                edisgo_tmp.analyze()
                edisgo_tmp.results.to_csv(
                    f"{res_dir}/{tariff}/results_after_reinforcement",
                    parameters={"powerflow_results": ["i_res", "v_res"],
                                "grid_expansion_results": ["grid_expansion_costs",
                                                           "unresolved_issues"]})
                edisgo_tmp.topology.to_csv(
                    f"{res_dir}/{tariff}/topology_after_reinforcement")
            except:
                print(f"Something went wrong in grid reinforcement for {grid_id} {tariff}.")
                traceback.print_exc()
        print(f"Finished analysis for {tariff}.")
    print("Success")
