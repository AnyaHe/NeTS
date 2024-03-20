from edisgo.edisgo import import_edisgo_from_files
import os
import pandas as pd

# Script to prepare results for EFf-NeTs
grid_id = 1056
res_dir = r"H:\Tariffs\{}".format(grid_id)
res_dir = r"C:\Users\aheider\Documents\Software\Tariffs\NeTS\results\test_update"
ref_dir = r"H:\Grids\1056\feeder\2"
tariffs = [file for file in os.listdir(res_dir) if not file.endswith(".csv")]


def get_p_percent_highest_peaks_and_customer_contribution(
        data, data_agg, edisgo_object, p=0.1):
    components = {
        "loads": ["hp", "ev"],
        "generators": ["pv"],
        "storage_units": ["bess"]
    }
    res_load = edisgo_object.timeseries.residual_load
    times_with_highest_absolute_peaks = \
        res_load.abs().sort_values(ascending=False).iloc[:int(p*len(res_load))].index
    data_agg["Aggregated Peak"] = \
        res_load[times_with_highest_absolute_peaks].abs().sum()
    names_loads = data["Consumer"]
    timeseries_loads = edisgo_object.timeseries.loads_active_power[names_loads]
    # get contribution of individual consumers
    for comp_type, ders in components.items():
        for der in ders:
            timeseries_tmp = pd.DataFrame(index=timeseries_loads.index,
                                          columns=timeseries_loads.columns)
            names_loads_comps = [
                name for name in names_loads if f"{der.upper()}_{name}" in
                                                getattr(edisgo_object.topology,
                                                        f"{comp_type}_df").index]
            names_comps = \
                [f"{der.upper()}_{name}" for name in names_loads_comps]
            timeseries_tmp[names_loads_comps] = \
                getattr(edisgo_object.timeseries,
                        f"{comp_type}_active_power")[names_comps].values
            if der in ["hp", "ev"]:
                timeseries_loads += timeseries_tmp.fillna(0)
            elif der in ["pv", "bess"]:
                timeseries_loads -= timeseries_tmp.fillna(0)
            else:
                raise ValueError("Unknown type of der.")
    data["Aggregated Peak"] = \
        timeseries_loads.loc[times_with_highest_absolute_peaks].T.divide(
            res_load[times_with_highest_absolute_peaks]).sum(axis=1).values
    return data, data_agg


if __name__ == "__main__":
    data_en = pd.DataFrame()
    data_en_agg = pd.DataFrame()
    # import reference scenario
    scenario = "reference"
    data_en_tmp = pd.DataFrame()
    data_en_agg_tmp = pd.Series()
    edisgo_obj = import_edisgo_from_files(os.path.join(ref_dir),
                                          import_timeseries=True)
    # get cost share of vulnerable consumers
    households = edisgo_obj.topology.loads_df.loc[
        edisgo_obj.topology.loads_df.sector == "residential"
        ]
    ac_mean = households.annual_consumption.mean()
    data_en_agg_tmp["Cost Share VU"] = households.loc[
        households.annual_consumption < ac_mean].annual_consumption.mean()/(
        households.loc[
            households.annual_consumption >= ac_mean].annual_consumption.mean()
    )
    # get aggregated peak
    p = 0.1
    res_load = edisgo_obj.timeseries.residual_load
    times_with_highest_absolute_peaks = \
        res_load.abs().sort_values(ascending=False).iloc[:int(p * len(res_load))].index
    data_en_agg_tmp["Aggregated Peak"] = \
        res_load[times_with_highest_absolute_peaks].abs().sum()
    # get capacities
    data_en_agg_tmp["Capacity"] = edisgo_obj.topology.loads_df.loc[
        edisgo_obj.topology.loads_df.sector == "residential"].peak_load.sum()
    # fill in general information
    data_en_agg_tmp["Scenario"] = scenario
    data_en_agg_tmp["Tariff"] = "Ec"
    data_en_agg = pd.concat([data_en_agg, data_en_agg_tmp], axis=1,
                            ignore_index=True)
    # import network tariffs in scenario full_flex
    scenario = "full_flex"
    for tariff in tariffs:
        scalars = pd.read_csv(os.path.join(res_dir, tariff, "scalars.csv"), index_col=0)
        data_en_tmp = pd.DataFrame()
        data_en_agg_tmp = pd.Series(dtype=float)
        data_en_tmp["Consumer"] = scalars.index
        data_en_tmp["Capacity"] = \
            scalars[["peak_load", "peak_feedin"]].max(axis=1).values
        data_en_agg_tmp["Capacity"] = \
            scalars[["peak_load", "peak_feedin"]].max(axis=1).sum()
        data_en_tmp["Electricity Purchased"] = scalars["energy_purchased"].values
        data_en_agg_tmp["Electricity Purchased"] = scalars["energy_purchased"].sum()
        data_en_tmp["Tariff Costs"] = scalars["costs_energy_tariff"].values + scalars[
            "costs_capacity_load_tariff"].values + scalars[
                                      "costs_capacity_feedin_tariff"].fillna(0).values
        data_en_agg_tmp["Reinforcement Costs"] = pd.read_csv(os.path.join(
            res_dir, tariff, "results_after_reinforcement", "grid_expansion_results",
            "grid_expansion_costs.csv"
        ), index_col=0).total_costs.sum()
        edisgo_obj = import_edisgo_from_files(os.path.join(res_dir, tariff),
                                              import_timeseries=True)
        data_en_tmp, data_en_agg_tmp = \
            get_p_percent_highest_peaks_and_customer_contribution(
                data=data_en_tmp,
                data_agg=data_en_agg_tmp,
                edisgo_object=edisgo_obj,
                p=0.1
            )
        # determine customer groups
        households = edisgo_obj.topology.loads_df.loc[
            edisgo_obj.topology.loads_df.sector == "residential"
            ]
        ev = edisgo_obj.topology.loads_df.loc[
            (edisgo_obj.topology.loads_df.type == "charging_point") &
            (edisgo_obj.topology.loads_df.bus.isin(households.bus))
            ]
        hp = edisgo_obj.topology.loads_df.loc[
            (edisgo_obj.topology.loads_df.type == "heat_pump") &
            (edisgo_obj.topology.loads_df.bus.isin(households.bus))
            ]
        pv = edisgo_obj.topology.generators_df.loc[
            (edisgo_obj.topology.generators_df.type == "solar") &
            (edisgo_obj.topology.generators_df.bus.isin(households.bus))
            ]
        bess = edisgo_obj.topology.storage_units_df.loc[
            edisgo_obj.topology.storage_units_df.bus.isin(households.bus)]
        households["has_hp"] = households.bus.isin(hp.bus)
        households["has_ev"] = households.bus.isin(ev.bus)
        households["has_pv"] = households.bus.isin(pv.bus)
        households["has_bess"] = households.bus.isin(
            edisgo_obj.topology.storage_units_df.bus)
        data_en_tmp["Consumer Group"] = households.loc[scalars.index].T.apply(
            lambda x: x.has_hp*"HP" + x.has_ev*"EV" + x.has_pv*"PV" + x.has_bess*"BESS"
        ).values
        data_en_tmp["Vulnerable"] = (households.loc[scalars.index].T.apply(lambda x: (
                    (x.annual_consumption < households.annual_consumption.mean()) and not
                    x.has_pv and not x.has_bess and not x.has_ev))).values
        data_en_tmp["Tariff"] = tariff
        data_en_agg_tmp["Tariff"] = tariff
        data_en_tmp["Scenario"] = scenario
        data_en_agg_tmp["Scenario"] = scenario
        data_en = pd.concat([data_en, data_en_tmp], ignore_index=True)
        data_en_agg = pd.concat([data_en_agg, data_en_agg_tmp], axis=1,
                                ignore_index=True)
    data_en.to_csv(os.path.join(res_dir, "data_effnets.csv"))
    data_en_agg.T.to_csv(os.path.join(res_dir, "data_agg_effnets.csv"))

    print("Success")
