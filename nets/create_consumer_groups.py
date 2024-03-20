import os
import pandas as pd

from edisgo.edisgo import import_edisgo_from_files


res_dir = r"H:\Tariffs\Consumer_Groups"
grid_id = 176
grid_dir = r"H:\Grids_SE_storage\{}".format(grid_id)
ev_dir = r"C:\Users\aheider\Documents\Grids\{}".format(grid_id)
n = 500

edisgo_obj = import_edisgo_from_files(
    grid_dir, import_timeseries=True, import_heat_pump=True)

# handle heat pumps
sub_dir_comp = "Heat_Pumps"
hp = edisgo_obj.topology.loads_df.loc[
    (edisgo_obj.topology.loads_df.type == "heat_pump")
].sample(n=n)
hp["tes_capacity"] = \
    edisgo_obj.heat_pump.thermal_storage_units_df.loc[hp.index, "capacity"]
hp.to_csv(os.path.join(res_dir, sub_dir_comp, "heat_pumps.csv"))
hp_ts = {}
for ts in ["heat_demand", "cop"]:
    ts_tmp = getattr(edisgo_obj.heat_pump, f"{ts}_df")[hp.index]
    ts_tmp.to_csv(os.path.join(res_dir, sub_dir_comp, f"{ts}.csv"))

# handle EVs
sub_dir_comp = "Electric_Vehicles"
evs = pd.read_csv(os.path.join(ev_dir, "dumb", "topology", "charging_points.csv"),
                  index_col=0)
evs = evs.loc[evs.use_case == "home"].sample(n=n)
evs.to_csv(os.path.join(res_dir, sub_dir_comp, f"charging_points.csv"))
reference_charging = pd.read_csv(os.path.join(ev_dir, "dumb", "timeseries",
                                              "charging_points_active_power.csv"),
                                 index_col=0, parse_dates=True)[evs.index]
reference_charging.resample("1h").mean().loc[edisgo_obj.timeseries.timeindex].to_csv(
    os.path.join(res_dir, sub_dir_comp, f"charging_points_active_power.csv")
)
for band in ["upper_power", "upper_energy", "lower_energy"]:
    ts_tmp = pd.read_csv(os.path.join(ev_dir, f"{band}_home.csv"), index_col=0,
                         parse_dates=True)[evs.index]
    if band == "upper_power":
        ts_tmp = ts_tmp.resample("1h").mean().loc[edisgo_obj.timeseries.timeindex]
    else:
        ts_tmp = ts_tmp.resample("1h").max().loc[edisgo_obj.timeseries.timeindex]
    ts_tmp.to_csv(os.path.join(res_dir, sub_dir_comp, f"{band}.csv"))

# handle BESS
sub_dir_comp = "Battery_Storage"
bess = edisgo_obj.topology.storage_units_df.sample(n=n)
bess.to_csv(os.path.join(res_dir, sub_dir_comp, "storage_units.csv"))
edisgo_obj.timeseries.storage_units_active_power[bess.index].to_csv(
    os.path.join(res_dir, sub_dir_comp, "storage_units_active_power.csv"))
# handle pv plants
sub_dir_comp = "Photovoltaic"
pv = edisgo_obj.topology.generators_df.loc[
    (edisgo_obj.topology.generators_df.type == "solar") &
    (edisgo_obj.topology.generators_df.bus.isin(bess.bus))
]
pv.to_csv(os.path.join(res_dir, sub_dir_comp, "generators.csv"))
edisgo_obj.timeseries.generators_active_power[pv.index].to_csv(
    os.path.join(res_dir, sub_dir_comp, "generators_active_power.csv"))

print("Success")
