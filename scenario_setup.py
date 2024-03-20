import pandas as pd


def tariff_scenarios(
        timesteps,
        dir_time_varying_suppliers_costs: str,
        mean_suppliers_costs: float = 80.0,
        energy_component: float = 80.0,
        reduction_capacity: float = 40.0,
        energy_component_day: float = 85.0,
        energy_component_night: float = 50.0,
        capacity_load_component: float = 135000,
        capacity_feedin_component: float = 13500,
):
    # set up day night tariff
    day_night = pd.Series(index=timesteps, data=energy_component_day)
    day_night[day_night.index.hour.isin([22, 23, 00, 1, 2, 3, 4, 5, 6])] = \
        energy_component_night
    day_night_reduced = day_night - reduction_capacity
    # get variable suppliers costs
    prices = pd.read_csv(dir_time_varying_suppliers_costs,
                         index_col=0, parse_dates=True, header=None)[1]
    prices.index = timesteps
    prices = prices.divide(prices.mean()).multiply(mean_suppliers_costs)
    return {
        "Ec": {"suppliers_costs": mean_suppliers_costs,
               "energy_component": energy_component, },
        "Edn": {"suppliers_costs": mean_suppliers_costs,
                "energy_component": day_night, },
        "Cl": {"suppliers_costs": mean_suppliers_costs,
               "energy_component": 0,
               "capacity_load_component": capacity_load_component},
        "Ec_Cl": {"suppliers_costs": mean_suppliers_costs,
                  "energy_component": energy_component-reduction_capacity,
                  "capacity_load_component": capacity_load_component/2},
        "Edn_Cl": {"suppliers_costs": mean_suppliers_costs,
                   "energy_component": day_night_reduced,
                   "capacity_load_component": capacity_load_component/2},
        "Ec_Clf":
            {"suppliers_costs": mean_suppliers_costs,
             "energy_component": energy_component-reduction_capacity,
             "capacity_load_component": capacity_load_component/2,
             "capacity_feedin_component": capacity_feedin_component, },
        "Edn_Clf":
            {"suppliers_costs": mean_suppliers_costs,
             "energy_component": day_night_reduced,
             "capacity_load_component": capacity_load_component/2,
             "capacity_feedin_component": capacity_feedin_component, },
        "Ec_Cl_Sv":
            {"suppliers_costs": prices,
             "energy_component": energy_component-reduction_capacity,
             "capacity_load_component": capacity_load_component/2, },
        "Ec_Clf_Sv":
            {"suppliers_costs": prices,
             "energy_component": energy_component-reduction_capacity,
             "capacity_load_component": capacity_load_component/2,
             "capacity_feedin_component": capacity_feedin_component, },
    }
