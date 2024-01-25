from typing import Union
import pandas as pd
import pyomo.environ as pm

from consumer_model import add_consumer_model


def run_model(
        load_ts: pd.Series,
        add_ev: bool,
        add_hp: bool,
        add_pv: bool,
        add_bess: bool,
        suppliers_costs: Union[pd.Series, float],
        feedin_tariff: float,
        energy_component: float = 0,
        capacity_load_component: float = 0,
        capacity_feedin_component: float = 0,
        free_capacity: float = 0,
        solver: str = "gurobi",
        **kwargs: dict
):
    if type(suppliers_costs) is float:
        suppliers_costs = pd.Series(index=load_ts.index, data=suppliers_costs)
    model = setup_model(
        load_ts=load_ts,
        add_ev=add_ev,
        add_hp=add_hp,
        add_pv=add_pv,
        add_bess=add_bess,
        suppliers_costs=suppliers_costs,
        feedin_tariff=feedin_tariff,
        energy_component=energy_component,
        capacity_load_component=capacity_load_component,
        capacity_feedin_component=capacity_feedin_component,
        free_capacity=free_capacity,
        **kwargs
    )
    results = optimise_model(model, load_ts.index, solver=solver)
    results["load"] = load_ts
    if add_pv:
        results["pv_orig"] = kwargs.get("gen_ts")
    return results


def setup_model(
        load_ts: pd.Series,
        add_ev: bool,
        add_hp: bool,
        add_pv: bool,
        add_bess: bool,
        suppliers_costs: pd.Series,
        feedin_tariff: float,
        energy_component: float = 0,
        capacity_load_component: float = 0,
        capacity_feedin_component: float = 0,
        free_capacity: float = 0,
        **kwargs: dict
):
    model = setup_base_model(load_ts.index)
    model = add_consumer_model(
        model=model,
        load_ts=load_ts,
        add_ev=add_ev,
        add_hp=add_hp,
        add_pv=add_pv,
        add_bess=add_bess,
        **kwargs
    )
    model = add_network_tariff_model(
        model=model,
        suppliers_costs=suppliers_costs,
        feedin_tariff=feedin_tariff,
        energy_component=energy_component,
        capacity_load_component=capacity_load_component,
        capacity_feedin_component=capacity_feedin_component,
        free_capacity=free_capacity
    )
    return model


def setup_base_model(timesteps):
    """
    Setup of basic model containing relevant time steps

    :param timesteps:
    :return:
    """
    # setup model
    model = pm.ConcreteModel()
    model.time_set = pm.RangeSet(0, len(timesteps) - 1)
    model.times_fixed_soc = pm.Set(
        initialize=[model.time_set.at(-1)]
    )
    model.timeindex = pm.Param(
        model.time_set,
        initialize={i: timesteps[i] for i in model.time_set},
        within=pm.Any,
        mutable=True,
    )
    model.time_increment = pd.infer_freq(timesteps)
    if not any(char.isdigit() for char in model.time_increment):
        model.time_increment = "1" + model.time_increment
    return model


def add_network_tariff_model(model: pm.ConcreteModel,
                             suppliers_costs: pd.Series,
                             feedin_tariff: float,
                             energy_component: float = 0,
                             capacity_load_component: float = 0,
                             capacity_feedin_component: float = 0,
                             free_capacity: float = 0):
    def peak_load(m, t):
        return m.peak_load >= m.effective_load[t]

    def peak_feedin(m, t):
        return m.peak_feedin >= m.effective_feedin[t]
    # set network tariff price components
    for tariff_component in ["energy_component", "capacity_load_component",
                             "capacity_feedin_component"]:
        # determine whether tariff component should be accounted for (>0)
        setattr(model, f"has_{tariff_component}", locals()[tariff_component] > 0)
        # add tariff component to model if relevant
        if getattr(model, f"has_{tariff_component}"):
            setattr(model, tariff_component, locals()[tariff_component])
    # set prices for purchase of electricity (suppliers costs) and feedin tariff for PV
    model.suppliers_costs = pm.Param(
        model.time_set,
        initialize={i: suppliers_costs[model.timeindex[i]] for i in model.time_set},
        mutable=True,
    )
    model.feedin_tariff = feedin_tariff
    if feedin_tariff > suppliers_costs.mean():
        raise Warning("Provided feed-in tariff is lower than mean of suppliers costs. "
                      "Note that some modelling choices are based on the assumption "
                      "that suppliers costs are higher than the feed-in tariff. The "
                      "results could therefore be wrong.")
    # if capacity components are included, add variables for peak load and/or peak
    # feed-in
    if model.has_capacity_load_component:
        model.peak_load = pm.Var(
            bounds=(0, None)
        )
        model.PeakLoad = pm.Constraint(
            model.time_set,
            rule=peak_load
        )
    if model.has_capacity_feedin_component:
        model.peak_feedin = pm.Var(
            bounds=(free_capacity, None)
        )
        model.PeakGen = pm.Constraint(
            model.time_set,
            rule=peak_feedin
        )
    # Set up objective
    model.objective = \
        pm.Objective(rule=minimize_total_consumer_costs, sense=pm.minimize,
                     doc='Define objective function to minimize household costs and '
                         'penalty for user discomfort')
    return model


def minimize_total_consumer_costs(model):
    if hasattr(model, "penalty_ev"):
        if model.ev_squared_penalty:
            penalty_ev = model.weight_pen_ev * \
                         sum(model.penalty_ev[t]**2 for t in model.time_set)
        else:
            penalty_ev = model.weight_pen_ev * \
                         sum(model.penalty_ev[t] for t in model.time_set)
    else:
        penalty_ev = 0
    if hasattr(model, "effective_feedin"):
        revenues = model.feedin_tariff * sum(model.effective_feedin[t]
                                             for t in model.time_set)
    else:
        revenues = 0
    energy_based_costs = \
        [(model.suppliers_costs[t] + model.energy_component) * model.effective_load[t]
         if model.has_energy_component else
         model.suppliers_costs[t] * model.effective_load[t]
         for t in model.time_set]
    capacity_based_costs = \
        [getattr(model, f"capacity_{typ}_component")*getattr(model, f"peak_{typ}")
         if getattr(model, f"has_capacity_{typ}_component") else 0
         for typ in ["load", "feedin"]]
    return sum(energy_based_costs) + sum(capacity_based_costs) - revenues + penalty_ev


def optimise_model(
        model,
        timesteps,
        solver: str = "gurobi",
):
    opt = pm.SolverFactory(solver)
    opt.solve(model, tee=True)

    result = pd.DataFrame(index=timesteps)
    # inflexible load
    result['effective_load'] = pd.Series(model.effective_load.extract_values()).values
    if model.has_capacity_load_component:
        result["peak_load"] = model.peak_load.value
    # PV
    if model.has_pv:
        result['effective_feedin'] = \
            pd.Series(model.effective_feedin.extract_values()).values
        result["pv_feedin"] = pd.Series(model.pv.extract_values()).values
        if model.has_capacity_feedin_component:
            result["peak_feedin"] = model.peak_feedin.value
    # BESS
    if model.has_bess:
        result["charging_bess"] = pd.Series(model.charging_bess.extract_values()).values
        result["soe_bess"] = pd.Series(model.soe_bess.extract_values()).values
    # EVs
    if model.has_ev:
        result["charging_ev"] = pd.Series(model.charging_ev.extract_values()).values
        result["energy_level_ev"] = \
            pd.Series(model.energy_level_ev.extract_values()).values
        result["penalty_ev"] = pd.Series(model.penalty_ev.extract_values()).values
    # HPs
    if model.has_hp:
        result["charging_hp"] = pd.Series(model.charging_hp.extract_values()).values
        result["charging_tes"] = pd.Series(model.charging_tes.extract_values()).values
        result["energy_level_tes"] = \
            pd.Series(model.energy_level_tes.extract_values()).values
    if pm.value(model.objective) < 0:
        raise ValueError
    return result

