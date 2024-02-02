import pandas as pd
import pyomo.environ as pm


# noinspection PyTypeChecker
def add_consumer_model(model: pm.ConcreteModel,
                       load_ts: pd.Series,
                       add_ev: bool = False,
                       add_hp: bool = False,
                       add_pv: bool = False,
                       add_bess: bool = False,
                       **kwargs: dict):
    """

    :param model: pm.ConcreteModel
    :param load_ts: pd.Series
    :param add_ev: bool
    :param add_hp: bool
    :param add_pv: bool
    :param add_bess: bool
    :param kwargs: dict
        Contains optional parameters for flexibility options, depending on the modelled
        options, the following should be included,
        For EVs:
            "ev_upper_power": pd.Series
                time series of maximum charging power for EV
            "ev_upper_energy": pd.Series
                time series of upper bound on cumulative energy consumption of EV
            "ev_lower_energy": pd.Series
                time series of lower bound on cumulative energy consumption of EV
            "ev_efficiency": float (optional)
                charging efficiency of EV, has to lie between 0 and 1, defaults to 0.9
            "ev_weight_penalty": float (optional)
                weight of penalty for deviation from reference charging, later used for
                objective, defaults to 0.001
            "ev_squared_penalty": bool (optional)
                determines whether penalty for deviation from reference charging is
                squared or not (then linear), defaults to True (-> squared penalty)
        For HPs:
            "hp_p_nom_el": float
                electrical nominal power of heat pump in kW
            "hp_capacity_tes": float
                thermal capacity of thermal energy storage in kWh
            "hp_heat_demand": pd.Series
                heat demand time series of household
            "hp_cop": pd.Series
                time series of coefficient of performance of heat pump
            "hp_fixed_soc_tes": float (optional)
                relative state of charge that should be obtained at beginning and end
                of simulation period, has be lie between 0 and 1, defaults to 0.5
        For PV:
            "gen_ts": pd.Series
                time series of generation units in households
        For BESS:
            "bess_capacity": float
                energy capacity of battery storage in kWh
            "bess_ratio_power_to_energy": float (optional)
                ratio of power to energy capacity in kW/kWh, defaults to 0.6 kW/kWh
            "bess_fixed_soc": float (optional)
                relative state of charge that should be obtained at beginning and end
                of simulation period, has be lie between 0 and 1, defaults to 0.5
    :return:
    """
    def power_balance_household(m, t):
        if m.has_pv:
            effective_feedin = m.effective_feedin[t]
            pv = m.pv[t]
        else:
            effective_feedin = 0
            pv = 0
        flexible_load = \
            [getattr(m, f"charging_{comp}")[t] if getattr(m, f"has_{comp}") else 0
             for comp in ["ev", "hp", "bess"]]
        return m.effective_load[t] - effective_feedin == model.demand[t] + \
            sum(flexible_load) - pv
    # set demand
    model.demand_ts = load_ts
    model.demand = pm.Param(
        model.time_set,
        initialize=set_load,
        mutable=True,
    )
    model.effective_load = pm.Var(
        model.time_set,
        bounds=(0, None),
    )
    # add EV if included
    if add_ev:
        model = add_ev_model(
            model=model,
            upper_power=kwargs["ev_upper_power"],
            upper_energy=kwargs["ev_upper_energy"],
            lower_energy=kwargs["ev_lower_energy"],
            efficiency=kwargs.get("ev_efficiency", 0.9),
            weight_penalty=kwargs.get("ev_weight_penalty", 0.001),
            squared_penalty=kwargs.get("ev_squared_penalty", True)
        )
    else:
        model.has_ev = False
    # add HP if included
    if add_hp:
        model = add_heat_pump_model(
            model=model,
            p_nom_hp=kwargs.get("hp_p_nom_el"),
            capacity_tes=kwargs.get("hp_capacity_tes"),
            heat_demand=kwargs.get("hp_heat_demand"),
            cop=kwargs.get("hp_cop"),
            fixed_soc=kwargs.get("hp_fixed_soc_tes", 0.5)
        )
    else:
        model.has_hp = False
    # add PV unit if included
    if add_pv:
        model = add_pv_model(
            model=model,
            gen_ts=kwargs.get("gen_ts")
        )
    else:
        model.has_pv = False
    # add BESS if included
    if add_bess:
        model = add_bess_model(
            model=model,
            capacity_bess=kwargs.get("bess_capacity"),
            ratio_power_to_energy=kwargs.get("bess_ratio_power_to_energy", 0.6),
            fixed_soc=kwargs.get("bess_fixed_soc", 0.5)
        )
    else:
        model.has_bess = False
    # Set constraint for residual load
    model.PowerBalanceHousehold = pm.Constraint(
        model.time_set,
        rule=power_balance_household
    )
    return model


def add_bess_model(
        model: pm.ConcreteModel,
        capacity_bess: float,
        ratio_power_to_energy: float = 0.6,
        fixed_soc: float = 0.5,
):
    def soc(m, time):
        """Constraint for battery charging"""
        if time == 0:
            soc_pre = model.fix_soe_bess
        else:
            soc_pre = m.soe_bess[time - 1]
        return m.soe_bess[time] == soc_pre + m.charging_bess[time] * (
                       pd.to_timedelta(model.time_increment) / pd.to_timedelta("1h")
               )

    def fix_soe(m, time):
        """Constraint with which state of charge at beginning and end of charging
        period is fixed at certain value"""
        return m.soe_bess[time] == model.fix_soe_bess

    # Set fix parameters
    model.has_bess = True
    model.capacity_bess = pm.Param(initialize=capacity_bess, mutable=True)
    model.fix_soe_bess = pm.Param(initialize=fixed_soc * capacity_bess, mutable=True)
    model.power_bess = pm.Param(initialize=ratio_power_to_energy * model.capacity_bess,
                                mutable=True)
    # Set variables
    model.charging_bess = pm.Var(
        model.time_set,
        bounds=(-model.power_bess, model.power_bess)
    )
    model.soe_bess = pm.Var(
        model.time_set,
        bounds=(0, model.capacity_bess)
    )
    # Set Constraints
    model.BESSCharging = pm.Constraint(
        model.time_set,
        rule=soc
    )
    model.BESSFixedSOC = pm.Constraint(
        model.times_fixed_soc,
        rule=fix_soe
    )
    return model


def add_pv_model(
        model: pm.ConcreteModel,
        gen_ts: pd.DataFrame,
):
    # Set fix parameters
    model.has_pv = True
    model.gen_ts = gen_ts
    # Set time-varying parameters
    model.gen = pm.Param(
        model.time_set,
        initialize=set_generation,
        mutable=True,
    )
    # Set variables
    model.pv = pm.Var(
        model.time_set,
        bounds=lambda m, t: (0, m.gen[t]),
    )
    model.effective_feedin = pm.Var(
        model.time_set,
        bounds=(0, None),
    )
    return model


def add_ev_model(
        model: pm.ConcreteModel,
        upper_power: pd.Series,
        upper_energy: pd.Series,
        lower_energy: pd.Series,
        efficiency: float = 0.9,
        weight_penalty: float = 0.001,
        squared_penalty: bool = True,
):
    def charging_ev(m, time):
        """Constraint for charging of EV that has to ly between the lower and upper
        energy band."""
        if time == 0:
            energy_level_pre = \
                (m.lower_bound_ev[time] + m.upper_bound_ev[time]) / 2
        else:
            energy_level_pre = m.energy_level_ev[time - 1]
        return m.energy_level_ev[time] == \
            energy_level_pre + m.ev_efficiency * m.charging_ev[
                time] * (pd.to_timedelta(m.time_increment) / pd.to_timedelta("1h"))

    def fixed_energy_level_ev(m, time):
        """Constraint to fix ev energy level of specifc charging point and time
        to 50% of flex band potential (1/2 (lower_band + upper_band))"""
        return (
                m.energy_level_ev[time]
                == (m.lower_bound_ev[time] + m.upper_bound_ev[time]) / 2
        )

    def deviation_from_default_ev_charging(m, time):
        """Constraint to determine penalty term for delayed ev charging"""
        return m.penalty_ev[time] == m.upper_ev_energy[time] - \
            m.energy_level_ev[time]
    # Set fix parameters
    model.has_ev = True
    model.upper_ev_power = upper_power
    model.upper_ev_energy = upper_energy
    model.lower_ev_energy = lower_energy
    model.ev_efficiency = pm.Param(initialize=efficiency, mutable=True)
    model.weight_pen_ev = weight_penalty
    model.ev_squared_penalty = squared_penalty
    # Set time-varying parameters
    model.lower_bound_ev = pm.Param(
        model.time_set,
        initialize=set_lower_band_ev,
        mutable=True,
    )
    model.upper_bound_ev = pm.Param(
        model.time_set,
        initialize=set_upper_band_ev,
        mutable=True,
    )
    model.power_bound_ev = pm.Param(
        model.time_set,
        initialize=set_power_band_ev,
        mutable=True,
    )
    # Variables
    model.charging_ev = pm.Var(
        model.time_set,
        bounds=lambda m, t: (0, m.power_bound_ev[t]),
    )

    model.energy_level_ev = pm.Var(
        model.time_set,
        bounds=lambda m, t: (m.lower_bound_ev[t], m.upper_bound_ev[t]),
    )

    model.penalty_ev = pm.Var(model.time_set)
    # Constraints
    model.EVCharging = pm.Constraint(
        model.time_set, rule=charging_ev
    )
    model.EVEnergyEnd = pm.Constraint(
        model.times_fixed_soc, rule=fixed_energy_level_ev
    )
    model.EVPenalty = pm.Constraint(
        model.time_set, rule=deviation_from_default_ev_charging
    )

    return model


def add_heat_pump_model(
        model: pm.ConcreteModel,
        p_nom_hp: float,
        capacity_tes: float,
        heat_demand: pd.Series,
        cop: pd.Series,
        fixed_soc: float = 0.5,
):
    def energy_balance_hp_tes(m, time):
        """Energy balance of heat pump, heat demand and tes"""
        return (
            m.charging_hp[time] * m.cop_hp[time]
            == m.heat_demand_hp[time] + m.charging_tes[time]
        )

    def charging_tes(m, time):
        """Charging balance of tes"""
        if time == 0:
            energy_level_pre = m.soe_fix
        else:
            energy_level_pre = m.energy_level_tes[time - 1]
        return m.energy_level_tes[time] == energy_level_pre + m.charging_tes[time] * (
            pd.to_timedelta(model.time_increment) / pd.to_timedelta("1h")
        )

    def fixed_energy_level_tes(m, time):
        """Fix state of energy of TES"""
        return m.energy_level_tes[time] == m.soe_fix
    # Set fix parameters
    model.has_hp = True
    model.p_nom_hp = pm.Param(initialize=p_nom_hp, mutable=True)
    model.soe_fix = pm.Param(initialize=fixed_soc * capacity_tes, mutable=True)
    model.capacity_tes = pm.Param(initialize=capacity_tes, mutable=True)
    model.heat_demand = heat_demand
    model.cop = cop
    # Set time-dependent parameters
    model.heat_demand_hp = pm.Param(
        model.time_set,
        initialize=set_heat_demand,
        mutable=True,
    )
    model.cop_hp = pm.Param(
        model.time_set,
        initialize=set_cop_hp,
        mutable=True,
    )
    # Define variables
    model.energy_level_tes = pm.Var(
        model.time_set,
        bounds=(0, model.capacity_tes)
    )  # SOC of the TES in kWh
    model.charging_hp = pm.Var(
        model.time_set,
        bounds=(0, model.p_nom_hp)
    )  # electric demand of HP in kW
    model.charging_tes = pm.Var(
        model.time_set,
    )  # thermal demand of TES in kW
    # Add constraints
    model.HPEnergyBalance = pm.Constraint(
        model.time_set,
        rule=energy_balance_hp_tes
    )
    model.HPChargingTES = pm.Constraint(
        model.time_set,
        rule=charging_tes
    )
    model.HPFixEnergyLevelTES = pm.Constraint(
        model.times_fixed_soc,
        rule=fixed_energy_level_tes
    )
    return model


def update_model_new_household(model,
                               load_ts,
                               **kwargs):
    """
    Updates model to new household. Note that the type of household should not change,
    i.e. the same ders should be present as these are currently not checked for.

    :param model: pm.ConcreteModel
    :param load_ts: pd.Series
    :param kwargs: dict
        Contains optional parameters for flexibility options, depending on the modelled
        options, the following should be included,
        For EVs:
            "ev_upper_power": pd.Series
                time series of maximum charging power for EV
            "ev_upper_energy": pd.Series
                time series of upper bound on cumulative energy consumption of EV
            "ev_lower_energy": pd.Series
                time series of lower bound on cumulative energy consumption of EV
            "ev_efficiency": float (optional)
                charging efficiency of EV, has to lie between 0 and 1, defaults to 0.9
        For HPs:
            "hp_p_nom_el": float
                electrical nominal power of heat pump in kW
            "hp_capacity_tes": float
                thermal capacity of thermal energy storage in kWh
            "hp_heat_demand": pd.Series
                heat demand time series of household
            "hp_cop": pd.Series
                time series of coefficient of performance of heat pump
            "hp_fixed_soc_tes": float (optional)
                relative state of charge that should be obtained at beginning and end
                of simulation period, has be lie between 0 and 1, defaults to 0.5
        For PV:
            "gen_ts": pd.Series
                time series of generation units in households
        For BESS:
            "bess_capacity": float
                energy capacity of battery storage in kWh
            "bess_ratio_power_to_energy": float (optional)
                ratio of power to energy capacity in kW/kWh, defaults to 0.6 kW/kWh
            "bess_fixed_soc": float (optional)
                relative state of charge that should be obtained at beginning and end
                of simulation period, has be lie between 0 and 1, defaults to 0.5
    :return:
    """
    # update fix parameters
    del model.demand_ts
    model.demand_ts = load_ts
    if model.has_ev:
        model.ev_efficiency.set_value(kwargs.get("ev_efficiency", 0.9))
        del model.upper_ev_power
        model.upper_ev_power = kwargs["ev_upper_power"]
        del model.upper_ev_energy
        model.upper_ev_energy = kwargs["ev_upper_energy"]
        del model.lower_ev_energy
        model.lower_ev_energy = kwargs["ev_lower_energy"]
    if model.has_hp:
        model.p_nom_hp.set_value(kwargs["hp_p_nom_el"])
        model.soe_fix.set_value(
            kwargs.get("hp_fixed_soc_tes", 0.5)*kwargs["hp_capacity_tes"])
        model.capacity_tes.set_value(kwargs["hp_capacity_tes"])
        del model.heat_demand
        model.heat_demand = kwargs["hp_heat_demand"]
        del model.cop
        model.cop = kwargs["hp_cop"]
    if model.has_pv:
        del model.gen_ts
        model.gen_ts = kwargs["gen_ts"]
    if model.has_bess:
        model.capacity_bess.set_value(kwargs["bess_capacity"])
        model.fix_soe_bess.set_value(
            kwargs.get("bess_fixed_soc", 0.5)*kwargs["bess_capacity"])
        model.power_bess.set_value(
            kwargs.get("bess_ratio_power_to_energy", 0.6)*kwargs["bess_capacity"])

    # update time-varying parameters
    for time in model.time_set:
        model.demand[time].set_value(set_load(model, time))
        if model.has_ev:
            model.lower_bound_ev[time].set_value(set_lower_band_ev(model, time))
            model.upper_bound_ev[time].set_value(set_upper_band_ev(model, time))
            model.power_bound_ev[time].set_value(set_power_band_ev(model, time))
        if model.has_hp:
            model.heat_demand_hp[time].set_value(set_heat_demand(model, time))
            model.cop_hp[time].set_value(set_cop_hp(model, time))
        if model.has_pv:
            model.gen[time].set_value(set_generation(model, time))
    return model


def set_load(model, time):
    return model.demand_ts[model.timeindex[time]]


def set_generation(model, time):
    return model.gen_ts[model.timeindex[time]]


def set_lower_band_ev(model, time):
    return model.lower_ev_energy[model.timeindex[time]]


def set_upper_band_ev(model, time):
    return model.upper_ev_energy[model.timeindex[time]]


def set_power_band_ev(model, time):
    return model.upper_ev_power[model.timeindex[time]]


def set_cop_hp(model, time):
    return model.cop[model.timeindex[time]]


def set_heat_demand(model, time):
    return model.heat_demand[model.timeindex[time]]

