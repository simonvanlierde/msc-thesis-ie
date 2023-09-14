"""Environmental impact calculation functions for the model.

@author: Simon van Lierde
"""

import pandas as pd


def calculate_environmental_parameters_for_cooling_technologies(
    cooling_technology_parameters: list[dict[str, float]],
    global_parameters: dict[str, float],
) -> list[dict[str, float]]:
    """Calculates the derivative cooling technology parameters for each cooling technology.

    Args:
        cooling_technology_parameters (list[dict[str, float]]): _description_
        global_parameters (dict[str, float]): _description_

    Returns:
        list[dict[str, float]]: An updated list of cooling technology parameters
    """
    # Unload global parameters:
    carbon_intensity_electric_grid_kgCO2eq_kWh = global_parameters["carbon_intensity_electric_grid_kgCO2eq_kWh"]  # Carbon intensity of the electric grid in kg CO2-eq per kWh
    gwp_refrigerant_kgCO2eq_kg = global_parameters[
        "gwp_refrigerant_kgCO2eq_kg"
    ]  # Global warming potential of the refrigerant in kg CO2-eq per kg refrigerant emitted into the atmosphere
    carbon_intensity_production_kgCO2eq_kg = global_parameters[
        "carbon_intensity_production_kgCO2eq_kg"
    ]  # Carbon intensity of the production of the cooling equipment in kg CO2-eq per kg cooling equipment
    carbon_intensity_EoL_kgCO2eq_kg = global_parameters[
        "carbon_intensity_EoL_kgCO2eq_kg"
    ]  # Carbon intensity of the end-of-life phase of the cooling equipment in kg CO2-eq per kg cooling equipment
    adp_intensity_cooling_equipment_kgSbeq_kg = global_parameters[
        "adp_intensity_cooling_equipment_kgSbeq_kg"
    ]  # Abiotic depletion potential of the cooling equipment in kg Sb-eq per kg cooling equipment
    csi_intensity_cooling_equipment_kgSbeq_kg = global_parameters[
        "csi_intensity_cooling_equipment_kgSbeq_kg"
    ]  # Critical scarcity index of the cooling equipment in kg Sb-eq per kg cooling equipment

    # Convert the cooling technology parameters to a DataFrame
    df_cooling_technologies = pd.DataFrame(cooling_technology_parameters)

    # Calculate the inverse Seasonal Energy Efficiency Ratio (SEER) for each cooling technology, for easier matrix multiplication
    df_cooling_technologies["SEER_inv"] = 1 / df_cooling_technologies["SEER"]  # The inverse SEER is the amount of electricity needed to produce 1 unit of cooling

    # Calculate the greenhouse gas (GHG) emissions in kg CO2-eq per kWh cooling energy demand
    df_cooling_technologies["GHG_emissions_electricity_kgCO2eq_kWh_cooling"] = carbon_intensity_electric_grid_kgCO2eq_kWh * df_cooling_technologies["SEER_inv"]

    # Calculate the effective annual refrigerant leakage in kg per kW of installation power
    df_cooling_technologies["refrigerant_leakage_kg_kW"] = df_cooling_technologies["refrigerant_leakage_rate_relative"] * df_cooling_technologies["refrigerant_density_kg_kW"]

    # Calculate the annual climate change impact from refrigerant leakage in kg CO2-eq per kW of installation power
    df_cooling_technologies["GHG_emissions_refrigerant_leaks_kgCO2eq_kW"] = gwp_refrigerant_kgCO2eq_kg * df_cooling_technologies["refrigerant_leakage_kg_kW"]

    # Calculate the annualized climate change impact from the production phase of the cooling technology in kg CO2-eq per kW of installation power
    df_cooling_technologies["GHG_emissions_production_phase_kgCO2eq_kW"] = (
        carbon_intensity_production_kgCO2eq_kg * df_cooling_technologies["material_density_kg_kW"] / df_cooling_technologies["average_lifetime_yr"]
    )

    # Calculate the annualized climate change impact from the end-of-life phase of the cooling technology in kg CO2-eq per kW of installation power
    df_cooling_technologies["GHG_emissions_EoL_phase_kgCO2eq_kW"] = (
        carbon_intensity_EoL_kgCO2eq_kg * df_cooling_technologies["material_density_kg_kW"] / df_cooling_technologies["average_lifetime_yr"]
    )

    # Calculate the annualized abiotic depletion potential (ADP) from the production phase of the cooling technology in kg Sb-eq per kW of installation power
    df_cooling_technologies["ADP_kgSbeq_kW"] = (
        adp_intensity_cooling_equipment_kgSbeq_kg * df_cooling_technologies["material_density_kg_kW"] / df_cooling_technologies["average_lifetime_yr"]
    )

    # Calculate the annualized crustal scarcity index (CSI) from the production phase of the cooling technology in kg Sb-eq per kW of installation power
    df_cooling_technologies["CSI_kgSieq_kW"] = (
        csi_intensity_cooling_equipment_kgSbeq_kg * df_cooling_technologies["material_density_kg_kW"] / df_cooling_technologies["average_lifetime_yr"]
    )

    return df_cooling_technologies


def calculate_impacts_from_cooling_energy_demand(buildings: pd.DataFrame, cap_percentile: int = 98) -> pd.DataFrame:
    """Calculates the environmental impacts that depend on the cooling energy demand of the buildings.

    Args:
        buildings (pd.DataFrame): The buildings DataFrame for which the cooling energy and power demand has been calculated.
        cap_percentile (int, optional): The percentile at which the cooling energy demand is capped. Defaults to 98.

    Returns:
        pd.DataFrame: The buildings DataFrame with the environmental impacts depending on cooling energy demand added.
    """
    # Fetch the cooling energy demand capped at the nth percentile of peak power demand, in kWh
    cooling_energy_demand_kWh = buildings[f"E_cooling_capped_at_{cap_percentile}th_percentile_kWh"]

    # Calculate the total electrical energy use of the buildings in kWh
    buildings["electricity_use_kWh"] = cooling_energy_demand_kWh * buildings["avg_SEER_inv"]

    # Calculate the total greenhouse gas (GHG) emissions from the cooling energy demand of the buildings in kg CO2-eq
    buildings["GHG_emissions_electricity_kgCO2eq"] = cooling_energy_demand_kWh * buildings["avg_GHG_emissions_electricity_kgCO2eq_kWh_cooling"]

    return buildings


def calculate_impacts_from_peak_cooling_power_demand(buildings: pd.DataFrame, cap_percentile: int = 98) -> pd.DataFrame:
    """Calculates the environmental impacts that depend on the equipment installation size, and thus the peak cooling power demand, of the buildings.

    Args:
        buildings (pd.DataFrame): The buildings DataFrame for which the cooling energy and power demand has been calculated.
        cap_percentile (int, optional): The percentile at which the peak power is capped. Defaults to 98.

    Returns:
        pd.DataFrame: The buildings DataFrame with the environmental impacts depending on peak cooling power demand added.
    """
    # Fetch the peak cooling power demand capped at the nth percentile, in kW
    peak_cooling_power_demand_kW = buildings[f"P_cooling_peak_{cap_percentile}th_percentile_kW"]

    # Calculate the total greenhouse gas (GHG) emissions from leaking refrigerants in kg CO2-eq
    buildings["GHG_emissions_refrigerant_leaks_kgCO2eq"] = peak_cooling_power_demand_kW * buildings["avg_GHG_emissions_refrigerant_leaks_kgCO2eq_kW"]

    # Calculate the GHG emissions from the production of cooling equipment in kg CO2-eq
    buildings["GHG_emissions_production_phase_kgCO2eq"] = peak_cooling_power_demand_kW * buildings["avg_GHG_emissions_production_phase_kgCO2eq_kW"]

    # Calculate the GHG emissions from the treatment of waste cooling equipment in kg CO2-eq
    buildings["GHG_emissions_EoL_phase_kgCO2eq"] = peak_cooling_power_demand_kW * buildings["avg_GHG_emissions_EoL_phase_kgCO2eq_kW"]

    # Calculate the mass of cooling equipment in kg
    buildings["mass_cooling_equipment_kg"] = peak_cooling_power_demand_kW * buildings["avg_material_density_kg_kW"]

    # Calculate the abiotic depletion potential (ADP) from the production of cooling equipment in kg Sb-eq
    buildings["ADP_kgSbeq"] = peak_cooling_power_demand_kW * buildings["avg_ADP_kgSbeq_kW"]

    # Calculate the crustal scarcity index (CSI) from the production of cooling equipment in kg Sb-eq
    buildings["CSI_kgSieq"] = peak_cooling_power_demand_kW * buildings["avg_CSI_kgSieq_kW"]

    return buildings


def create_impact_summary(
    buildings: pd.DataFrame,
    cap_percentile: int = 98,
) -> dict:
    """Creates a summary of the environmental impacts of the buildings. Mostly used in sensitivity analysis.

    Args:
        buildings (pd.DataFrame): The buildings DataFrame for which the environmental impacts have been calculated.
        cap_percentile (int, optional): The percentile at which the peak power is capped. Defaults to 98.

    Returns:
        dict: A summary of the environmental impacts of the total building stock.
    """
    # Create a brief dictionary with the total environmental impacts of the total building stock
    return {
        "total_impacts": {
            "E_cooling_demand_capped_at_percentile_GWh": round(
                buildings[f"E_cooling_capped_at_{cap_percentile}th_percentile_kWh"].sum() / 1_000_000,
                5,
            ),  # Total cooling energy demand in GWh
            "P_cooling_demand_peak_percentile_MW": round(buildings[f"P_cooling_peak_{cap_percentile}th_percentile_kW"].sum() / 1_000, 5),  # Total peak cooling power demand in MW
            "electricity_use_total_GWh": round(buildings["electricity_use_kWh"].sum() / 1_000_000, 5),  # Total electricity use in GWh
            "GHG_emissions_total_tonneCO2eq": round(buildings["GHG_emissions_total_kgCO2eq"].sum() / 1_000, 5),  # Total GHG emissions in tonne CO2-eq
            "material_use_total_tonne": round(buildings["mass_cooling_equipment_kg"].sum() / 1_000, 5),  # Total material use in tonne
        },
        "impacts_per_floor_area": {
            "E_cooling_capped_at_percentile_kWh_m2": round(
                buildings[f"E_cooling_capped_at_{cap_percentile}th_percentile_Wh_m2"].mean() / 1000,
                5,
            ),  # Cooling energy demand intensity in GWh/m2
            "P_cooling_peak_percentile_W_m2": round(buildings[f"P_cooling_peak_{cap_percentile}th_percentile_W_m2"].mean(), 5),  # Peak cooling power demand intensity in W/m2
            "electricity_use_intensity_kWh_m2": round(buildings["electricity_use_intensity_kWh_m2"].mean(), 5),  # Electricity use intensity in kWh/m2
            "GHG_emissions_intensity_kgCO2eq_m2": round(buildings["GHG_emissions_intensity_kgCO2eq_m2"].mean(), 5),  # GHG emissions intensity in kg CO2eq/m2
            "material_use_intensity_kg_m2": round(buildings["material_use_intensity_kg_m2"].mean(), 5),  # Material use intensity in kg/m2
        },
    }


def calculate_environmental_impacts_from_cooling_demand(
    buildings: pd.DataFrame,
    global_parameters: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Aggregation function that calculates the total environmental impacts from cooling energy and power demand of the buildings.

    Args:
        buildings (pd.DataFrame): The buildings DataFrame for which the cooling energy and power demand has been calculated.
        global_parameters (dict[str, float]): A dictionary with global parameters, containing the percentile at which the cooling energy and power demand is capped.

    Returns:
        tuple[pd.DataFrame, dict[str, float]]: The buildings DataFrame with the environmental impacts depending on cooling energy and power demand added, and an impact summary for sensitivity analysis.
    """
    # Unload global parameters
    cap_percentile = int(global_parameters["peak_cooling_percentile_cap"])  # The percentile at which the cooling energy and power demand is capped

    # Calculate the environmental impacts from cooling energy demand
    buildings = calculate_impacts_from_cooling_energy_demand(buildings, cap_percentile)

    # Calculate the environmental impacts from peak cooling power demand
    buildings = calculate_impacts_from_peak_cooling_power_demand(buildings, cap_percentile)

    # Calculate the total GHG emissions in kg CO2-eq
    buildings["GHG_emissions_total_kgCO2eq"] = (
        buildings["GHG_emissions_electricity_kgCO2eq"]
        + buildings["GHG_emissions_refrigerant_leaks_kgCO2eq"]
        + buildings["GHG_emissions_production_phase_kgCO2eq"]
        + buildings["GHG_emissions_EoL_phase_kgCO2eq"]
    )

    # Calculate the impact intensity (per floor area) for the three main impact categories
    buildings["electricity_use_intensity_kWh_m2"] = buildings["electricity_use_kWh"] / buildings["floor_area_total_m2"]
    buildings["material_use_intensity_kg_m2"] = buildings["mass_cooling_equipment_kg"] / buildings["floor_area_total_m2"]
    buildings["GHG_emissions_intensity_kgCO2eq_m2"] = buildings["GHG_emissions_total_kgCO2eq"] / buildings["floor_area_total_m2"]

    # Create an impact summary for sensitivity analysis
    impact_summary = create_impact_summary(buildings, cap_percentile)

    return buildings, impact_summary
