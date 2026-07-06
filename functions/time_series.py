"""This file contains the functions to get the weather data from the API and create time series for the cooling demand model.

@author: Simon van Lierde
"""

import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

MIN_RESPONSE_LINES_FOR_DATA = 15  # Below this line count, the KNMI API returned only header info and no data


def _parse_knmi_response(text: str) -> pd.DataFrame:
    """Parse a KNMI hourly-data response, locating the header row dynamically.

    The export prefixes the data with a variable number of ``#`` comment lines;
    the column header is the last comment line before the first data row. Column
    names are normalised (leading ``# `` stripped, ``HH`` -> ``H``) to match the
    model, which insulates against the KNMI format drifting.
    """
    lines = text.splitlines()
    first_data_row = next(
        index for index, line in enumerate(lines) if line.strip() and not line.lstrip().startswith("#")
    )
    weather = pd.read_csv(io.StringIO(text), header=first_data_row - 1, skipinitialspace=True)
    weather.columns = weather.columns.str.lstrip("# ").str.strip()
    weather = weather.rename(columns={"HH": "H"})
    if len(weather) <= MIN_RESPONSE_LINES_FOR_DATA:
        msg = "KNMI API returned no data rows."
        raise ValueError(msg)
    return weather


def get_raw_weather_data(global_parameters: dict[str, float]) -> pd.DataFrame:
    """Get raw weather data from KNMI API and convert to DataFrame.

    Args:
        global_parameters: dict[str, float]: The dictionary containing the global parameters for the cooling demand model.

    Returns:
        pd.DataFrame:  DataFrame containing the raw weather data.
    """
    # Unpack parameters from the global parameters dictionary
    start_year = int(global_parameters["weather_data_start_year"])  # The starting year of the desired weather series.
    end_year = int(global_parameters["weather_data_end_year"])  # The ending year of the desired weather series.
    station = int(
        global_parameters["weather_station"],
    )  # The number of the KNMI weather station. Defaults to 330 (Hoek van Holland).
    # All weather station numbers can be found here: https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/AWS_stationsmetadata.txt

    # Define parameters for the KNMI API request
    start_time = f"{start_year}010101"  # Start time for KNMI API request
    end_time = f"{end_year}123124"  # End time for KNMI API request
    weather_measurements = "T:Q"  # Temperature (in tenths of °C) and solar radiation (in J/cm2)

    knmi_url = "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"  # KNMI API url
    knmi_params = {
        "start": start_time,
        "end": end_time,
        "vars": weather_measurements,
        "stns": str(station),
    }  # KNMI API parameters

    backup_local_weather_data_path = (
        "data/input/parameters/raw_weather_data_2018_2022_HvH.csv"  # Local backup used when the API is unavailable
    )

    # Fetch from the KNMI API, falling back to the committed backup file if the
    # request fails or the response can't be parsed (timeout, empty body, or the
    # API format drifting from what the parser expects).
    used_backup = False
    try:
        response = requests.post(knmi_url, data=knmi_params, timeout=10)  # Send request to KNMI API
        response.raise_for_status()
        weather_series_df = _parse_knmi_response(response.text)
    except (requests.exceptions.RequestException, pd.errors.ParserError, ValueError, StopIteration) as error:
        warnings.warn(
            f"KNMI API unavailable or unparseable ({error}); using local backup weather data.",
            stacklevel=2,
        )
        weather_series_df = _parse_knmi_response(Path(backup_local_weather_data_path).read_text())
        used_backup = True

    weather_series_df.columns = (
        weather_series_df.columns.str.strip()
    )  # Strip all columns names from extraneous whitespace

    # Convert the raw temperature data to numeric and divide by 10 to get °C
    weather_series_df["T_outdoor_raw_C"] = pd.to_numeric(weather_series_df["T"], errors="coerce").astype("float") / 10

    # Read in string as datetime
    weather_series_df["date"] = pd.to_datetime(
        weather_series_df["YYYYMMDD"].astype(str) + (weather_series_df["H"] - 1).astype(str).str.zfill(2),
        format="%Y%m%d%H",
        errors="coerce",
    )

    # Remove leap days to prevent errors
    weather_series_df = weather_series_df[weather_series_df["date"].dt.strftime("%m-%d") != "02-29"].reset_index(
        drop=True,
    )

    # The API sometimes returns the first day of the year after end_year, so cut anything past end_year
    weather_series_df = weather_series_df[weather_series_df["date"].dt.year <= end_year].reset_index(drop=True)

    # Never silently model the wrong period: the backup only covers a fixed set of years, so refuse it
    # (rather than returning stale data) when it does not cover the requested weather years.
    if used_backup:
        available_years = {int(year) for year in weather_series_df["date"].dt.year.unique()}
        requested_years = set(range(start_year, end_year + 1))
        if not requested_years <= available_years:
            msg = (
                f"The local backup weather file covers years {sorted(available_years)}, but the model requested "
                f"{start_year}-{end_year}. Download the matching KNMI data or adjust weather_data_start_year / "
                f"weather_data_end_year."
            )
            raise ValueError(msg)

    return weather_series_df


def add_UHI_effect(weather_series_df: pd.DataFrame, UHI_effect_day_C: float, UHI_effect_night_C: float) -> pd.DataFrame:
    """Add UHI time of day boosts to weather DataFrame.

    Args:
        weather_series_df (pd.DataFrame): Weather DataFrame containing the air temperature in °C.
        UHI_effect_day_C (float): UHI effect during the day, in °C.
        UHI_effect_night_C (float): UHI effect during the night, in °C.

    Returns:
        pd.DataFrame: Weather DataFrame containing the time of day dependent UHI boosts for each hour in the time series.
    """
    day_start_hour = 8  # Start hour of the day, non-inclusive
    day_end_hour = 20  # End hour of the day, inclusive

    # Apply the day boost when the hour is between 8 and 20, otherwise the night boost
    is_day = (weather_series_df["H"] > day_start_hour) & (weather_series_df["H"] <= day_end_hour)
    weather_series_df["UHI_effect_C"] = np.where(is_day, UHI_effect_day_C, UHI_effect_night_C)

    return weather_series_df


def add_seasonal_temperature_boosts(
    weather_series_df: pd.DataFrame,
    delta_T_winter_C: float,
    delta_T_spring_C: float,
    delta_T_summer_C: float,
    delta_T_autumn_C: float,
) -> pd.DataFrame:
    """Add seasonal temperature boosts to weather DataFrame.

    Args:
        weather_series_df (pd.DataFrame): Weather DataFrame containing the air temperature in °C.
        delta_T_winter_C (float): Temperature boost during the winter, in °C.
        delta_T_spring_C (float): Temperature boost during the spring, in °C.
        delta_T_summer_C (float): Temperature boost during the summer, in °C.
        delta_T_autumn_C (float): Temperature boost during the autumn, in °C.

    Returns:
        pd.DataFrame: Weather DataFrame containing the seasonal temperature boosts for each hour in the time series.
    """
    # Define the start and end months of each season (except autumn, which is the rest of the year)
    winter_months = [12, 1, 2]
    spring_months = [3, 4, 5]
    summer_months = [6, 7, 8]

    # Map each hour to its seasonal boost based on the month, defaulting to autumn
    month = weather_series_df["date"].dt.month
    weather_series_df["delta_T_season_C"] = np.select(
        [month.isin(winter_months), month.isin(spring_months), month.isin(summer_months)],
        [delta_T_winter_C, delta_T_spring_C, delta_T_summer_C],
        default=delta_T_autumn_C,
    )

    return weather_series_df


def add_seasonal_solar_radiation_boosts(
    weather_series_df: pd.DataFrame,
    delta_P_solar_summer: float,
    delta_P_solar_RoY: float,
) -> pd.DataFrame:
    """Add seasonal solar radiation boosts to weather DataFrame.

    Args:
        weather_series_df (pd.DataFrame): Weather DataFrame containing the hourly solar radiation in J/cm2.
        delta_P_solar_summer (float): Relative increase factor for solar radiation during the summer.
        delta_P_solar_RoY (float): Relative increase factor for solar radiation during the rest of the year.

    Returns:
        pd.DataFrame: Weather DataFrame containing the seasonal solar radiation boosts for each hour in the time series.
    """
    # Define the start and end months of the summer season
    summer_months = [6, 7, 8]

    # Apply the summer boost in the summer months, otherwise the rest-of-year boost
    weather_series_df["delta_P_solar"] = np.where(
        weather_series_df["date"].dt.month.isin(summer_months),
        delta_P_solar_summer,
        delta_P_solar_RoY,
    )

    # Rename the original column containing solar radiation data
    weather_series_df = weather_series_df.rename(columns={"Q": "Q_sol_raw_J_cm2"})

    # Multiply the solar radiation with the seasonal solar radiation boost factor and convert from J/cm2 to W/m2 (which can be done as long as the data is in hourly resolution)
    weather_series_df["P_sol_total_W_m2"] = (
        weather_series_df["Q_sol_raw_J_cm2"] * (1 + weather_series_df["delta_P_solar"]) / 3600 * 10000
    )

    return weather_series_df


def add_multidirectional_solar_radiation(
    weather_series_df: pd.DataFrame,
    multidirectional_solar_radiation_fractions_path: str,
) -> pd.DataFrame:
    """Add multidirectional solar radiation to weather DataFrame.

    Args:
        weather_series_df (pd.DataFrame): Weather DataFrame containing the solar radiation in W/m2.
        multidirectional_solar_radiation_fractions_path (str): Path to file containing multidirectional solar radiation fractions for one year.

    Returns:
        pd.DataFrame: Weather DataFrame containing the multidirectional solar radiation in W/m2 for each hour in the time series.
    """
    # Read in multidirectional solar radiation fractions
    multidirectional_solar_radiation_fractions = pd.read_csv(multidirectional_solar_radiation_fractions_path)

    # Determine the amount of years of the weather DataFrame
    years = len(weather_series_df["date"].dt.year.unique())

    # Repeat the multidirectional solar radiation fractions for the amount of years
    multidirectional_solar_radiation_fractions = pd.concat(
        [multidirectional_solar_radiation_fractions] * years,
    ).reset_index(drop=True)

    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]  # Define the eight compass directions

    # Create a new column for each of the eight compass directions (N, NE, E, etc.) multiplying the "Q" radiation with the multidirectional solar radiation fraction in that direction
    for direction in directions:
        weather_series_df[f"P_sol_{direction}_W_m2"] = (
            weather_series_df["P_sol_total_W_m2"] * multidirectional_solar_radiation_fractions[direction]
        )

    return weather_series_df


def add_presence_load_factors(time_series_df: pd.DataFrame, presence_load_factors_path: str) -> pd.DataFrame:
    """Add presence load factors to a time series DataFrame.

    Args:
        time_series_df (pd.DataFrame): Time series DataFrame that the presence load factors will be added to.
        presence_load_factors_path (str): Path to file containing presence load factors for one day.

    Returns:
        pd.DataFrame: Weather DataFrame containing the presence load factors for each hour in the time series.
    """
    # Read in presence load factors
    presence_load_factors = pd.read_csv(presence_load_factors_path)

    # Determine the amount of days of the hourly time series
    days = len(time_series_df) / 24

    # Repeat the presence load DataFrame factors for the amount of days
    presence_load_factors = pd.concat([presence_load_factors] * int(days)).reset_index(drop=True)

    # Join the presence load factors to the weather DataFrame
    return time_series_df.join(presence_load_factors)


def create_time_series(
    global_parameters: dict[str, float],
    raw_weather_df: pd.DataFrame,
    multidirectional_solar_radiation_fractions_path: str,
    presence_load_factors_path: str,
) -> dict[str, np.ndarray]:
    """Assembles the hourly model time series (weather, solar and presence factors) into a dictionary of numpy arrays.

    Args:
        global_parameters (dict[str, float]): The dictionary containing the global parameters for the cooling demand model.
        raw_weather_df (pd.DataFrame): DataFrame containing the raw weather data.
        multidirectional_solar_radiation_fractions_path (str): Path to file containing multidirectional solar radiation fractions for one year.
        presence_load_factors_path (str): Path to file containing presence load factors for one day.

    Returns:
        dict[str, np.ndarray]: The dictionary containing the time series data.
    """
    # Unpack parameters from the global parameters dictionary
    UHI_effect_day_C = global_parameters["UHI_effect_day_C"]  # UHI effect during the day, in °C
    UHI_effect_night_C = global_parameters["UHI_effect_night_C"]  # UHI effect during the night, in °C
    delta_T_winter_C = global_parameters["delta_T_winter_C"]  # Temperature boost during the winter, in °C
    delta_T_spring_C = global_parameters["delta_T_spring_C"]  # Temperature boost during the spring, in °C
    delta_T_summer_C = global_parameters["delta_T_summer_C"]  # Temperature boost during the summer, in °C
    delta_T_autumn_C = global_parameters["delta_T_autumn_C"]  # Temperature boost during the autumn, in °C
    delta_P_solar_summer = global_parameters[
        "delta_P_solar_summer"
    ]  # Relative increase factor for solar radiation during the summer
    delta_P_solar_RoY = global_parameters[
        "delta_P_solar_RoY"
    ]  # Relative increase factor for solar radiation during the rest of the year
    T_thresh_C = global_parameters["T_thresh_C"]  # Cooling threshold temperature, in °C

    # Add seasonal solar radiation boosts to time series DataFrame
    time_series_df = add_seasonal_solar_radiation_boosts(raw_weather_df, delta_P_solar_summer, delta_P_solar_RoY)

    # Add multidirectional solar radiation to time series DataFrame
    time_series_df = add_multidirectional_solar_radiation(
        time_series_df,
        multidirectional_solar_radiation_fractions_path,
    )

    # Add UHI time of day boosts to time series DataFrame
    time_series_df = add_UHI_effect(time_series_df, UHI_effect_day_C, UHI_effect_night_C)

    # Add seasonal temperature boosts to time series DataFrame
    time_series_df = add_seasonal_temperature_boosts(
        time_series_df,
        delta_T_winter_C,
        delta_T_spring_C,
        delta_T_summer_C,
        delta_T_autumn_C,
    )

    # Create a new column "T_outdoor_C" which sums the T_outdoor_raw_C, UHI_effect_C and delta_T_season_C columns
    time_series_df["T_outdoor_C"] = (
        time_series_df["T_outdoor_raw_C"] + time_series_df["UHI_effect_C"] + time_series_df["delta_T_season_C"]
    )

    # Add a column to the time series dictionary with the difference between the inside cooling threshold temperature and the outside air temperature in °C
    time_series_df["T_outdoor_minus_indoor_C"] = time_series_df["T_outdoor_C"] - T_thresh_C

    # Add presence load factors to time series DataFrame
    time_series_df = add_presence_load_factors(time_series_df, presence_load_factors_path)

    # Convert the time series DataFrame to a dictionary of numpy arrays
    return {column: time_series_df[column].to_numpy() for column in time_series_df.columns}


def read_dynamic_subsurface_temperature(
    subsurface_temperature_path: str,
    start_year: int,
    end_year: int,
    measurement_depth_cm: int = 100,
) -> np.ndarray:
    """Reads the dynamic subsurface temperature data from a csv file and converts it to a numpy array.

    Args:
        subsurface_temperature_path (str): The path to the csv file containing the dynamic subsurface temperature data.
        start_year (int): The starting year of the desired time series.
        end_year (int): The ending year of the desired time series.
        measurement_depth_cm (int, optional): The depth at which the subsurface temperature is measured, in cm. Defaults to 100.

    Returns:
        np.ndarray: The numpy array containing the dynamic subsurface temperature data at a depth of 100 cm, in °C.
    """
    # Measurement depth mapping
    depth_mapping = {5: 1, 10: 2, 20: 3, 50: 4, 100: 5}

    # Check if the provided measurement_depth is valid
    if measurement_depth_cm not in depth_mapping:
        msg = "Invalid measurement_depth. Choose from 5, 10, 20, 50, or 100 cm."
        raise ValueError(msg)

    # Read in the dynamic (in blocks of 6 hours) subsurface temperature data of De Bilt (Closest available weather station with ground temperature records)
    subsurface_temperature_df = pd.read_csv(subsurface_temperature_path, skiprows=16)

    # Strip all columns names from extraneous whitespace
    subsurface_temperature_df.columns = subsurface_temperature_df.columns.str.strip()

    # Read in string as datetime
    subsurface_temperature_df["date"] = pd.to_datetime(
        subsurface_temperature_df["YYYYMMDD"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )

    # Extract the data between the indicated start and end year. The full file contains data from 1981 to 2022.
    subsurface_temperature_df = subsurface_temperature_df[
        (subsurface_temperature_df["date"].dt.year >= start_year)
        & (subsurface_temperature_df["date"].dt.year <= end_year)
    ]

    # Remove leap days to prevent errors
    subsurface_temperature_df = subsurface_temperature_df[
        subsurface_temperature_df["date"].dt.strftime("%m-%d") != "02-29"
    ].reset_index(drop=True)

    # Repeat the subsurface temperature data to go from 6-hourly to hourly time-steps
    ground_temperature_series = (
        subsurface_temperature_df[f"TB{depth_mapping[measurement_depth_cm]}"].repeat(6).reset_index(drop=True)
    )

    # Convert the subsurface temperature data to numeric and divide by 10 to get °C
    ground_temperature_series = ground_temperature_series.astype("float") / 10

    # Convert the series to a numpy array for faster calculations
    return ground_temperature_series.to_numpy()
