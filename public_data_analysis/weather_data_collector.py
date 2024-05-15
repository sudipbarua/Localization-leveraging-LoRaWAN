import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry
import datetime
import time

def get_weather_info_mod(ds_row):
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": ds_row["Latitude"],
        "longitude": ds_row["Longitude"],
        "elevation": 5.0,
        "start_hour": f"{ds_row['Weather param sense time']}",
        "end_hour": f"{ds_row['Weather param sense time']}",
        "hourly": ["temperature_2m", 
                "relative_humidity_2m",
                "dew_point_2m", 
                "apparent_temperature", 
                "precipitation", 
                "rain", 
                "snowfall", 
                "snow_depth", 
                "weather_code", 
                "pressure_msl", 
                "surface_pressure", 
                "cloud_cover", 
                "cloud_cover_low", 
                "cloud_cover_mid", 
                "cloud_cover_high", 
                "et0_fao_evapotranspiration", 
                "vapour_pressure_deficit", 
                "wind_speed_10m", 
                "wind_speed_100m", 
                "wind_direction_10m", 
                "wind_direction_100m", 
                "wind_gusts_10m", 
                "shortwave_radiation", 
                "direct_radiation",
                "diffuse_radiation", 
                "direct_normal_irradiance", 
                "global_tilted_irradiance", 
                "terrestrial_radiation", 
                "shortwave_radiation_instant",
                "direct_radiation_instant", 
                "diffuse_radiation_instant", 
                "direct_normal_irradiance_instant", 
                "global_tilted_irradiance_instant", 
                "terrestrial_radiation_instant"]
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_dew_point_2m = hourly.Variables(2).ValuesAsNumpy()
    hourly_apparent_temperature = hourly.Variables(3).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(4).ValuesAsNumpy()
    hourly_rain = hourly.Variables(5).ValuesAsNumpy()
    hourly_snowfall = hourly.Variables(6).ValuesAsNumpy()
    hourly_snow_depth = hourly.Variables(7).ValuesAsNumpy()
    hourly_weather_code = hourly.Variables(8).ValuesAsNumpy()
    hourly_pressure_msl = hourly.Variables(9).ValuesAsNumpy()
    hourly_surface_pressure = hourly.Variables(10).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(11).ValuesAsNumpy()
    hourly_cloud_cover_low = hourly.Variables(12).ValuesAsNumpy()
    hourly_cloud_cover_mid = hourly.Variables(13).ValuesAsNumpy()
    hourly_cloud_cover_high = hourly.Variables(14).ValuesAsNumpy()
    hourly_et0_fao_evapotranspiration = hourly.Variables(15).ValuesAsNumpy()
    hourly_vapour_pressure_deficit = hourly.Variables(16).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(17).ValuesAsNumpy()
    hourly_wind_speed_100m = hourly.Variables(18).ValuesAsNumpy()
    hourly_wind_direction_10m = hourly.Variables(19).ValuesAsNumpy()
    hourly_wind_direction_100m = hourly.Variables(20).ValuesAsNumpy()
    hourly_wind_gusts_10m = hourly.Variables(21).ValuesAsNumpy()
    hourly_shortwave_radiation = hourly.Variables(22).ValuesAsNumpy()
    hourly_direct_radiation = hourly.Variables(23).ValuesAsNumpy()
    hourly_diffuse_radiation = hourly.Variables(24).ValuesAsNumpy()
    hourly_direct_normal_irradiance = hourly.Variables(25).ValuesAsNumpy()
    hourly_global_tilted_irradiance = hourly.Variables(26).ValuesAsNumpy()
    hourly_terrestrial_radiation = hourly.Variables(27).ValuesAsNumpy()
    hourly_shortwave_radiation_instant = hourly.Variables(28).ValuesAsNumpy()
    hourly_direct_radiation_instant = hourly.Variables(29).ValuesAsNumpy()
    hourly_diffuse_radiation_instant = hourly.Variables(30).ValuesAsNumpy()
    hourly_direct_normal_irradiance_instant = hourly.Variables(31).ValuesAsNumpy()
    hourly_global_tilted_irradiance_instant = hourly.Variables(32).ValuesAsNumpy()
    hourly_terrestrial_radiation_instant = hourly.Variables(33).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["dew_point_2m"] = hourly_dew_point_2m
    hourly_data["apparent_temperature"] = hourly_apparent_temperature
    hourly_data["precipitation"] = hourly_precipitation
    hourly_data["rain"] = hourly_rain
    hourly_data["snowfall"] = hourly_snowfall
    hourly_data["snow_depth"] = hourly_snow_depth
    hourly_data["weather_code"] = hourly_weather_code
    hourly_data["pressure_msl"] = hourly_pressure_msl
    hourly_data["surface_pressure"] = hourly_surface_pressure
    hourly_data["cloud_cover"] = hourly_cloud_cover
    hourly_data["cloud_cover_low"] = hourly_cloud_cover_low
    hourly_data["cloud_cover_mid"] = hourly_cloud_cover_mid
    hourly_data["cloud_cover_high"] = hourly_cloud_cover_high
    hourly_data["et0_fao_evapotranspiration"] = hourly_et0_fao_evapotranspiration
    hourly_data["vapour_pressure_deficit"] = hourly_vapour_pressure_deficit
    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
    hourly_data["wind_speed_100m"] = hourly_wind_speed_100m
    hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
    hourly_data["wind_direction_100m"] = hourly_wind_direction_100m
    hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m
    hourly_data["shortwave_radiation"] = hourly_shortwave_radiation
    hourly_data["direct_radiation"] = hourly_direct_radiation
    hourly_data["diffuse_radiation"] = hourly_diffuse_radiation
    hourly_data["direct_normal_irradiance"] = hourly_direct_normal_irradiance
    hourly_data["global_tilted_irradiance"] = hourly_global_tilted_irradiance
    hourly_data["terrestrial_radiation"] = hourly_terrestrial_radiation
    hourly_data["shortwave_radiation_instant"] = hourly_shortwave_radiation_instant
    hourly_data["direct_radiation_instant"] = hourly_direct_radiation_instant
    hourly_data["diffuse_radiation_instant"] = hourly_diffuse_radiation_instant
    hourly_data["direct_normal_irradiance_instant"] = hourly_direct_normal_irradiance_instant
    hourly_data["global_tilted_irradiance_instant"] = hourly_global_tilted_irradiance_instant
    hourly_data["terrestrial_radiation_instant"] = hourly_terrestrial_radiation_instant

    hourly_dataframe = pd.DataFrame(data = hourly_data, index=[ds_row.name])  # Here we set index to the value of the original row index
    # this will help us concatinating the weather data to the main dataset
    # the index are only identifier of each sample and their corresponding weather data 
    return hourly_dataframe


ds = pd.read_csv("/root/db_stack/public_data_analysis/antwerp_ds_modified.csv")
# Convert the RX Time values to datetime format ISO8601 and store in a separate column upto minutes precision
# Expected format "2017-12-28T16:14"
ds["Weather param sense time"] = pd.to_datetime(ds["RX Time"], format='ISO8601').dt.strftime('%Y-%m-%dT%H:%M')

rdf = pd.DataFrame()

i = 0
while i < len(ds):

    try:
        rdf = pd.concat([rdf, get_weather_info_mod(ds.iloc[i, :])], axis=0)
        print(f'saving data{i}')
        rdf.to_csv('/root/db_stack/public_data_analysis/weather_data.csv')
        i += 1  
    except Exception as e:
        if 'Daily API request limit exceeded. Please try again tomorrow' in str(e):
            print(f'waiting untill {datetime.datetime.now() + datetime.timedelta(days=1)}')
            time.sleep(datetime.timedelta(days=1).total_seconds())

        elif 'Hourly API request limit exceeded. Please try again in the next hour' in str(e):
            print(f'waiting untill {datetime.datetime.now() + datetime.timedelta(hours=1)}')
            time.sleep(datetime.timedelta(hours=1).total_seconds())

        elif 'Minutely API request limit exceeded. Please try again in one minute' in str(e):
            print(f'waiting untill {datetime.datetime.now() + datetime.timedelta(minutes=1)}')
            time.sleep(datetime.timedelta(minutes=1).total_seconds())

        else:
            raise e