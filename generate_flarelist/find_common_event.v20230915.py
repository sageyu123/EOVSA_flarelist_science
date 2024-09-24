import pandas as pd
import os
import astropy.units as u
from sunpy.coordinates import get_body_heliographic_stonyhurst, get_horizons_coord
from astropy.time import Time
from scipy.interpolate import interp1d

# Determine the path of the current script
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

# Read the CSV files into DataFrames
eovsa_flare_data = pd.read_csv(os.path.join(script_dir, 'EOVSA_flarelist_wiki_hek_v20230913.csv'))
stix_flare_data = pd.read_csv('https://raw.githubusercontent.com/hayesla/stix_flarelist_science/main/STIX_flarelist_w_locations_20210214_20230430_version1.csv')

# Convert the time columns to datetime format for easier manipulation and comparison
eovsa_flare_data['Date Time (UT)'] = pd.to_datetime(eovsa_flare_data['Date'] + ' ' + eovsa_flare_data['Time (UT)'])
stix_flare_data['peak_UTC'] = pd.to_datetime(stix_flare_data['peak_UTC'])

# Adjust the peak_UTC time based on light travel time
stix_flare_data['peak_UTC_ltc'] = stix_flare_data['peak_UTC'] + pd.to_timedelta(stix_flare_data['light_travel_time'], unit='s')

# Sort both dataframes by flare peak times
eovsa_sorted = eovsa_flare_data.sort_values(by='Date Time (UT)').reset_index(drop=True)
stix_sorted = stix_flare_data.sort_values(by='peak_UTC_ltc').reset_index(drop=True)

# Set datetime columns as indices for efficient merging
eovsa_sorted.set_index('Date Time (UT)', inplace=True)
stix_sorted.set_index('peak_UTC_ltc', inplace=True)

# Merge flares based on the nearest times within a 5-minute window
common_flares = pd.merge_asof(eovsa_sorted, stix_sorted,
                             left_index=True, right_index=True,
                             direction='nearest', tolerance=pd.Timedelta('10 minutes'))

# Reset index for merged DataFrame
common_flares.reset_index(inplace=True)

# Filter flares which have STIX coverage
filtered_flares = common_flares[common_flares['STIX Coverage'].str.contains("yes", case=False, na=False)]
flare_peak_times = Time(list(filtered_flares['HEK_flare_tpeak'].values))

# Fetch the location of Solar Orbiter for the given time range
time_range = [flare_peak_times[0], flare_peak_times[-1]]
solo_coord = get_horizons_coord('Solar Orbiter',
                                {'start': time_range[0],
                                         'stop': time_range[1],
                                         'step': '180m'})

# Extract Solar Orbiter's position data
solo_timestamps = solo_coord.obstime
solo_lon = solo_coord.lon.to(u.deg).value
solo_lat = solo_coord.lat.to(u.deg).value
solo_radius = solo_coord.radius.to(u.AU).value

# Create a DataFrame from the Solar Orbiter data
solo_df = pd.DataFrame({
    'timestamp': pd.to_datetime(solo_timestamps.isot),
    'lon': solo_lon,
    'lat': solo_lat,
    'radius': solo_radius
})

# Convert flare timestamps to Modified Julian Date (MJD) for interpolation
flare_timestamps_mjd = Time(filtered_flares['Date Time (UT)']).mjd

# Create interpolation functions for the Solar Orbiter data
lon_interpolator = interp1d(solo_timestamps.mjd, solo_df['lon'], fill_value="extrapolate")
lat_interpolator = interp1d(solo_timestamps.mjd, solo_df['lat'], fill_value="extrapolate")
radius_interpolator = interp1d(solo_timestamps.mjd, solo_df['radius'], fill_value="extrapolate")

# Use interpolators to generate values for the filtered flare timestamps
filtered_flares['interpolated_lon'] = lon_interpolator(flare_timestamps_mjd)
filtered_flares['interpolated_lat'] = lat_interpolator(flare_timestamps_mjd)
filtered_flares['interpolated_radius'] = radius_interpolator(flare_timestamps_mjd)

# Replace missing values in the filtered flares DataFrame using the interpolated values
filtered_flares['solo_position_lat'].fillna(filtered_flares['interpolated_lat'], inplace=True)
filtered_flares['solo_position_lon'].fillna(filtered_flares['interpolated_lon'], inplace=True)
filtered_flares['solo_position_AU_distance'].fillna(filtered_flares['interpolated_radius'], inplace=True)

# Drop temporary columns used for interpolation
columns_to_drop = ['interpolated_lon', 'interpolated_lat', 'interpolated_radius']
filtered_flares.drop(columns=columns_to_drop, inplace=True)

# Fetch the location of Earth for the given time range
earth_coord = get_body_heliographic_stonyhurst('Earth', time_range)

# Extract Earth's position data
earth_timestamps = earth_coord.obstime
earth_lon = earth_coord.lon.to(u.deg).value
earth_lat = earth_coord.lat.to(u.deg).value
earth_radius = earth_coord.radius.to(u.AU).value


# Create interpolation functions for the Solar Orbiter data
lon_interpolator = interp1d(earth_timestamps.mjd, earth_lon, fill_value="extrapolate")
lat_interpolator = interp1d(earth_timestamps.mjd, earth_lat, fill_value="extrapolate")
radius_interpolator = interp1d(earth_timestamps.mjd, earth_radius, fill_value="extrapolate")

# Use interpolators to generate values for the filtered flare timestamps
filtered_flares['earth_position_lon'] = lon_interpolator(flare_timestamps_mjd)
filtered_flares['earth_position_lat'] = lat_interpolator(flare_timestamps_mjd)
filtered_flares['earth_position_AU_distance'] = radius_interpolator(flare_timestamps_mjd)

# Save the updated DataFrame to a CSV file
filtered_flares.to_csv(os.path.join(script_dir, "EOVSA_STIX_common_flares.v20240930.csv"), index=False)
