import pandas as pd
import os
import astropy.units as u
from sunpy.coordinates import get_body_heliographic_stonyhurst, get_horizons_coord
from astropy.time import Time
from scipy.interpolate import interp1d


# Determine the path of the current script
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

os.chdir('/Users/fisher/myworkspace')
# Read the CSV files into DataFrames
## EOVSA flare list with xy coordinates (provided by Xingyao Chen)
df_eovsa_flare = pd.read_csv(os.path.join(script_dir, 'EOVSA_flare_list_from_wiki_xycen.csv'))
df_stix_flare = pd.read_csv(
    'https://raw.githubusercontent.com/hayesla/stix_flarelist_science/main/STIX_flarelist_w_locations_20210318_20240801_version1_pythom.csv')

df_stix_flare['peak_UTC'] = pd.to_datetime(df_stix_flare['peak_UTC'])
df_eovsa_flare['EO_tpeak'] = pd.to_datetime(df_eovsa_flare['EO_tpeak'])

# Sort both DataFrames by their peak time columns
df_stix_flare_sorted = df_stix_flare.sort_values('peak_UTC')
df_eovsa_flare_sorted = df_eovsa_flare.sort_values('EO_tpeak')

# Merge the two DataFrames based on nearest peak times within 10 minutes tolerance
common_flares = pd.merge_asof(df_stix_flare_sorted, df_eovsa_flare_sorted,
                              left_on='peak_UTC', right_on='EO_tpeak',
                              direction='nearest',
                              tolerance=pd.Timedelta('5 minutes'))

# Drop rows where EO_tpeak is missing (NaN)
common_flares_cleaned = common_flares.dropna(subset=['EO_tpeak'])

flare_timestamps_mjd = Time(common_flares_cleaned['EO_tpeak']).mjd

# Fetch the location of Earth for the given time range
flare_peak_times = Time(list(common_flares_cleaned['EO_tpeak'].values))
time_range = [flare_peak_times[0], flare_peak_times[-1]]
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
common_flares_cleaned.loc[:, 'earth_position_lon'] = lon_interpolator(flare_timestamps_mjd)
common_flares_cleaned.loc[:, 'earth_position_lat'] = lat_interpolator(flare_timestamps_mjd)
common_flares_cleaned.loc[:, 'earth_position_AU_distance'] = radius_interpolator(flare_timestamps_mjd)



# Save the merged DataFrame to a new CSV file
outfile = os.path.join(script_dir, 'EOVSA_STIX_joint_flarelist.csv')
common_flares_cleaned.to_csv(outfile, index=False)

print(f"Joint flare list saved to {outfile}")
