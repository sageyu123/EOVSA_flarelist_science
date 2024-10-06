# ------------------------------------------------------------------------------
# 1. IMPORTS AND SETUP
# ------------------------------------------------------------------------------

import pandas as pd
import os
from astropy.time import Time
import numpy as np
import requests


# ------------------------------------------------------------------------------
# 2. UTILITY FUNCTIONS
# ------------------------------------------------------------------------------



def format_flare_id(flare_id):
    """Convert float Flare_ID to integer format without decimal and return it as a string."""
    return f'{flare_id:.0f}'


def generate_url(flare_id):
    """
    Generate the URL based on the flare_id.
    The URL follows the pattern:
    https://www.ovsa.njit.edu/SynopticImg/eovsamedia/eovsa-browser/yyyy/mm/dd/eovsa.lev1_mbd_12s.flare_id_yyyymmddhhmmss.mp4
    """
    flare_id_str = format_flare_id(flare_id)
    year = flare_id_str[:4]
    month = flare_id_str[4:6]
    day = flare_id_str[6:8]

    url = f"https://www.ovsa.njit.edu/SynopticImg/eovsamedia/eovsa-browser/{year}/{month}/{day}/eovsa.lev1_mbd_12s.flare_id_{flare_id_str}.mp4"

    return url


def download_file(url, destination_directory):
    """Download the file from the generated URL to the destination directory."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        file_name = url.split('/')[-1]
        file_path = os.path.join(destination_directory, file_name)

        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"Downloaded: {file_name}")
    else:
        print(f"Failed to download file from {url}. HTTP Status code: {response.status_code}")


# ------------------------------------------------------------------------------
# 3. DATA LOADING AND PRE-PROCESSING
# ------------------------------------------------------------------------------

# Set working directory (if needed)
os.chdir('/Users/fisher/myworkspace')

# Define data directory and load the flare data CSV file
data_directory = '/Users/fisher/Library/CloudStorage/Dropbox/PycharmProjects/EOVSA_flarelist_science/generate_flarelist'
df_flare = pd.read_csv(os.path.join(data_directory, 'EOVSA_STIX_joint_flarelist.csv'))

# Filter rows based on certain conditions
df_flare_filtered = df_flare.dropna(subset=['EO_xcen'])
df_flare_filtered.drop_duplicates(inplace=True)

# ------------------------------------------------------------------------------
# 4. GENERATE URL AND DOWNLOAD MP4 FILE
# ------------------------------------------------------------------------------

# Create a directory to store downloaded files
download_directory = '/Users/fisher/myworkspace/eovsa_flare_videos'
os.makedirs(download_directory, exist_ok=True)

# Loop through filtered flare data, generate URLs, and download the files
for flare_id in df_flare_filtered['Flare_ID']:
    flare_id_str = format_flare_id(flare_id)  # Convert flare ID to required format
    url = generate_url(flare_id)  # Generate URL for the flare
    print(f"Generated URL: {url}")  # Print the generated URL for verification
    download_file(url, download_directory)  # Download the file

