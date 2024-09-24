import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from astropy.time import Time
from sunpy.net import Fido
from sunpy.net import attrs as a
import numpy as np
import astropy.units as u

# Define the URL to scrape
url = "http://www.ovsa.njit.edu/wiki/index.php/Recent_Flare_List_(2021-)"

# Fetch the content of the webpage
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Extracting all tables with the class "wikitable"
tables = soup.find_all("table", {"class": "wikitable"})

# Extract desired columns: "Date", "Time (UT)", "GOES Class", and "STIX Coverage"
all_data = []
for table in tables:
    rows = table.find_all("tr")
    for row in rows[1:]:  # Skip the header row
        cols = row.find_all("td")
        if len(cols) >= 4:  # Check if there are at least 4 columns to extract data from
            date = cols[0].get_text(strip=True)
            time_ut = cols[1].get_text(strip=True)
            goes_class = cols[2].get_text(strip=True)
            stix_coverage = cols[4].get_text(strip=True)
            all_data.append([date, time_ut, goes_class, stix_coverage])

# Create a DataFrame
df_extracted = pd.DataFrame(all_data, columns=["Date", "Time (UT)", "GOES Class", "STIX Coverage"])


# Function to cluster timestamps by day
def cluster_timestamps_by_day(timestamps):
    clusters = {}
    for timestamp in timestamps:
        date_str = timestamp.split("T")[0]
        if date_str not in clusters:
            clusters[date_str] = []
        clusters[date_str].append(timestamp)
    return clusters

# Function to cluster timestamps by month
def cluster_timestamps_by_month(timestamps):
    clusters = {}
    for timestamp in timestamps:
        date_obj = datetime.strptime(timestamp.split("T")[0], "%Y-%m-%d")
        year_month_str = date_obj.strftime("%Y-%m")
        if year_month_str not in clusters:
            clusters[year_month_str] = []
        clusters[year_month_str].append(timestamp)
    return clusters


def get_flare_info_for_cluster(timestamps):

    timestamps=Time(timestamps)
    if len(timestamps) < 2:
        tpeak = timestamps[0]

        tstart = tpeak - 30*u.min
        tend = tpeak + 30*u.min
    else:
        tstart = timestamps[0] - 30*u.min
        tend = timestamps[-1] + 30*u.min


    try:
        HEK_results = Fido.search(a.Time(tstart, tend),
                             # a.hek.EventType(event_type),
                             a.hek.FL.GOESCls > "A1.0",
                             a.hek.OBS.Observatory == "GOES")
    except:
        HEK_results = Fido.search(a.Time(tstart, tend),
                             a.hek.OBS.Observatory == "GOES")

    results = []
    for timestamp in timestamps:
        flare_class, HEK_flare_tstart, HEK_flare_tpeak, HEK_flare_tend, HEK_flare_hgc_x, HEK_flare_hgc_y, HEK_flare_hpc_x, HEK_flare_hpc_y = get_flare_info(
            timestamp,HEK_results)
        results.append((flare_class, HEK_flare_tstart, HEK_flare_tpeak, HEK_flare_tend,
                        HEK_flare_hgc_x, HEK_flare_hgc_y, HEK_flare_hpc_x, HEK_flare_hpc_y))
    return results


##=========================Step 2: get the start, peak, end time of the flare=========================
##ipython get_time_from_wiki.py
def get_flare_info(timestamp, HEK_results):
    ##tpeak_str="2019-04-15T19:30:00"
    # tpeak_str=tpeak_str
    tpeak = Time(timestamp)
    hek_results = HEK_results["hek"]

    flare_class = '??'
    HEK_flare_tstart = (tpeak - 0.05 * u.hour).iso
    HEK_flare_tpeak = tpeak.iso
    HEK_flare_tend = (tpeak + 0.05 * u.hour).iso
    HEK_flare_hgc_x = 0.0
    HEK_flare_hgc_y = 0.0
    HEK_flare_hpc_x = 0.0
    HEK_flare_hpc_y = 0.0


    if len(hek_results) == 0:
        print(f"no flare is found on {timestamp}")
        return flare_class, HEK_flare_tstart, HEK_flare_tpeak, HEK_flare_tend, HEK_flare_hgc_x, HEK_flare_hgc_y, HEK_flare_hpc_x, HEK_flare_hpc_y
    else:
        # filtered_results = hek_results["fl_goescls", "event_starttime", "event_peaktime",
        # "event_endtime", "ar_noaanum", "hgc_x", "hgc_y", "hpc_x", "hpc_y"]

        if isinstance(hek_results["event_peaktime"], Time):
            flare_tpeak = hek_results["event_peaktime"]
        else:
            nonzeroidx = hek_results["event_peaktime"].nonzero()
            if len(nonzeroidx[0])==0:
                print(f"no flare is found on {timestamp}")
                return flare_class, HEK_flare_tstart, HEK_flare_tpeak, HEK_flare_tend, HEK_flare_hgc_x, HEK_flare_hgc_y, HEK_flare_hpc_x, HEK_flare_hpc_y
            hek_results = hek_results[nonzeroidx]
            flare_tpeak = Time(hek_results["event_peaktime"].tolist())

        if len(flare_tpeak) == 1:
            ind = 0
        if len(flare_tpeak) > 1:
            ind = np.argmin(abs(flare_tpeak - tpeak))
        if len(flare_tpeak) < 1:
            print(f"no flare is found on {timestamp}")
            return flare_class, HEK_flare_tstart, HEK_flare_tpeak, HEK_flare_tend, HEK_flare_hgc_x, HEK_flare_hgc_y, HEK_flare_hpc_x, HEK_flare_hpc_y

        flare_class = (hek_results["fl_goescls"])[ind]
        HEK_flare_tstart = (hek_results["event_starttime"])[ind].iso
        HEK_flare_tpeak = (flare_tpeak)[ind].iso
        HEK_flare_tend = (hek_results["event_endtime"])[ind].iso
        HEK_flare_hgc_x = (hek_results["hgc_x"])[ind]
        HEK_flare_hgc_y = (hek_results["hgc_y"])[ind]
        HEK_flare_hpc_x = (hek_results["hpc_x"])[ind]
        HEK_flare_hpc_y = (hek_results["hpc_y"])[ind]

        print(f"Class {flare_class} HEK flare peaks on {HEK_flare_tpeak} / radio on {timestamp}")
        return flare_class, HEK_flare_tstart, HEK_flare_tpeak, HEK_flare_tend, HEK_flare_hgc_x, HEK_flare_hgc_y, HEK_flare_hpc_x, HEK_flare_hpc_y


# Updated logic to cluster timestamps by day and fetch flare info in batches
isot_times = [f"{row['Date']}T{row['Time (UT)']}:00" for _, row in df_extracted.iterrows()]
clusters = cluster_timestamps_by_day(isot_times)
# clusters = cluster_timestamps_by_month(isot_times)

# Lists to store the flare details
flare_ids = []
HEK_flare_classes = []
HEK_flare_tstarts = []
HEK_flare_tpeaks = []
HEK_flare_tends = []
HEK_flare_hgc_xs = []
HEK_flare_hgc_ys = []
HEK_flare_hpc_xs = []
HEK_flare_hpc_ys = []


# Fetching flare details for each cluster
for _, timestamps in clusters.items():
    flare_details = get_flare_info_for_cluster(timestamps)
    for details in flare_details:
        flare_class, HEK_flare_tstart, HEK_flare_tpeak, HEK_flare_tend, \
        HEK_flare_hgc_x, HEK_flare_hgc_y, HEK_flare_hpc_x, HEK_flare_hpc_y = details

        # Create Flare_ID based on the peak time
        flare_id = Time(HEK_flare_tpeak).datetime.strftime('%Y%m%d%H%M%S')

        # Append the details to the lists
        flare_ids.append(flare_id)
        HEK_flare_classes.append(flare_class)
        HEK_flare_tstarts.append(HEK_flare_tstart)
        HEK_flare_tpeaks.append(HEK_flare_tpeak)
        HEK_flare_tends.append(HEK_flare_tend)
        HEK_flare_hgc_xs.append(HEK_flare_hgc_x)
        HEK_flare_hgc_ys.append(HEK_flare_hgc_y)
        HEK_flare_hpc_xs.append(HEK_flare_hpc_x)
        HEK_flare_hpc_ys.append(HEK_flare_hpc_y)

# Create the new DataFrame with all the columns
df_flare_info = pd.DataFrame({
    "Flare_ID": flare_ids,
    "Date": df_extracted["Date"],
    "Time (UT)": df_extracted["Time (UT)"],
    "GOES Class": df_extracted["GOES Class"],
    "STIX Coverage": df_extracted["STIX Coverage"],
    "HEK_flare_class": HEK_flare_classes,
    "HEK_flare_tstart": HEK_flare_tstarts,
    "HEK_flare_tpeak": HEK_flare_tpeaks,
    "HEK_flare_tend": HEK_flare_tends,
    "HEK_flare_hgc_x": HEK_flare_hgc_xs,
    "HEK_flare_hgc_y": HEK_flare_hgc_ys,
    "HEK_flare_hpc_x": HEK_flare_hpc_xs,
    "HEK_flare_hpc_y": HEK_flare_hpc_ys
})

df_flare_info.head()  # Display the first few rows for verification

# Save the cleaned DataFrame to a CSV file in the same directory as the script
df_flare_info.to_csv("EOVSA_flarelist_wiki_hek_v20230913.csv", index=False)
