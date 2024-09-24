# ------------------------------------------------------------------------------
# 1. IMPORTS AND SETUP
# ------------------------------------------------------------------------------
'''
Note the flare class entry
20240507165400,2024-05-07,16:54:00,C?,2024-05-07 16:53:21,2024-05-07 16:55:38,2024-05-07 16:58:19,140.1,-269.5,eovsa.spec.flare_id_20240507165400.fits
was modifield because it contains a question mark. it was changed to C9
'''

import pandas as pd
import os
from astropy.time import Time
from astropy.coordinates import SkyCoord
from sunpy import map as smap
from sunpy.util import MetaDict
import numpy as np
import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap
from sunpy.coordinates import frames

# Define custom colormap
vals = np.linspace(0.2, 1, 256)
colors = cm.Blues(vals)
custom_cmap = LinearSegmentedColormap.from_list("BluesTruncated", colors)
colors2 = cm.Reds(vals)
custom_cmap2 = LinearSegmentedColormap.from_list("RedsTruncated", colors2)
colors3 = cm.RdYlBu(vals)
custom_cmap3 = cm.RdBu_r # LinearSegmentedColormap.from_list("RdYlBuTruncated", colors3)
# Get the current time
time_now = Time.now()


# ------------------------------------------------------------------------------
# 2. UTILITY FUNCTIONS
# ------------------------------------------------------------------------------

def flare_class_to_linear_scale(flare_class):
    """Convert flare class to a linear scale value."""
    flare_class_map = {
        'A': 0,
        'B': 1,
        'C': 2,
        'M': 3,
        'X': 4
    }

    if pd.isnull(flare_class):
        return None
    flare_letter = flare_class[0].upper()
    flare_number = float(flare_class[1:])

    return flare_number * (10 ** flare_class_map[flare_letter])


# ------------------------------------------------------------------------------
# 3. DATA LOADING AND PRE-PROCESSING
# ------------------------------------------------------------------------------

os.chdir('/Users/fisher/myworkspace')

data_directory = '/Users/fisher/Library/CloudStorage/Dropbox/PycharmProjects/EOVSA_flarelist/generate_flarelist'
df_flare = pd.read_csv(os.path.join(data_directory, 'EOVSA_STIX_joint_flarelist.csv'))

# Filter rows based on certain conditions
df_flare_filtered = df_flare.dropna(subset=['EO_xcen'])
df_flare_filtered.drop_duplicates(inplace=True)

# Replace 'GOES Class' with 'HEK_flare_class' where necessary

df_flare_filtered['GOES_class_time_of_flare'] = df_flare_filtered['GOES_class_time_of_flare'].fillna(
    df_flare_filtered['flare_class'])


## Filter rows where 'visible_from_earth' is False
# df_flare_filtered = df_flare_filtered[df_flare_filtered['visible_from_earth'] == False]
# Filter rows where 'hgs_lon' is in the range of 90 to 110 or -110 to -90 (occulted flares)
# df_flare_filtered = df_flare_filtered[
#     ((df_flare_filtered['hgs_lon'] >= 90) & (df_flare_filtered['hgs_lon'] <= 110)) |
#     ((df_flare_filtered['hgs_lon'] >= -110) & (df_flare_filtered['hgs_lon'] <= -90))
# ]



# df_flare_filtered.to_csv(os.path.join(data_directory, 'EOVSA_STIX_joint_flarelist_occulted.csv'), index=False)


# Convert flare classes to linear scale
df_flare_filtered['GOES_fl_scls'] = df_flare_filtered['GOES_class_time_of_flare'].apply(flare_class_to_linear_scale)


# Define a function to convert flare classes to numerical values
def flare_class_to_numeric(flare_class):
    class_map = {'A': 1, 'B': 2, 'C': 3, 'M': 4, 'X': 5}
    if flare_class and flare_class[0] in class_map:
        return class_map[flare_class[0]]
    return None


# Apply the function to the 'GOES Class' column to create a new column 'numeric_class'
df_flare_filtered['numeric_class'] = df_flare_filtered['GOES_class_time_of_flare'].map(flare_class_to_numeric)

df_flare_filtered = df_flare_filtered.dropna(subset=['numeric_class'])

# Filter rows where 'numeric_class' is 4 or above (M and X class flares)
# df_flare_filtered = df_flare_filtered[df_flare_filtered['numeric_class'] >= 4]


# Filter out rows with no numeric class
# filtered_numeric_classes = filtered_flares['numeric_class'].dropna()
filtered_numeric_classes = df_flare_filtered['numeric_class']

# Extract relevant columns for plotting
goes_class= df_flare_filtered['GOES_class_time_of_flare'].values



flare_peak_times = Time(list(df_flare_filtered['EO_tpeak'].values))
hpc_x, hpc_y = df_flare_filtered['EO_xcen'], df_flare_filtered['EO_ycen']
# hpc_x, hpc_y = df_flare_filtered['hpc_x_earth'], df_flare_filtered['hpc_y_earth']
hgs_lon, hgs_lat = df_flare_filtered['hgs_lon'], df_flare_filtered['hgs_lat']

# ------------------------------------------------------------------------------
# 4. PLOT CREATION
# ------------------------------------------------------------------------------

# Set up dummy solar map for plotting
dummy_data = np.ones((10, 10))
metadata_ref = MetaDict({
    'ctype1': 'HPLN-TAN', 'ctype2': 'HPLT-TAN',
    'cunit1': 'arcsec', 'cunit2': 'arcsec',
    'crpix1': (dummy_data.shape[0] + 1) / 2., 'crpix2': (dummy_data.shape[1] + 1) / 2.,
    'cdelt1': 1.0, 'cdelt2': 1.0, 'crval1': 0.0, 'crval2': 0.0,
    'hgln_obs': 0.0,  ## Stonyhurst heliographic longitude in degree
    'hglt_obs': 0.0,  ## Stonyhurst heliographic latitude in degree
    'dsun_obs': const.au.to(u.m).value, 'dsun_ref': const.au.to(u.m).value,
    'rsun_ref': const.R_sun.to(u.m).value,
    'rsun_obs': ((const.R_sun / const.au).decompose() * u.radian).to(u.arcsec).value,
    't_obs': time_now.iso, 'date-obs': time_now.iso,
})
dummy_map_ref = smap.GenericMap(dummy_data, metadata_ref)

# Create SkyCoord for flare locations based on available data
flare_coords = [
    SkyCoord(lon=lon * u.deg, lat=lat * u.deg, radius=const.R_sun, frame='heliographic_stonyhurst',
             obstime=dummy_map_ref.date).transform_to(dummy_map_ref.coordinate_frame)
    if pd.notnull(lon) and pd.notnull(lat) else
    SkyCoord(Tx=x * u.arcsec, Ty=y * u.arcsec, frame=dummy_map_ref.coordinate_frame)
    for obstime, lon, lat, x, y in zip(flare_peak_times, hgs_lon, hgs_lat, hpc_x, hpc_y)
]
flare_coords = SkyCoord(flare_coords)

# Create flare intensity to marker size mapping
flare_scale_function = lambda x: x / 1e2 * 8
marker_sizes = flare_scale_function(df_flare_filtered['GOES_fl_scls'].values)

# Set up color mapping for flare peak times
flare_times_mpl = flare_peak_times.plot_date
norm = Normalize(vmin=flare_times_mpl.min(), vmax=flare_times_mpl.max())
colors = custom_cmap(norm(flare_times_mpl))

# Plotting
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': dummy_map_ref})

# Overlay flare locations on the solar map
flare_coords_plt = flare_coords.transform_to(dummy_map_ref.coordinate_frame)
fl_hpc_x_arr = flare_coords_plt.Tx
fl_hpc_y_arr = flare_coords_plt.Ty
ax.scatter_coord(flare_coords_plt, c=colors, s=marker_sizes,
                 edgecolors='face', linewidths=0.5, alpha=0.5, label="Flare Locations", cmap=custom_cmap)

from matplotlib.widgets import RectangleSelector


def onselect(eclick, erelease):
    "Function to be called when the rectangular region is selected"

    # Convert the pixel coordinates of the selection to world coordinates
    world_coord1 = dummy_map_ref.pixel_to_world(eclick.xdata * u.pixel, eclick.ydata * u.pixel)
    world_coord2 = dummy_map_ref.pixel_to_world(erelease.xdata * u.pixel, erelease.ydata * u.pixel)

    # print(world_coord1,world_coord2)
    x1, y1 = world_coord1.Tx, world_coord1.Ty
    x2, y2 = world_coord2.Tx, world_coord2.Ty

    # Now x1, y1, x2, y2 are in arcsec (Quantity objects with unit of arcsec)
    print(f'{x1:.01f}",{x2:.01f}",{y1:.01f}",{y2:.01f}"')
    selected_indices = [i for i, coord in enumerate(flare_coords_plt)
                        if x1 < coord.Tx < x2 and y1 < coord.Ty < y2]

    print(selected_indices)
    if len(selected_indices)>0:
        # Print a separator
        print("======== Selection Event Start========")
        for sidx, selcidx in enumerate(selected_indices):
            print(f'{sidx} No. {selcidx} {flare_peak_times[selcidx]} {goes_class[selcidx]} x:{fl_hpc_x_arr[selcidx]:.1f}" y:{fl_hpc_y_arr[selcidx]:.1f}"')
        print("======== Selection Event End ========")

# Create the rectangle selector
rs = RectangleSelector(ax, onselect,
                       useblit=True,
                       button=[1],  # Use left mouse button to draw the rectangle
                       minspanx=5, minspany=5,  # Minimum size of the rectangle
                       spancoords='pixels',
                       interactive=True)

dummy_coord = SkyCoord(-5000 * u.arcsec, -5000 * u.arcsec, frame=dummy_map_ref.coordinate_frame)

GOESclass = {1: 'B', 2: 'C', 3: 'M', 4: 'X'}
for l in np.arange(1, 5):
    ax.scatter_coord(dummy_coord, c=[custom_cmap(0.9)],
                     # s=np.array([l]) ** 2 * 60,
                     s=flare_scale_function(10 ** np.array([l])),
                     edgecolors='face',
                     linewidths=0.5,
                     alpha=0.5, label=GOESclass[l])



# Create the rectangle selector
rs = RectangleSelector(ax, onselect,
                       useblit=True,
                       button=[1],  # Use left mouse button to draw the rectangle
                       minspanx=5, minspany=5,  # Minimum size of the rectangle
                       spancoords='pixels',
                       interactive=True)


dummy_map_ref.plot(alpha=0, extent=[-1200, 1200, -1200, 1200], title=False, axes=ax)
dummy_map_ref.draw_grid(axes=ax, grid_spacing=10 * u.deg, color='k', lw=0.5)
dummy_map_ref.draw_limb(axes=ax, color='k')

# Get legend handlers and labels
handlers, labels = ax.get_legend_handles_labels()

# Remove handler and label for 'Flare Locations'
handlers = [h for h, l in zip(handlers, labels) if l != 'Flare Locations']
labels = [l for l in labels if l != 'Flare Locations']

# Display the legend with modified handlers and labels
lgd = ax.legend(handles=handlers, labels=labels, loc='lower left', framealpha=0.15, facecolor='gray', edgecolor='none')
lgd.set_title('GOES class')
# ax.set_xlabel('Solar-X [arcsec]')
# ax.set_ylabel('Solar-y [arcsec]')
ax.grid(False)

axins = inset_axes(ax,
                   width="85%",  # width = 85% of parent_bbox width
                   height="2%",  # height = 2% of parent_bbox height
                   loc='upper center',
                   borderpad=2)

# # Add colorbar to the new axis
cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=custom_cmap), cax=axins,
                    orientation='horizontal')  # , pad=0.1, location='top')

date_format = mdates.DateFormatter('%Y %b')
cbar.ax.xaxis.set_major_formatter(date_format)

# Set tick locator to every four months
locator = mdates.MonthLocator(interval=6)
cbar.ax.xaxis.set_major_locator(locator)
# Move ticks and tick labels to the top
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')

# Save the plot
# fig.savefig('fig-eovsa_stix_flare_loc.pdf', dpi=300)

fig_stat, axs_stat = plt.subplots(ncols=1, nrows=4, figsize=(3.5, 6))
# Sample modified code for plotting flare classes with different colors

flare_class_colors = {
    # 1: custom_cmap(0.1),  # 'A' class
    2: custom_cmap3(0.2),  # 'B' class
    3: custom_cmap3(0.4),  # 'C' class
    4: custom_cmap3(0.6),  # 'M' class
    5: custom_cmap3(0.8)  # 'X' class
}

# Convert the provided lon and lat columns to SkyCoord objects for both vectors
solo_coords = SkyCoord(lon=df_flare_filtered['solo_position_lon'].values * u.deg,
                       lat=df_flare_filtered['solo_position_lat'].values * u.deg,
                       radius=df_flare_filtered['solo_position_AU_distance'].values * u.AU,
                       obstime=flare_peak_times,
                       frame='heliographic_stonyhurst')

earth_coords = SkyCoord(lon=df_flare_filtered['earth_position_lon'].values * u.deg,
                        lat=df_flare_filtered['earth_position_lat'].values * u.deg,
                        radius=df_flare_filtered['earth_position_AU_distance'].values * u.AU,
                        obstime=flare_peak_times,
                        frame='heliographic_stonyhurst')

# Compute the separation angle between the two vectors
angles = solo_coords.separation(earth_coords).deg

idx_excess_r = np.sqrt(hpc_x ** 2 + hpc_y ** 2) >= ((const.R_sun / const.au).decompose() * u.radian).to(u.arcsec).value
hpc_x[idx_excess_r] = hpc_x[idx_excess_r] * 0.96
hpc_y[idx_excess_r] = hpc_y[idx_excess_r] * 0.96
flare_coords2 = SkyCoord(hpc_x.values * u.arcsec, hpc_y.values * u.arcsec, obstime=flare_peak_times,
                         observer='earth', frame=frames.Helioprojective)
# angles_flare = solo_coords.separation(flare_coords).deg
angles_flare = (flare_coords2.transform_to(solo_coords).lon - solo_coords.lon).value
idx_wrap = angles_flare<-180
angles_flare[idx_wrap] = angles_flare[idx_wrap]+360
angles_flare[angles_flare>90] = 90
angles_flare[angles_flare<-90] = -90

df_flare_filtered['angles'] = angles
df_flare_filtered['angles_flare'] = angles_flare

# Separate data by flare class
angles_data = {
    2: df_flare_filtered[df_flare_filtered['numeric_class'] == 2]['angles'].values,
    3: df_flare_filtered[df_flare_filtered['numeric_class'] == 3]['angles'].values,
    4: df_flare_filtered[df_flare_filtered['numeric_class'] == 4]['angles'].values,
    5: df_flare_filtered[df_flare_filtered['numeric_class'] == 5]['angles'].values
}

angles_flare_data = {
    2: df_flare_filtered[df_flare_filtered['numeric_class'] == 2]['angles_flare'].values,
    3: df_flare_filtered[df_flare_filtered['numeric_class'] == 3]['angles_flare'].values,
    4: df_flare_filtered[df_flare_filtered['numeric_class'] == 4]['angles_flare'].values,
    5: df_flare_filtered[df_flare_filtered['numeric_class'] == 5]['angles_flare'].values
}

ax_angle = axs_stat[1]
# Plotting the histogram
# fig_angle, ax_angle = plt.subplots(figsize=(4, 2))
colors = flare_class_colors.values()
labels = ['B', 'C', 'M', 'X']
# ax_angle.hist(angles, bins=10, color=custom_cmap(0.9), edgecolor=custom_cmap(0.9), lw=0.5, alpha=0.9)
ax_angle.hist([angles_data[2], angles_data[3], angles_data[4], angles_data[5]], bins=12, stacked=True, color=colors,
              label=labels, lw=0.5, alpha=1.0)
ax_angle.set_xlabel(r'$SolO$-Earth Sep. [deg]')
ax_angle.set_ylabel('# of Flares')
# ax_angle.set_title('Histogram of Angles between SolO and Earth')
ax_angle.set_xticks(np.arange(0, 181, 45))
# ax_angle.legend(loc='best', framealpha=0.15, facecolor='gray', edgecolor='none', ncol=2, handlelength=1.5,
#                 handletextpad=0.3, columnspacing=1.0)
# fig_angle.tight_layout()
# fig_angle.savefig('fig-angle_histogram.pdf', dpi=300)

#
ax_angle_flare = axs_stat[2]
# Plotting the histogram
# fig_angle, ax_angle = plt.subplots(figsize=(4, 2))

# ax_angle_flare.hist(angles_flare, bins=10, color=custom_cmap(0.9), edgecolor=custom_cmap(0.9), lw=0.5, alpha=1.0)
ax_angle_flare.hist([angles_flare_data[2], angles_flare_data[3], angles_flare_data[4], angles_flare_data[5]], bins=12,
                    stacked=True, color=colors,
                    label=labels, lw=0.5, alpha=1.0)
ax_angle_flare.set_xlabel(r'Longtitude in $SolO$ view [deg]')
ax_angle_flare.set_ylabel('# of Flares')
ax_angle_flare.legend(loc='best', framealpha=0.15, facecolor='gray', edgecolor='none', ncol=2, handlelength=1.,
                      handletextpad=0.2, columnspacing=0.75)
ax_angle_flare.set_xticks(np.arange(-90, 91, 45))
# # fig_angle.tight_layout()
# # fig_angle.savefig('fig-angle_histogram.pdf', dpi=300)


# Plotting the histogram for flare classes
# fig_class, ax_class = plt.subplots(figsize=(4, 2))
ax_class = axs_stat[0]
# ax_class.hist(filtered_numeric_classes, bins=4, range=(1.5, 5.5), color=custom_cmap(0.9), edgecolor=custom_cmap(0.9),
#               lw=0.5, alpha=1.0)

# Instead of a single histogram, we loop through the unique flare classes and plot them individually
for flare_numeric_class, color in flare_class_colors.items():
    ax_class.hist(
        [value for value in filtered_numeric_classes if value == flare_numeric_class],
        bins=1,
        range=(flare_numeric_class - 0.5, flare_numeric_class + 0.5),
        color=color,
        edgecolor=color,
        lw=0.5, alpha=1.0
    )

ax_class.set_xlabel('Flare Class')
ax_class.set_ylabel('# of Flares')
# ax_class.set_title('Histogram of Flare Classes')
ax_class.set_xticks([2, 3, 4, 5])
ax_class.set_xticklabels(['B', 'C', 'M', 'X'])

# fig_class.tight_layout()
# fig_class.savefig('fig-flare_class_histogram.pdf', dpi=300)

# Extract the required columns for plotting
df_flare_filtered['peak_UTC'] = pd.to_datetime(df_flare_filtered['peak_UTC'])
# flux_values = filtered_flares['4-10 keV']


egs = ['4-10 keV', '25-50 keV']
# egs = ['4-10 keV', '50-84 keV']
ax_flux = axs_stat[3]
# Set the y-axis to a logarithmic scale and plot the data
# fig_flux, ax_flux = plt.subplots(figsize=(4, 2))
df_flare_filtered.plot(x='peak_UTC', y=egs, ax=ax_flux,
                     color=[custom_cmap(0.9), custom_cmap2(0.6)], alpha=0.2, lw=0.5, linestyle='', marker='o')
ax_flux.set_yscale('log')
ax_flux.set_ylabel('Counts')
# ax_flux.set_xlabel('Time [UT]')
ax_flux.set_xlabel('Date')
ax_flux.legend(loc='best', framealpha=0.8, facecolor='w', edgecolor='none')
# lgd_flux = ax_flux.get_legend()
# lgd_flux.set_alpha(0.15)
# lgd_flux.set_facecolor('gray')
# lgd_flux.set_edgecolor('none')
# loc='best', framealpha=0.15, facecolor='gray', edgecolor='none',
# ax_flux.set_title('Flux (4-10 keV) vs Time')
# ax_flux.grid(True, which="both", ls="--", c='0.7')
# fig_flux.tight_layout()
# fig_flux.savefig('fig-stix_counts.pdf', dpi=300)
fig_stat.tight_layout()
# fig_stat.savefig('fig-flare_statistics.pdf', dpi=300)
