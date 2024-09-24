"""
===================================================
Obtaining a spacecraft trajectory from JPL Horizons
===================================================

This example shows how to obtain the trajectory of a spacecraft from JPL Horizons
and plot it relative to other bodies in the solar system.

JPL `Horizons <https://ssd.jpl.nasa.gov/horizons/>`__ can return the locations of
planets and minor bodies (e.g., asteroids) in the solar system, and it can also
return the location of a variety of major spacecraft.

You will need `astroquery <https://astroquery.readthedocs.io/>`__ installed.
"""

import pandas as pd
import os
from astropy.time import Time
from sunpy import map as smap
from sunpy.util import MetaDict
import numpy as np
import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.dates as mdates
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colorbar as mcolorbar
from sunpy.coordinates import get_body_heliographic_stonyhurst, get_horizons_coord
from sunpy.time import parse_time

# Get the current time
time_now = Time.now()

data_dir = './generate_flarelist'

# Read the CSV file to get the required columns
flare_data = pd.read_csv(os.path.join(data_dir, 'EOVSA_STIX_common_flares.csv'))
flare_data_filtered = flare_data[flare_data['STIX Coverage'].str.contains("yes", case=False, na=False)]
flare_data_filtered = flare_data_filtered[flare_data_filtered['HEK_flare_class'] != '??']
flare_data_filtered = flare_data_filtered.drop_duplicates()

# If 'HEK_flare_class' has data, replace 'GOES Class' with 'HEK_flare_class'
condition = flare_data_filtered['HEK_flare_class'].notnull() & flare_data_filtered['HEK_flare_class'].str.strip().ne('')
flare_data_filtered.loc[condition, 'GOES Class'] = flare_data_filtered.loc[condition, 'HEK_flare_class']

# Drop rows where HEK_flare_hpc_x is exactly 0 and hgs_lon has no data
condition_to_drop = (flare_data_filtered['HEK_flare_hpc_x'] == 0) & pd.isnull(flare_data_filtered['hgs_lon'])
flare_data_filtered = flare_data_filtered[~condition_to_drop]

flare_peak_time = flare_data_filtered['HEK_flare_tpeak']
flare_peak_times = Time(list(flare_peak_time.values))
flare_intensity = flare_data_filtered['GOES_flux_time_of_flare']
hgs_lon = flare_data_filtered['hgs_lon']
hgs_lat = flare_data_filtered['hgs_lat']

flare_data[['HEK_flare_tpeak', 'GOES_flux_time_of_flare', 'hgs_lon', 'hgs_lat']].head()


##############################################################################
# We use :func:`~sunpy.coordinates.get_horizons_coord` to query JPL Horizons
# for the trajectory of Parker Solar Probe (PSP).  Let's request 50 days on
# either side of PSP's 14th closest approach to the Sun.

# perihelion_14 = parse_time('2022-12-11 13:16')
trange = [flare_peak_times[0],flare_peak_times[-1]]
psp = get_horizons_coord('Parker Solar Probe',
                         {'start': trange[0],
                          'stop': trange[1],
                          'step': '180m'})

solo = get_horizons_coord('Solar Orbiter',
                         {'start': trange[0],
                          'stop': trange[1],
                          'step': '180m'})

##############################################################################
# We also obtain the location of Earth at PSP perihelion.  We could query
# JPL Horizons again, but :func:`~sunpy.coordinates.get_body_heliographic_stonyhurst` returns
# a comparably accurate location using the Astropy ephemeris.

earth = get_body_heliographic_stonyhurst('Earth', trange[0])

##############################################################################
# For the purposes of plotting on a Matplotlib polar plot, we create a short
# convenience function to extract the necessary values in the appropriate units.


def coord_to_polar(coord):
    return coord.lon.to_value('rad'), coord.radius.to_value('AU')

##############################################################################
# Finally, we plot the trajectory on a polar plot.  Be aware that the
# orientation of the Stonyhurst heliographic coordinate system rotates
# over time such that the Earth is always at zero longitude.
# Accordingly, when we directly plot the trajectory, it does not appear
# as a simple ellipse because each trajectory point has a different
# observation time and thus a different orientation of the coordinate
# system.  To see the elliptical orbit, the trajectory can be
# transformed to the coordinate frame of Earth at the single time of
# PSP perihelion (``earth``), so that the trajectory is represented in
# a non-rotating coordinate frame.


fig = plt.figure()
ax = fig.add_subplot(projection='polar')
ax.plot(0, 0, 'o', label='Sun', color='orange')
ax.plot(*coord_to_polar(earth), 'o', label='Earth', color='blue')
ax.plot(*coord_to_polar(psp),
        label='PSP (as seen from Earth)', color='purple')
ax.plot(*coord_to_polar(solo),
        label='SolO (as seen from Earth)', color='green')
# ax.plot(*coord_to_polar(psp.transform_to(earth)),
#         label='PSP (non-rotating frame)', color='purple', linestyle='dashed')
ax.set_title('Stonyhurst heliographic coordinates')
ax.legend(loc='best')

# plt.show()

##############################################################################
# There are other tools that enable a similar style of figure.
# `solarmach <https://github.com/jgieseler/solarmach#usage>`__ is one such example.
