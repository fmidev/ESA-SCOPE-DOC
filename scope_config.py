# ESA-SCOPE project config

import os
from numpy import float32

# data parameters used in model training
shorelimit = 300  # km, use olu data this far from shore
doclimit = 100  # max doc value in input data
doclimit2 = 400
docmin = 40  # min doc value in input data
docmax = doclimit
latlimit = 70  # latitude limit
mindepth = 100  # minimum depth for data
maxsalt = 50 # max salinity value
minsalt = 30  # min salinity value

# defaults data type
DTYPE = float32

# directory for permanent data
DATADIR = os.path.expanduser('~/DATA/ESA-SCOPE/')
# directory to (temporarily) store the downloaded files
#CACHE_DIR = '/var/tmp/ESA-SCOPE-cache'
#CACHE_DIR = '/usr/local/share/DATA/ESA-SCOPE-cache'
CACHE_DIR = '/Volumes/Samsung_T7/data/ESA-SCOPE'

# directory to store model weights and data
MODEL_DIR = './models/'

# *Location of data files for SCOPE DOC model input*
# these are all monthly files
# The parameters are: 0 = year, 1 = month, 2 = day
datafiles = {
    # Monthly CCI ocean colour V6.0 data
    'OC': os.path.join(CACHE_DIR, 'OC_Rrs_{0}_{1:02}.nc'),
    # Monthly SST data
    'SST': os.path.join(CACHE_DIR, 'sst_monthly_mean_{0}_{1:02}.nc'),
    # Monthly salinity data
    'SSS': os.path.join(CACHE_DIR, 'salinity_clorys_{0}_{1:02}.nc'),
    # Monthly PP data from Gemma
    'PP': os.path.join(CACHE_DIR, 'SCOPE_NCEO_PP_ESA-OC-L3S-MERGED-1M_MONTHLY_9km_mapped_{0}{1:02}-fv6.0.nc'),
    # Monthly PAR data from Gemma
    'PAR': os.path.join(CACHE_DIR,'SCOPE_NCEO_PAR_NASA_MERGED-1M_MONTHLY_9km_mapped_{0}{1:02}.nc'),
    # distance to shore data
    'dts': os.path.join(DATADIR, 'dts_rot_1279l4_0.1x0.1.grb_v4_unpack.nc'),
    # Bathymetry data
    'bathy': os.path.join(DATADIR, 'bathy_9km_new_fill_val.nc'),
}

# URLs for the original data sources
# These are used to donwload the data if local data is not available
# data urls, parameters 0 = year, 1 = month, 2 = day
urls = {
    'oc_daily': 'http://www.oceancolour.org/thredds/dodsC/CCI_ALL-v6.0-DAILY',
    'oc_5day': 'http://www.oceancolour.org/thredds/dodsC/CCI_ALL-v6.0-5DAY',
    'oc_8day': 'http://www.oceancolour.org/thredds/dodsC/CCI_ALL-v6.0-8DAY',
    'oc_monthly': 'http://www.oceancolour.org/thredds/dodsC/CCI_ALL-v6.0-MONTHLY',
    'pp': 'https://rsg.pml.ac.uk/shared_files/gku/SCOPE/PP_v6_9km/SCOPE_NCEO_PP_ESA-OC-L3S-MERGED-1M_MONTHLY_9km_mapped_{0}{1:02}-fv6.0.nc',
    'par': 'https://rsg.pml.ac.uk/shared_files/gku/SCOPE/PAR_NASA_9km/SCOPE_NCEO_PAR_NASA_MERGED-1M_MONTHLY_9km_mapped_{0}{1:02}.nc',
    'sst_day': 'https://dap.ceda.ac.uk/thredds/dodsC/neodc/eocis/data/global_and_regional/sea_surface_temperature/CDR_v3/Analysis/L4/v3.0.1/{0}/{1:02}/{2:02}/{0}{1:02}{2:02}120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR3.0-v02.0-fv01.0.nc',
    'sss_monthly': 'https://dap.ceda.ac.uk/thredds/dodsC/neodc/esacci/sea_surface_salinity/data/v04.41/GLOBALv4.41/30days/{0}/ESACCI-SEASURFACESALINITY-L4-SSS-GLOBAL-MERGED_OI_Monthly_CENTRED_15Day_0.25deg-{0}{1:02}01-fv4.41.nc',
}

USE_PML_SST = False  # Use PML provided monthly SST when available
BICEP_DATADIR = os.path.expanduser('~/DATA/ESA-BICEPT/')

# Variables available for the model. Not all are used in the final model.
Rrs = ['Rrs_412', 'Rrs_443', 'Rrs_490', 'Rrs_510', 'Rrs_560', 'Rrs_665']
other = ['chlor_a']
xvariables = ['pp', 'salt', 'temp', 'lat', 'lon', 'time', 'dts', 'depth']
docvar = 'DOC'
allvars = [docvar] + Rrs + xvariables

# units for plots etc
units = {
    'DOC': 'µmol kg⁻¹',
    'pp': 'mgC m⁻² d⁻¹',
    'sqrtpp': 'mgC m⁻² d⁻¹',
    'par': 'mol m⁻² d⁻¹',
    'salt': 'PSS-78',
    'temp': '°C',
    'depth': 'm',
    'dts': 'km',
    'Rrs': 'sr⁻¹',
    'lat': 'degrees north',
    'lon': 'degrees east',
    'month': 'month number',
    }
