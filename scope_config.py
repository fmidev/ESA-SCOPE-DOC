# ESA-SCOPE project config

import os

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


DATADIR = os.path.expanduser('~/DATA/ESA-SCOPE/')
BICEP_DATADIR = os.path.expanduser('~/DATA/ESA-BICEP/')
# directory to (temporarily) store the downloaded files
#CACHE_DIR = '/var/tmp/ESA-SCOPE-cache'
#CACHE_DIR = '/usr/local/share/DATA/ESA-SCOPE-cache'
CACHE_DIR = '/Volumes/Samsung_T7/data/ESA-SCOPE'

MODEL_DIR = './models/'

USE_PML_SST = True  # Use PML provided monthly SST when available

# data urls, parameters 0 = year, 1 = month, 2 = day
urls = {
    'oc_daily': 'http://www.oceancolour.org/thredds/dodsC/CCI_ALL-v6.0-DAILY',
    'oc_5day': 'http://www.oceancolour.org/thredds/dodsC/CCI_ALL-v6.0-5DAY',
    'oc_8day': 'http://www.oceancolour.org/thredds/dodsC/CCI_ALL-v6.0-8DAY',
    'oc_monthly': 'http://www.oceancolour.org/thredds/dodsC/CCI_ALL-v6.0-MONTHLY',
    'pp': 'https://rsg.pml.ac.uk/shared_files/gku/SCOPE/PP_v6_9km/SCOPE_NCEO_PP_ESA-OC-L3S-MERGED-1M_MONTHLY_9km_mapped_{0}{1:02}-fv6.0.nc',
    'par': 'https://rsg.pml.ac.uk/shared_files/gku/SCOPE/PAR_NASA_9km/SCOPE_NCEO_PAR_NASA_MERGED-1M_MONTHLY_9km_mapped_{0}{1:02}.nc',
    'sst_day': 'https://dap.ceda.ac.uk/thredds/dodsC/neodc/eocis/data/global_and_regional/sea_surface_temperature/CDR_v3/Analysis/L4/v3.0.1/{0}/{1:02}/{2:02}/{0}{1:02}{2:02}120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR3.0-v02.0-fv01.0.nc',
    'sst_day_v2.1': 'https://dap.ceda.ac.uk/thredds/dodsC/neodc/esacci/sst/data/CDR_v2/Analysis/L4/v2.1/{0}/{1:02}/{2:02}/{0}{1:02}{2:02}120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.1-v02.0-fv01.0.nc',
    'sss_monthly': 'https://dap.ceda.ac.uk/thredds/dodsC/neodc/esacci/sea_surface_salinity/data/v04.41/GLOBALv4.41/30days/{0}/ESACCI-SEASURFACESALINITY-L4-SSS-GLOBAL-MERGED_OI_Monthly_CENTRED_15Day_0.25deg-{0}{1:02}01-fv4.41.nc',
    'currents_dir': 'https://tds0.ifremer.fr/thredds/dodsC/GLOBCURRENT-L4-CUREUL_15M-ALT_SUM_NRT-V03.0.html',
    'currents': 'https://tds0.ifremer.fr/thredds/dodsC//home/datawork-cersat-public/project/globcurrent/data/globcurrent/nrt/global_025_deg/total_15m/0-14-GLOBCURRENT-L4-CUReul_15m-ALT_SUM_NRT-v03.0-fv01.0.nc',
}

# Variables available for the model. Not all are used in the final model.
Rrs = ['Rrs_412', 'Rrs_443', 'Rrs_490', 'Rrs_510', 'Rrs_560', 'Rrs_665']
other = ['chlor_a']
xvariables = ['pp', 'salt', 'temp', 'lat', 'lon', 'time', 'dts', 'depth']
docvar = 'DOC'
allvars = [docvar] + Rrs + xvariables


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
