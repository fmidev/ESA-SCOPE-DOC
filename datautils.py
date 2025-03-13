"""ESA-SCOPE data utilities.

Utilities to load oceancolour and similar data.

Some data sets are loaded to directory CACHE_DIR on the first open and then accessed from there.


function names:
download_x, download from remote
open_remote_x, open remote data using xarray and save locally 
open_x, open locally, if exists, otherwise downloads and saves it to a local file

Files are saved to CACHE_DIR, some smaller datasets are in DATADIR

This clears copernicus marine data cache, in case it complains
copernicusmarine describe --overwrite-metadata-cache

"""

import os
# from zipfile import ZipFile
from urllib import request
import calendar

import numpy as np
import xarray as xr
# import pandas as pd

# import cdsapi
import copernicusmarine

from scope_config import urls, Rrs, DATADIR, BICEP_DATADIR, CACHE_DIR
from scope_config import USE_PML_SST
from scope_config import datafiles, DTYPE

xr.set_options(keep_attrs=True)

try:
    os.makedirs(CACHE_DIR, exist_ok=True)
except Exception as e:
    print(f'Warning: can not create {CACHE_DIR}')
    print(e)


def open_clorys(variables, year, month=None, day=None, load=False):
    """Opens Clorys dataset from Copernicus Marine"""
    #prod_id = 'GLOBAL_MULTIYEAR_PHY_001_030'
    if day is None:
        data_id = 'cmems_mod_glo_phy_my_0.083deg_P1M-m'
        if month is None:
            start_datetime = f"{year}-01-01"
            end_datetime = f"{year}-12-31"
        else:
            lastday = calendar.monthrange(year, month)[1]
            start_datetime = f"{year}-{month:02}-01"
            end_datetime = f"{year}-{month:02}-{lastday:02}"
    else:
        data_id = 'cmems_mod_glo_phy_myint_0.083deg_P1D-m'
        start_datetime = f"{year}-{month:02}-{day:02}"
        end_datetime = f"{year}-{month:02}-{day:02}"

    okvariables = ["bottomT", "mlotst", "siconc", "sithick", "so", "thetao", "uo", "usi", "vo", "vsi", "zos"]
  
    #if not np.isin(variable, okvariables):
    #    print(f'{variable} not in {okvariables}')
    #    return None
  
    #vars = ['northward_sea_water_velocity',  'eastward_sea_water_velocity']
    #vars = ['sea_water_salinity']

    ds = copernicusmarine.open_dataset(
        dataset_id = data_id,
        start_datetime = start_datetime,
        end_datetime = end_datetime,
        variables = variables,
    )
    if load:
        ds.load()
    return ds


def open_salinity(year, month, day=None, load=False):
    local_file = datafiles['SSS'].format(year, month)
    if not os.path.exists(local_file):
        ds = open_clorys(['so'], year=year, month=month, day=day, load=load)
        ds = ds['so'].isel(depth=0).rename({'latitude': 'lat', 'longitude': 'lon'})
        ds.to_netcdf(local_file)
    else:
        ds = xr.open_dataset(local_file)
        if "depth" in ds.dims:  # original data
            ds = ds['so'].isel(depth=0).rename({'latitude': 'lat', 'longitude': 'lon'})
        else:
            ds = ds['so']
    if load:
        ds.load()
    return ds.isel(time=0)


# monthly oc data
def open_oc(year, month, vars=Rrs, load=True):
    """OC data loader."""
    url = urls['oc_monthly']
    local_file = datafiles['OC'].format(year, month)
    if not os.path.exists(local_file):
        time = f'{year}-{month:02d}-01'
        ds = xr.open_dataset(url)
        ds = ds.sel(time=time)[vars]
        ds.to_netcdf(local_file)
    else:  # load all variables from local file
        ds = xr.open_dataset(local_file)[vars]
    if load:
        ds.load()
    return ds


def download_ocdata(year, months=np.arange(1, 13), vars=Rrs):
    """OC data loader."""
    url = urls['oc_monthly']

    ds = xr.open_dataset(url)

    for month in months:
        print(f'Loading month {month:02d}')
        local_file = datafiles['OC'].format(year, month)
        if not os.path.exists(local_file):
            time = f'{year}-{month:02d}-01'
            dsi = ds.sel(time=time)[vars]
            dsi.to_netcdf(local_file)
        else:
            print('  ...already loaded')
    return


def open_sst_daily(year, month=None, day=None, load=False):
    if day is None:
        # monthly only until 2016 (not really)
        #url = urls['sst_monthly'].format(year, month, 15)
        raise ValueError("Give a day, too")
    else:
        url = urls['sst_day'].format(year, month, day)
    ds = xr.open_dataset(url)
    ds = ds.rename({'analysed_sst': 'sst'})['sst'] - 273.15
    ds.attrs['units'] = '°C'
    if load:
        ds.load()
    return ds


# 2010-2018
def open_pmlsst(year, month, load=False):
    """Load PML SST data."""
    file = os.path.expanduser(
        os.path.join(
        BICEP_DATADIR, f'SST/{year}{month:02d}-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB_REP_UPSCALE9KM-v02.0-fv02.0.nc')
        )
    da = xr.open_dataset(file).sst.isel(time=0)
    da['time'] = da['time'] + np.timedelta64(14, 'D')
    da = da.expand_dims('time')
    da = da - 273.15
    da.attrs['units'] = '°C'
    if load:
        da.load()
    da = da.isel(time=0).astype(DTYPE)
    return da


def open_dts():
    """Read distance to shore data set."""
    local_file = datafiles['dts']
    dts = xr.open_dataset(local_file)
    dts = dts.rename({'longitude': 'lon', 'latitude': 'lat'})
    dts = dts.layer.rename('dts') / 1000  # to km
    return dts


def open_lsm(d=200):
    """Load land sea mask"""
    dts = open_dts()
    lsm = dts < d
    return lsm


def open_bathymetry(load=True):
    local_file = datafiles['bathy']
    b = xr.open_dataset(local_file)
    b = b['bathymetry']
    if load:
        b.load()
    return b


# Phytoplankton Primary Production
def open_pp(year, month, load=False):
    ppdir = CACHE_DIR
    os.makedirs(ppdir, exist_ok=True)

    remote_file = urls['pp'].format(year, month)
    local_file = datafiles['PP'].format(year, month)
    if not os.path.exists(local_file):
        request.urlretrieve(remote_file, local_file)

    da = xr.open_dataset(local_file)['pp']
    if load:
        da.load()
    return da.isel(time=0) # .astype(DTYPE)


# Daily Mean Photosynthetically Available Radiation
def open_par(year, month, load=False):
    pardir = CACHE_DIR
    os.makedirs(pardir, exist_ok=True)

    remote_file = urls['par'].format(year, month)
    local_file = datafiles['PAR'].format(year, month)

    if not os.path.exists(local_file):
        request.urlretrieve(remote_file, local_file)

    da = xr.open_dataset(local_file)['par']
    da = da.expand_dims({'time': [np.datetime64(f'{year}-{month:02d}-15', 'ns')]})
    if load:
        da.load()
    return da.isel(time=0)


def open_sst_monthly(year, month, load=False, use_plm_sst=USE_PML_SST):
    """Global monthly mean SST."""

    if use_plm_sst & (year >= 2010) & (year <= 2018):
        return open_pmlsst(year, month, load=load)
    

    local_file = datafiles['SST'].format(year, month)
    if os.path.exists(local_file):
        ds = xr.open_dataset(local_file).isel(time=0)  # remove time here
        vars = ds.data_vars
        if 'analysed_sst' in vars:
            ds = ds.rename({'analysed_sst': 'sst'})
        ds = ds['sst']
        if ds.max() > 100:
            ds = ds - 273.15
        if load:
            ds.load()
        return ds

    # else load each day and update mean    
    time = [np.datetime64(f'{year}-{month:02d}-15', 'ns')]
    numdays = calendar.monthrange(year, month)[1]
    print(f'month {month}')
    print(f'day 1')
    ds = open_sst_daily(year, month, 1)
    ds.coords['time'] = time
    for day in range(2, numdays+1):
        print(f'day {day}')
        dsi = open_sst_daily(year, month, day)
        dsi.coords['time'] = time
        ds = ds + dsi
    ds = ds / numdays
    
    if load:
        ds.load()
    ds.to_netcdf(local_file)
    ds = ds.isel(time=0)  # remove time dimension
    return ds


# point interpolation from dataset to dataframe (obs inplace now!, not anymore)
def interp(df, ds, coords=['time', 'lat', 'lon'], vars=None, method='nearest'):
    if vars is None:
        vars = list(ds.data_vars)
    out = df.copy()
    out[vars] = ds[vars].interp(out[coords].to_xarray(), method=method).to_dataframe()[vars]
    return out


def fixlon(ds):
    ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180
    ds = ds.sortby(ds.lon)
    return ds


# save with compression
def savenc(ds, file, complevel=5):
    encoding = {}
    encoding_keys = ("_FillValue", "dtype", "scale_factor", "add_offset", "grid_mapping")
    for data_var in ds.data_vars:
        encoding[data_var] = {key: value for key, value in ds[data_var].encoding.items() if key in encoding_keys}
        encoding[data_var].update(zlib=True, complevel=complevel)
    if complevel == 0:
        ds.to_netcdf(file)
    else:
        ds.to_netcdf(file, encoding=encoding)


def generate_monthly_data(year, month, Rrs=Rrs, coarsen=0, load=True):
    """Generate monthly dataset to be used as input to the DOC model."""

    # rename = {'so': 'salt', 'sst': 'temp', 'bathymetry': 'depth'}
    rename = {'so': 'salt', 'sst': 'temp'}
    
    dts = open_dts().load()
    # depths = open_bathymetry().load()
    oc = open_oc(year, month, vars=Rrs, load=load)
    npp = open_pp(year, month, load=load).astype(DTYPE)
    if coarsen > 0:
        npp = npp.coarsen(lon=coarsen, lat=coarsen).mean()
    #par = open_par(year, month)
    sss = open_salinity(year, month, load=load).astype(DTYPE)
    # PML version of SST from Bror
    #sst = (pmlsst(year, month) - 273.15).isel(time=0)
    sst = open_sst_monthly(year, month, load=load)

    # scale all to same as NPP (1/12°)
    sss = sss.reindex_like(npp, method='nearest')
    oc = oc.reindex_like(npp, method='nearest')
    #par = par.reindex_like(npp, method='nearest')
    sst = sst.reindex_like(npp, method='nearest')
    dts = dts.reindex_like(npp, method='nearest')
    # depth = depths.reindex_like(npp, method='nearest')

    # combine all
    out = sss.to_dataset().merge(npp)
    # out = out.merge(par2, compat='override')
    out = out.merge(oc, compat='override')
    out = out.merge(sst, compat='override')
    out = out.merge(dts, compat='override')
    # out = out.merge(depth, compat='override')
    
    # rename some variables
    out = out.drop_vars(list(rename.values()) , errors='ignore')
    out = out.rename(rename)
    
    # set time to day 15.
    out.coords['time'] = np.datetime64(f'{year}-{month:02}-15', 'ns')
    # fix some attributes
    out['dts'].attrs['long_name'] = 'distance to shore'
    out['dts'].attrs['units'] = 'km'
    # out['depth'].attrs['long_name'] = 'water depth'
    # out['depth'].attrs['units'] = 'm'

    return out
