# ESA SCOPE WP2.4 DOC model

Code for dissolved organic carbon (DOC) model in ESA funded project SCOPE -
Satellite-based observations of Carbon in the Ocean: Pools, Fluxes and Exchanges (https://oceancarbon-scope.org).

## Model

Model is based on neural network and it is implemented using PyTorch package in Python.

 - In-situ DOC data is used as training data set.
 - The trained model is used to calculate monthly estimates in 1/24° spatial grid.
 
Features used in the model: 6 wavelengths of Ocean Colour reflectance (Rrs), sea surface temperature (SST), salinity, primary production (PP), latitude (lat), and distance to shore (dts).
The model uses 1D convolution for Rrs using 8 filters and kernel size 2, and dense layer with width 32 for other explanatory features.

## Data

### Ocean Colour

Ocean Colour reflectances (Rrs_nnn) from 
OC-CCI v6 https://dx.doi.org/10.5285/5011d22aae5a4671b0cbc7d05c56c4f0

```
Rrs = ['Rrs_412', 'Rrs_443', 'Rrs_490', 'Rrs_510', 'Rrs_560', 'Rrs_665']
```

The monthly OC dataset has dimensions lat: 4320, lon: 8640, time: 322, which means 1/24° global spatial resolution. Time spans as time: 1997-09-04 ... 2024-06-01.

Citable as: Sathyendranath, S.; Jackson, T.; Brockmann, C.; Brotas, V.; Calton, B.; Chuprin, A.; Clements, O.; Cipollini, P.; Danne, O.; Dingle, J.; Donlon, C.; Grant, M.; Groom, S.; Krasemann, H.; Lavender, S.; Mazeran, C.; Mélin, F.; Müller, D.; Steinmetz, F.; Valente, A.; Zühlke, M.; Feldman, G.; Franz, B.; Frouin, R.; Werdell, J.; Platt, T. (2023): ESA Ocean Colour Climate Change Initiative (Ocean_Colour_cci): Version 6.0, 4km resolution data. NERC EDS Centre for Environmental Data Analysis, 08 August 2023. https://dx.doi.org/10.5285/5011d22aae5a4671b0cbc7d05c56c4f0

###  Primary Production

From Gemma's repository at
https://rsg.pml.ac.uk/shared_files/gku/SCOPE/PP_v6_9km/

Monthly data 1998 - 2021, 1/12° (Lat: 2160, lon: 4320)

### SST

SST-CCI v3.0 https://dx.doi.org/10.5285/4a9654136a7148e39b7feb56f8bb02d2

This dataset provides daily-mean sea surface temperatures (SST), presented on global 0.05° latitude-longitude grid, spanning 1980 to present. This is a Level 4 product, with gaps between available daily observations filled by statistical means.

Citable as:  Good, S.A.; Embury, O. (2024): ESA Sea Surface Temperature Climate Change Initiative (SST_cci): Level 4 Analysis product, version 3.0. NERC EDS Centre for Environmental Data Analysis, 09 April 2024. doi:10.5285/4a9654136a7148e39b7feb56f8bb02d2. https://dx.doi.org/10.5285/4a9654136a7148e39b7feb56f8bb02d2

Available for 1998-2021. The daily data are used to calculate monthly averages that are used in the model. (Only partly now, data for 2010-2018 is from PLM/Bror.)

### Salinity 

The GLORYS12V1 product is the CMEMS global ocean eddy-resolving (1/12° horizontal resolution, 50 vertical levels) reanalysis covering the altimetry (1993 onward).

https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description

https://doi.org/10.48670/moi-00021

But salinity in GLORYS ends at 2021/06.


### In-situ

Th in-situ data is obtained from
Ocean Carbon and Acidification Data System (OCADS), GLobal Ocean Data Analysis Project Version 2.2023
https://www.ncei.noaa.gov/access/ocean-carbon-acidification-data-system/oceans/GLODAPv2_2023/

The data file used is:
https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0283442/GLODAPv2.2023_Merged_Master_File.csv (last accessed 2024-12-03)

Data is aggregated monthly and to 1/24° spatial lon-lat grid and filtered using selection criteria:

    0 < DOC ≤ 100, -70° ≤ LON ≤ 70°, pressure ≤ 30

The usable in-situ has 1929 rows × 18 columns. OC Rrs, PP, salinity, SST, dts, and depth are interporlated from monthly global data to the aggregated in-situ loctions using nearest neighbour method.

### Distance to shore and land-sea-mask

From ERA5, 0.1° resolution. File `dts_rot_1279l4_0.1x0.1.grb_v4_unpack.nc`.

### Bathymetry

File `bathy_9km_new_fill_val.nc` obtained from PML.

