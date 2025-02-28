#!/usr/bin/env python3

"""
Command line version of SCOPE open ocean DOC model.

See scope_config.py for location of the input data files used by "generate_monthly_data".

Basic usage:

./run_model.py --year 2020 --month 6 -o SCOPE_DOC_2020_06.nc

This generates global DOC estimates for one month in 1/24° resolution.

"""

import argparse
import xarray as xr

from scope_doc_model import get_torch_device, load_model, estimate_DOC
from datautils import generate_monthly_data, savenc

parser = argparse.ArgumentParser(description='SCOPE DOC open ocean model')
parser.add_argument('-i', '--input_file', type=str, default=None, help='Input NetCDF file')
parser.add_argument('-o', '--output_file', type=str, required=True, help='Output NetCDF file')
parser.add_argument('-m', '--model_dir', type=str, default='models', help='Model directory')
parser.add_argument('-v', '--model_version', type=str, default='v1.0', help='Model version to use')
parser.add_argument('--keep', action='store_true', help='keep input data in output file')
parser.add_argument('--coarsen', type=int, default=0, help='Coarsen the input data from default')
parser.add_argument('--complevel', type=int, default=0, help='Compression level for the output')
args, remaining_argv = parser.parse_known_args()
if args.input_file is None:
    parser.add_argument('--year', type=int, required=True, help='Year')
    parser.add_argument('--month', type=int, required=True, help='Month')
args = parser.parse_args()

device = get_torch_device()

data, model = load_model(args.model_version, modeldir=args.model_dir, device=device)

if args.input_file is not None:
    ds = xr.open_dataset(args.input_file)
else:
    ds = generate_monthly_data(args.year, args.month, coarsen=args.coarsen)

out = estimate_DOC(ds, model, data, keep_input=args.keep)

out.to_netcdf(args.output_file)
savenc(out, args.output_file, complevel=args.complevel)