#!/usr/bin/env python3

"""
Generate input data for SCOPE DOC model.

See scope_config.py for location of the input data files used by "generate_monthly_data".

Basic usage:

./run_model.py --year 2020 --month 6 -o SCOPE_DOC_2020_06.nc

This generates global input estimates for one month in 1/24° resolution.

"""

import argparse
import xarray as xr

from scope_doc_model import get_torch_device, load_model, estimate_DOC
from datautils import generate_monthly_data, savenc

parser = argparse.ArgumentParser(description='SCOPE DOC open ocean model input data generation')
parser.add_argument('--pp', type=str, default=None, help='PP data file')
parser.add_argument('--sst', type=str, default=None, help='SST data file')
parser.add_argument('--oc', type=str, default=None, help='OC data file')
parser.add_argument('--sss', type=str, default=None, help='SSS data file')
parser.add_argument('--dts', type=str, default=None, help='DTS data file')
parser.add_argument('-o', '--output_file', type=str, required=True, help='Output NetCDF file')
parser.add_argument('--coarsen', type=int, default=0, help='Coarsen the input data from default')
parser.add_argument('--complevel', type=int, default=0, help='Compression level for the output')
parser.add_argument('-m', '--model_dir', type=str, default='models', help='Model directory')
parser.add_argument('-v', '--model_version', type=str, default='v1.0', help='Model version to use')
parser.add_argument('--keep', action='store_true', help='keep input data in output file')
parser.add_argument('--dataonly', action='store_true', help='generate input data only')

#args, remaining_argv = parser.parse_known_args()
#if args.input_file is None:
parser.add_argument('--year', type=int, default=None, help='Year')
parser.add_argument('--month', type=int, default=None, help='Month')
args = parser.parse_args()


ds = generate_monthly_data(args.year, args.month, coarsen=args.coarsen,
                           ppfile=args.pp, sstfile=args.sst, ocfile=args.oc, sssfile=args.sss, dtsfile=args.dts)


if not args.dataonly:
    device = get_torch_device()
    data, model = load_model(args.model_version, modeldir=args.model_dir, device=device)
    out = estimate_DOC(ds, model, data, keep_input=args.keep)
else:
    out = ds

savenc(out, args.output_file, complevel=args.complevel)