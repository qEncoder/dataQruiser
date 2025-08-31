"""
Minimal converter from a txt to HDF5.
TXT File is expected to be tab-separated with two numeric columns.
"""

from etiket_client.sync.backends.filebase.converters.base import FileConverter
import pathlib
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime


class TxtToHdf5Converter(FileConverter):
    """Convert a simple T1 measurement text file into an HDF5 (NetCDF) dataset."""

    input_type = 'txt'
    output_type = 'hdf5'

    def convert(self) -> pathlib.Path:
        """Convert the input text file to HDF5 using the fixed two-column schema."""
        output_path = pathlib.Path(self.temp_dir.name) / f"{self.file_path.stem}.hdf5"

        # Load data as DataFrame with header
        df = pd.read_csv(self.file_path, sep='\t', index_col=0)

        # Convert to xarray Dataset
        ds = xr.Dataset.from_dataframe(df)
        ds = ds.assign_attrs(
            source_file=self.file_path.name,
            conversion_time=datetime.now().isoformat(),
            converter='TxtToHdf5Converter',
            columns='\t'.join(df.columns.astype(str))
        )
        ds = extract_units_from_labels(ds) # parse the units from the labels

        # save the xarray to a hdf5 file
        ds.to_netcdf(output_path, format='NETCDF4', engine='h5netcdf')
        return output_path
    



def extract_units_from_labels(ds: xr.Dataset) -> xr.Dataset:
    """Extract units from variable and coordinate labels in parentheses and add as attributes."""
    def extract_unit_info(name_str):
        """Extract clean name and unit from a name string."""
        if '(' in name_str and ')' in name_str:
            parts = name_str.split('(', 1)
            return parts[0].strip(), parts[1].split(')', 1)[0].strip()
        return name_str, ''
    
    renames = {}
    
    # Process both data variables and coordinates
    for name in list(ds.data_vars) + list(ds.coords):
        clean_name, unit = extract_unit_info(str(name))
        ds[name].attrs['units'] = unit
        if clean_name != str(name):
            renames[name] = clean_name
    
    return ds.rename(renames) if renames else ds