"""
Minimal converter from a known TSV produced by numpy.savetxt to HDF5.

Assumptions (from the producer code):
- File is tab-separated with two numeric columns: "Time(us)" and "Signal".
- Header line is commented by numpy (starts with "# ").
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
    output_type = 'txt'

    def convert(self) -> pathlib.Path:
        output_path = pathlib.Path(self.temp_dir.name) / "converted.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("converted")
        return output_path

    # def convert(self) -> pathlib.Path:
    #     """Convert the input text file to HDF5 using the fixed two-column schema."""
    #     output_path = pathlib.Path(self.temp_dir.name) / f"{self.file_path.stem}.hdf5"

    #     # Parse header (last commented line) to recover column names if present
    #     header_line = ""
    #     try:
    #         with open(self.file_path, 'r', encoding='utf-8') as f:
    #             for line in f:
    #                 s = line.lstrip()
    #                 if s.startswith('#'):
    #                     header_line = s[1:].strip()
    #     except OSError:
    #         header_line = ""

    #     parsed_columns = []
    #     if header_line:
    #         if '\t' in header_line:
    #             parsed_columns = [c.strip() for c in header_line.split('\t') if c.strip()]
    #         else:
    #             parsed_columns = header_line.split()

    #     # Load data as DataFrame, skipping commented header lines
    #     try:
    #         df = pd.read_csv(self.file_path, sep='\t', comment='#', header=None)
    #     except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
    #         empty = xr.Dataset(
    #             attrs={
    #                 'source_file': self.file_path.name,
    #                 'conversion_time': datetime.now().isoformat(),
    #                 'converter': 'TxtToHdf5Converter',
    #                 'status': 'no_data'
    #             }
    #         )
    #         empty.to_netcdf(output_path, format='NETCDF4', engine='h5netcdf')
    #         return output_path

    #     # Assign column names
    #     num_cols = df.shape[1]
    #     if parsed_columns and len(parsed_columns) == num_cols:
    #         df.columns = parsed_columns
    #     else:
    #         df.columns = [f'variable_{i+1}' for i in range(num_cols)]

    #     # Convert to xarray with a friendly index name
    #     df.index.name = 'measurement_index'
    #     ds = df.to_xarray()
    #     ds = ds.assign_attrs(
    #         source_file=self.file_path.name,
    #         conversion_time=datetime.now().isoformat(),
    #         converter='TxtToHdf5Converter',
    #         columns='\t'.join(df.columns.astype(str))
    #     )

    #     ds.to_netcdf(output_path, format='NETCDF4', engine='h5netcdf')
    #     return output_path

