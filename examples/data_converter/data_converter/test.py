import sys
from pathlib import Path
import tempfile
import numpy as np
from data_converter.converter_txt_to_hdf5 import TxtToHdf5Converter


def main() -> None:
    # Generate a sample measurement.txt in a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        input_txt = tmp_dir_path / "measurement.txt"

        # Create data files (T1 measurement data)
        t1_times = np.linspace(0, 100, 50)  # microseconds
        t1_signal = np.exp(-t1_times / 25) + 0.02 * np.random.randn(50)  # T1 ≈ 25 us
        np.savetxt(
            str(input_txt),
            np.column_stack([t1_times, t1_signal]),
            header="Time(us)\tSignal",
            delimiter="\t",
        )

        converter = TxtToHdf5Converter(input_txt)
        with converter:
            output_path = converter.convert()
                        
            # Load and display the converted HDF5 file
            import xarray as xr
            ds = xr.open_dataset(output_path)
            print(ds)
            print(ds.attrs)


if __name__ == "__main__":
    main()