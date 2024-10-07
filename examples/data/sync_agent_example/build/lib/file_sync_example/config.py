import pathlib
import dataclasses

@dataclasses.dataclass
class FileSyncExampleConfig:
    data_storage_location: pathlib.Path
    set_up: str
    sample_name : str