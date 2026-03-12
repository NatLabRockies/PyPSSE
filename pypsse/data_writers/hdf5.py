# Standard libraries
# from common import dtype_MAPPING
import os
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from loguru import logger

from pypsse.common import DEFAULT_RESULTS_FILENAME


class HDF5Writer:
    """Class that handles writing simulation results to hdf5 files."""

    def __init__(self, log_dir: Path, column_length: int, filename_prefix:str=""):
        """Constructor for hdf5 writer

        Args:
            log_dir (Path): output path (dirctory)
            column_length (int): number of data columns
        """

        self.log_dir = log_dir
        filename = filename_prefix + "_"  + DEFAULT_RESULTS_FILENAME
        self.store = h5py.File(
            os.path.join(log_dir, filename), "w"
        )
        self.store_groups = {}
        self.store_datasets = {}
        self.row = {}
        self.column_length = column_length

        self.chunkRows = 1
        self.step = 0
        self.dfs = {}
        self.convergence = self.store.create_dataset(
            "Convergence",
            shape=(self.column_length,),
            maxshape=(None,),
            chunks=True,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            dtype=np.int16,
        )
        self.Timestamp = self.store.create_dataset(
            "Time stamp",
            shape=(self.column_length,),
            maxshape=(None,),
            chunks=True,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            dtype="S30",
        )
        # Create arrow writer for each object type

    def write(
        self, currenttime: datetime, powerflow_output: dict, convergence: int
    ):
        """Writes the status of assets at a particular timestep to a hdf5 file.

        Args:
            currenttime (datetime): simulator time step
            powerflow_output (dict): simulation results
            convergence (int): simulation convergence status
        """
        # Iterate through each object type
        for obj_type in powerflow_output:
            data = pd.DataFrame(powerflow_output[obj_type], index=[self.step])
            data = data.fillna(0)

            if obj_type not in self.row:
                self.row[obj_type] = 0
                self.store_groups[obj_type] = self.store.create_group(obj_type)
                self.store_datasets[obj_type] = {}
                for col_name in powerflow_output[obj_type].keys():
                    dtype = data[col_name].dtype
                    if dtype == object:
                        dtype = "S30"

                    self.store_datasets[obj_type][self.column_name(col_name)] = self.store_groups[
                        obj_type
                    ].create_dataset(
                        self.column_name(col_name),
                        shape=(self.column_length,),
                        maxshape=(None,),
                        chunks=True,
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                        dtype=dtype,
                    )

            if obj_type not in self.dfs:
                self.dfs[obj_type] = data
            else:
                if self.dfs[obj_type] is None:
                    self.dfs[obj_type] = data
                else:
                    self.dfs[obj_type] = self.dfs[obj_type].append(
                        data, ignore_index=True
                    )

            if self.step % self.chunkRows == self.chunkRows - 1:
                si = int(self.step / self.chunkRows) * self.chunkRows
                ei = si + self.chunkRows
                for col_name in powerflow_output[obj_type].keys():
                    r = self.store_datasets[obj_type][self.column_name(col_name)].shape[0]
                    if ei >= r:
                        self.store_datasets[obj_type][self.column_name(col_name)].resize((ei,))
                    self.store_datasets[obj_type][self.column_name(col_name)][si:ei] = self.dfs[
                        obj_type
                    ][col_name]
                self.dfs[obj_type] = None
            if self.step >= len(self.Timestamp):
                self.Timestamp.resize((len(self.Timestamp) + 1,))
                self.convergence.resize((len(self.convergence) + 1,))
            self.Timestamp[self.step - 1] = np.bytes_(currenttime.strftime("%Y-%m-%d %H:%M:%S.%f"))
            self.convergence[self.step - 1] = convergence
            # Add object status data to a DataFrame
            self.store.flush()
        self.step += 1

    def column_name(self, column_name:str)-> str:
        return str(column_name).replace(" ", "")

    def close_store(self):
        try:
            k = list(self.dfs.keys())[0]
            if self.dfs[k] is not None:
                length = len(self.dfs[k])
                if length > 0:
                    for obj_type in self.dfs.keys():
                        for col_name in self.dfs[k].columns:
                            self.store_datasets[obj_type][self.column_name(col_name)][
                                self.column_length - length :
                            ] = self.dfs[obj_type][col_name]

            self.store.flush()
            self.store.close()
        except Exception as e:
            logger.info(str(e))

    def __del__(self):
        self.close_store()
