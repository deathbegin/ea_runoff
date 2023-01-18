"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G., "Benchmarking
a Catchment-Aware Long Short-Term Memory Network (LSTM) for Large-Scale Hydrological Modeling".
submitted to Hydrol. Earth Syst. Sci. Discussions (2019)

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import sqlite3
from pathlib import Path, PosixPath
from typing import List, Tuple
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from numba import njit

# from main import GLOBAL_SETTINGS

# CAMELS catchment characteristics ignored in this study
INVALID_ATTR = [
    'gauge_name', 'area_geospa_fabric', 'geol_1st_class', 'glim_1st_class_frac', 'geol_2nd_class',
    'glim_2nd_class_frac', 'dom_land_cover_frac', 'dom_land_cover', 'high_prec_timing',
    'low_prec_timing', 'huc', 'q_mean', 'runoff_ratio', 'stream_elas', 'slope_fdc',
    'baseflow_index', 'hfd_mean', 'q5', 'q95', 'high_q_freq', 'high_q_dur', 'low_q_freq',
    'low_q_dur', 'zero_q_freq', 'geol_porostiy', 'root_depth_50', 'root_depth_99', 'organic_frac',
    'water_frac', 'other_frac'
]

# Maurer mean/std calculated over all basins in period 01.10.1999 until 30.09.2008
SCALER = {
    # 'input_means': np.array([3.17563234, 372.01003929, 17.31934062, 3.97393362, 924.98004197]),
    # 'input_stds': np.array([6.94344737, 131.63560881, 10.86689718, 10.3940032, 629.44576432]),
    # 'output_mean': np.array([1.49996196]),
    # 'output_std': np.array([3.62443672])
    'input_means': np.array([]),
    'input_stds': np.array([]),
    'output_mean': np.array([]),
    'output_std': np.array([])
}


def add_camels_attributes(camels_root: PosixPath, db_path: str = None):
    """Load catchment characteristics from txt files and store them in a sqlite3 table
    
    Parameters
    ----------
    camels_root : PosixPath
        Path to the main directory of the CAMELS data set
    db_path : str, optional
        Path to where the database file should be saved. If None, stores the database in the 
        `data` directory in the main folder of this repository., by default None
    
    Raises
    ------
    RuntimeError
        If CAMELS attributes folder could not be found.
    """
    attributes_path = Path(camels_root) / 'camels_attributes_v2.0'

    if not attributes_path.exists():
        raise RuntimeError(f"Attribute folder not found at {attributes_path}")

    txt_files = attributes_path.glob('camels_*.txt')

    # Read-in attributes into one big dataframe
    df = None
    for f in txt_files:
        df_temp = pd.read_csv(f, sep=';', header=0, dtype={'gauge_id': str})
        df_temp = df_temp.set_index('gauge_id')

        if df is None:
            df = df_temp.copy()
        else:
            df = pd.concat([df, df_temp], axis=1)

    # convert huc column to double digit strings
    df['huc'] = df['huc_02'].apply(lambda x: str(x).zfill(2))
    df = df.drop('huc_02', axis=1)

    if db_path is None:
        db_path = str(Path(__file__).absolute().parent.parent / 'data' / 'attributes.db')

    with sqlite3.connect(db_path) as conn:
        # insert into databse
        df.to_sql('basin_attributes', conn)

    print(f"Sucessfully stored basin attributes in {db_path}.")


def load_attributes(db_path: str,
                    basins: List,
                    drop_lat_lon: bool = True,
                    keep_features: List = None) -> pd.DataFrame:
    """Load attributes from database file into DataFrame

    Parameters
    ----------
    db_path : str
        Path to sqlite3 database file
    basins : List
        List containing the 8-digit USGS gauge id
    drop_lat_lon : bool
        If True, drops latitude and longitude column from final data frame, by default True
    keep_features : List
        If a list is passed, a pd.DataFrame containing these features will be returned. By default,
        returns a pd.DataFrame containing the features used for training.

    Returns
    -------
    pd.DataFrame
        Attributes in a pandas DataFrame. Index is USGS gauge id. Latitude and Longitude are
        transformed to x, y, z on a unit sphere.
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM 'basin_attributes'", conn, index_col='gauge_id')

    # drop rows of basins not contained in data set
    drop_basins = [b for b in df.index if b not in basins]
    df = df.drop(drop_basins, axis=0)

    # drop lat/lon col
    if drop_lat_lon:
        df = df.drop(['gauge_lat', 'gauge_lon'], axis=1)

    # drop invalid attributes
    if keep_features is not None:
        drop_names = [c for c in df.columns if c not in keep_features]
    else:
        drop_names = [c for c in df.columns if c in INVALID_ATTR]

    df = df.drop(drop_names, axis=1)

    return df


def normalize_features(feature: np.ndarray, variable: str) -> np.ndarray:
    """Normalize features using global pre-computed statistics.

    Parameters
    ----------
    feature : np.ndarray
        Data to normalize
    variable : str
        One of ['inputs', 'output'], where `inputs` mean, that the `feature` input are the model
        inputs (meteorological forcing data) and `output` that the `feature` input are discharge
        values.

    Returns
    -------
    np.ndarray
        Normalized features

    Raises
    ------
    RuntimeError
        If `variable` is neither 'inputs' nor 'output'
    """

    if variable == 'inputs':
        feature = (feature - SCALER["input_means"]) / SCALER["input_stds"]
    elif variable == 'output':
        feature = (feature - SCALER["output_mean"]) / SCALER["output_std"]
    else:
        raise RuntimeError(f"Unknown variable type {variable}")

    return feature


def rescale_features(feature: np.ndarray, variable: str) -> np.ndarray:
    """Rescale features using global pre-computed statistics.

    Parameters
    ----------
    feature : np.ndarray
        Data to rescale
    variable : str
        One of ['inputs', 'output'], where `inputs` mean, that the `feature` input are the model
        inputs (meteorological forcing data) and `output` that the `feature` input are discharge
        values.

    Returns
    -------
    np.ndarray
        Rescaled features

    Raises
    ------
    RuntimeError
        If `variable` is neither 'inputs' nor 'output'
    """
    if variable == 'inputs':
        feature = feature * SCALER["input_stds"] + SCALER["input_means"]
        # print(SCALER["input_stds"], SCALER["input_means"])
    elif variable == 'output':
        feature = feature * SCALER["output_std"] + SCALER["output_mean"]
        # print(SCALER["output_std"], SCALER["output_mean"])
    else:
        raise RuntimeError(f"Unknown variable type {variable}")

    return feature


@njit
def reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape data into LSTM many-to-one input samples

    Parameters
    ----------
    x : np.ndarray
        Input features of shape [num_samples, num_features]
    y : np.ndarray
        Output feature of shape [num_samples, 1]
    seq_length : int
        Length of the requested input sequences.

    Returns
    -------
    x_new: np.ndarray
        Reshaped input features of shape [num_samples*, seq_length, num_features], where 
        num_samples* is equal to num_samples - seq_length + 1, due to the need of a warm start at
        the beginning
    y_new: np.ndarray
        The target value for each sample in x_new
    """
    num_samples, num_features = x.shape

    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    y_new = np.zeros((num_samples - seq_length + 1, 1))

    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i:i + seq_length, :]
        y_new[i, :] = y[i + seq_length - 1, 0]

    return x_new, y_new


def load_forcing(camels_root: PosixPath, basin: str, hdtype='daymet') -> Tuple[pd.DataFrame, int]:
    """Load Maurer forcing data from text files.

    Parameters
    ----------
    camels_root : PosixPath
        Path to the main directory of the CAMELS data set
    basin : str
        8-digit USGS gauge id
    hdtype: str
        hydro type

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the Maurer forcing
    area: int
        Catchment area (read-out from the header of the forcing file)

    Raises
    ------
    RuntimeError
        If not forcing file was found.
    """
    # forcing_path = camels_root / 'basin_mean_forcing' / 'maurer_extended'
    forcing_path = camels_root / 'basin_mean_forcing' / hdtype  # CHANGE
    files = list(forcing_path.glob('**/*_forcing_leap.txt'))
    file_path = [f for f in files if f.name[:8] == basin]
    if len(file_path) == 0:
        raise RuntimeError(f'No file for Basin {basin} at {file_path}')
    else:
        file_path = file_path[0]

    df = pd.read_csv(file_path, sep=r'\s+', header=3)
    dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str))
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    # load area from header
    with open(file_path, 'r') as fp:
        content = fp.readlines()
        area = int(content[2])

    return df, area


def load_discharge(camels_root: PosixPath, basin: str, area: int) -> pd.Series:
    """[summary]

    Parameters
    ----------
    camels_root : PosixPath
        Path to the main directory of the CAMELS data set
    basin : str
        8-digit USGS gauge id
    area : int
        Catchment area, used to normalize the discharge to mm/day

    Returns
    -------
    pd.Series
        A Series containing the discharge values.

    Raises
    ------
    RuntimeError
        If no discharge file was found.
    """
    discharge_path = camels_root / 'usgs_streamflow'
    files = list(discharge_path.glob('**/*_streamflow_qc.txt'))
    file_path = [f for f in files if f.name[:8] == basin]
    if len(file_path) == 0:
        raise RuntimeError(f'No file for Basin {basin} at {file_path}')
    else:
        file_path = file_path[0]

    col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
    df = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)
    dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str))
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    # normalize discharge from cubic feed per second to mm per day
    df.QObs = 28316846.592 * df.QObs * 86400 / (area * 10 ** 6)

    return df.QObs


################################################################################
class AbstractReader(metaclass=ABCMeta):
    """Abstract data reader.

    Its subclasses need to ensure conversion from raw data to pandas.DataFrame,
    and process invalid data item
    """

    @abstractmethod
    def _load_data(self, *args, **kwargs):
        """
        Subclasses must implement loading inputs and target data.
        """
        pass

    @abstractmethod
    def _process_invalid_data(self, *args, **kwargs):
        """
        Subclasses must implement how to process invalid data item.
        """
        pass

    @abstractmethod
    def get_df_x(self):
        """
        Subclasses must return inputs data with a form of pandas.DataFrame.
        """
        pass

    @abstractmethod
    def get_df_y(self):
        """
        Subclasses must return target data with a form of pandas.DataFrame.
        """
        pass


class DaymetHydroReader(AbstractReader):
    camels_root = None  # needs to be set in class method "init_root"
    forcing_root = None
    discharge_root = None
    forcing_cols = ["Year", "Mnth", "Day", "Hr", "dayl(s)", "prcp(mm/day)", "srad(W/m2)", "swe(mm)", "tmax(C)",
                    "tmin(C)", "vp(Pa)"]
    features = ["prcp(mm/day)", "srad(W/m2)", "tmax(C)", "tmin(C)", "vp(Pa)"]
    discharge_cols = ["basin", "Year", "Mnth", "Day", "QObs", "flag"]
    target = ["QObs(mm/d)"]

    @classmethod
    def init_root(cls, camels_root):  # often be rewritten
        cls.camels_root = Path(camels_root)
        cls.forcing_root = cls.camels_root / "basin_mean_forcing" / "daymet"
        cls.discharge_root = cls.camels_root / "usgs_streamflow"

    def __init__(self, basin: str):
        self.basin = basin
        self.area = None
        df = self._load_data()
        df = self._process_invalid_data(df)
        self.df_x = df[self.features]  # Datetime as index
        self.df_y = df[self.target]  # Datetime as index

    def get_df_x(self):
        return self.df_x

    def get_df_y(self):
        return self.df_y

    def _load_data(self):
        df_forcing = self._load_forcing()
        df_discharge = self._load_discharge()
        df = pd.concat([df_forcing, df_discharge], axis=1)

        return df

    # Loading meteorological data
    def _load_forcing(self):
        files = list(self.forcing_root.glob(f"**/{self.basin}_*.txt"))
        if len(files) == 0:
            raise RuntimeError(f"No forcing file found for Basin {self.basin}")
        elif len(files) >= 2:
            raise RuntimeError(f"Redundant forcing files found for Basin {self.basin}")
        else:
            file_path = files[0]

        # read-in data and convert date to datetime index
        df = pd.read_csv(file_path, sep=r"\s+", header=3)  # \s+ means matching any whitespace character
        dates = df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str)
        df.index = pd.to_datetime(dates, format="%Y/%m/%d")

        # Line 2 (starting at 0) of the file is the area value
        with open(file_path) as fp:
            # readline is faster than readines, if only read two lines
            fp.readline()
            fp.readline()
            content = fp.readline().strip()
            area = int(content)
        self.area = area

        return df[self.features]

    # Loading runoff data
    def _load_discharge(self):
        files = list(self.discharge_root.glob(f"**/{self.basin}_*.txt"))
        if len(files) == 0:
            raise RuntimeError(f"No discharge file found for Basin {self.basin}")
        elif len(files) >= 2:
            raise RuntimeError(f"Redundant discharge files found for Basin {self.basin}")
        else:
            file_path = files[0]

        df = pd.read_csv(file_path, sep=r"\s+", header=None, names=self.discharge_cols)
        dates = df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str)
        df.index = pd.to_datetime(dates, format="%Y/%m/%d")

        # normalize discharge from cubic feed per second to mm per day
        assert len(self.target) == 1
        df[self.target[0]] = 28316846.592 * df["QObs"] * 86400 / (self.area * 10 ** 6)

        return df[self.target]

    # Processing invalid data
    def _process_invalid_data(self, df: pd.DataFrame):
        # Delete all row, where exits NaN (only discharge has NaN in this dataset)
        len_raw = len(df)
        df = df.dropna()
        len_drop_nan = len(df)
        if len_raw > len_drop_nan:
            pass
            # print(f"Deleted {len_raw - len_drop_nan} records because of NaNs {self.basin}")

        # Deletes all records, where no discharge was measured (-999)
        df = df.drop((df[df['QObs(mm/d)'] < 0]).index)
        len_drop_neg = len(df)
        if len_drop_nan > len_drop_neg:
            pass
            # print(f"Deleted {len_drop_nan - len_drop_neg} records because of negative discharge {self.basin}")

        return df


class MaurerExtHydroReader(DaymetHydroReader):
    camels_root = None  # needs to be set in class method "init_root"
    forcing_root = None
    discharge_root = None

    @classmethod
    def init_root(cls, camels_root):
        cls.camels_root = Path(camels_root)
        cls.forcing_root = cls.camels_root / "basin_mean_forcing" / "maurer_extended"
        cls.discharge_root = cls.camels_root / "usgs_streamflow"

    def __init__(self, basin: str):
        super().__init__(basin)


class NldasExtHydroReader(DaymetHydroReader):
    camels_root = None  # needs to be set in class method "init_root"
    forcing_root = None
    discharge_root = None
    forcing_cols = ["Year", "Mnth", "Day", "Hr", "Dayl(s)", "PRCP(mm/day)", "SRAD(W/m2)", "SWE(mm)", "Tmax(C)",
                    "Tmin(C)", "Vp(Pa)"]
    features = ["PRCP(mm/day)", "SRAD(W/m2)", "Tmax(C)", "Tmin(C)", "Vp(Pa)"]

    @classmethod
    def init_root(cls, camels_root):
        cls.camels_root = Path(camels_root)
        cls.forcing_root = cls.camels_root / "basin_mean_forcing" / "nldas_extended"
        cls.discharge_root = cls.camels_root / "usgs_streamflow"

    def __init__(self, basin: str):
        super().__init__(basin)


class HydroReaderFactory:
    """
    Simple factory for producing HydroReader
    """

    @staticmethod
    def get_hydro_reader(camels_root, forcing_type, basin):
        if forcing_type == "daymet":
            DaymetHydroReader.init_root(camels_root)
            reader = DaymetHydroReader(basin)
        elif forcing_type == "maurer_extended":
            MaurerExtHydroReader.init_root(camels_root)
            reader = MaurerExtHydroReader(basin)
        elif forcing_type == "nldas_extended":
            NldasExtHydroReader.init_root(camels_root)
            reader = NldasExtHydroReader(basin)
        else:
            raise RuntimeError(f"No such hydro reader type: {forcing_type}")

        return reader


# SCALER = {
#     # 'input_means': np.array([3.17563234, 372.01003929, 17.31934062, 3.97393362, 924.98004197]),
#     # 'input_stds': np.array([6.94344737, 131.63560881, 10.86689718, 10.3940032, 629.44576432]),
#     # 'output_mean': np.array([1.49996196]),
#     # 'output_std': np.array([3.62443672])
#     'input_means': np.array([]),
#     'input_stds': np.array([]),
#     'output_mean': np.array([]),
#     'output_std': np.array([])
# }

def calc_mean_and_std(data_dict):
    data_all = np.concatenate(list(data_dict.values()), axis=0)  # CAN NOT serializable
    nan_mean = np.nanmean(data_all, axis=0)
    nan_std = np.nanstd(data_all, axis=0)
    return nan_mean, nan_std


def cal_scaler(camels_root, basins, use_runoff):
    x_dict = dict()
    y_dict = dict()
    global SCALER
    for idx, basin in enumerate(basins):
        reader = HydroReaderFactory.get_hydro_reader(camels_root, "daymet", basin)
        df_x = reader.get_df_x()
        df_y = reader.get_df_y()
        train_start = pd.to_datetime("1980-10-01", format="%Y-%m-%d") - pd.DateOffset(days=14 - 1)
        train_end = pd.to_datetime("1995-09-30", format="%Y-%m-%d")
        train_start_runoff = train_start - pd.DateOffset(days=1)
        train_end_runoff = train_end - pd.DateOffset(days=1)

        if use_runoff:
            df_y = df_y[train_start_runoff: train_end]
            df_runoff = df_y[train_start_runoff:df_y.index[-1] - pd.DateOffset(days=1)]
            df_x = df_x[df_runoff.index[0] + pd.DateOffset(days=1):train_end]
            df_y = df_y[df_runoff.index[0] + pd.DateOffset(days=1):train_end]
            # print("before:")
            # print(df_runoff.shape, df_runoff)
            # print(df_x.shape, df_x)
            # print(df_y.shape, df_y)
            # if basin == "02096846":
            #     print(df_runoff, df_x, df_y)

            df_x = df_x.reset_index(drop=True)
            df_runoff = df_runoff.reset_index(drop=True)

            df_x = pd.concat([df_runoff, df_x], axis=1)
            # print("after:")
            # print(df_runoff.shape, df_runoff)
            # print(df_x.shape, df_x)
            # print(df_y.shape, df_y)
            # if basin == "02096846":
            #     print("after")
            #     print(df_runoff, df_x, df_y)
        else:
            df_x = df_x[train_start: train_end]
            # print(df_x.shape, df_x.head())
            df_y = df_y[train_start: train_end]
            # print(df_y)

        x = df_x.values.astype("float32")
        y = df_y.values.astype("float32")
        # print("final")
        # print(x.shape)
        # print(y.shape)
        x_dict[basin] = x
        y_dict[basin] = y

    SCALER['output_mean'], SCALER['output_std'] = calc_mean_and_std(y_dict)
    SCALER['input_means'], SCALER['input_stds'] = calc_mean_and_std(x_dict)
    print("scaler down:", SCALER)
