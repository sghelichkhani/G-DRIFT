from .earthmodel3d import EarthModel3D
from .io import load_dataset
from numbers import Number

AVAILABLE_SEISMIC_MODELS = [
    "MITP08",
    "GyPSuM",
    "SPani",
    "SAW642ANb",
    "SAW642AN",
    "SEMum",
    "S40RTS",
    "OJP",
    "HMSL-P06",
    "SAW24B16",
    "TX2000",
    "SGLOBE-rani",
    "SEMUCB-WM1",
    "GAP",
    "S362ANI+M",
    "S20RTS",
    "TX2011",
    "SEISGLOB2",
    "SP12RTS",
    "TX2019slab",
    "S362ANI",
    "REVEAL",
    "HMSL-S06",
    "S362WMANI",
    "LLNL-G3Dv3",
]


class SeismicModel(EarthModel3D):
    def __init__(self, model_name, nearest_neighbours: int = 8, default_max_distance: float = 200e3):
        """SeismicModel is a class for handling 3D seismic models.

        This class inherits from EarthModel3D and is used to load and manage seismic models.
        It allows for the initialization of a seismic model with a specified name, number of nearest neighbours,
        and a default maximum distance. The model data is loaded from a dataset file, and quantities and coordinates
        are set accordingly.
                max_distance (float, optional): The default maximum distance for the model in meters. Defaults to 200e3.
        Attributes:
            model_name (str): The name of the seismic model.
            nearest_neighbours (int): The number of nearest neighbours to consider.
            default_max_distance (float): The default maximum distance for the model in meters.

        Methods:
            _load_available_models(): Loads and returns available seismic models from the data directory.
        """
        if model_name not in AVAILABLE_SEISMIC_MODELS:
            raise ValueError(f"Model '{model_name}' not found in available models. Choose from: {', '.join(AVAILABLE_SEISMIC_MODELS)}")

        raw_model = load_dataset(f"3d_seismic_{model_name}")

        super().__init__(nearest_neighbours=nearest_neighbours, default_max_distance=default_max_distance)

        for quantity_name in raw_model.keys():
            if quantity_name != "coordinates":
                self.add_quantity(quantity_name, raw_model[quantity_name])
            else:
                self.set_coordinates(raw_model[quantity_name])
