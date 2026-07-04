"""Lazy marker-positivity loading extracted from ``config.py``.

``LazyMarkerPositivityDict`` defers reading per-dataset marker-positivity
DataFrames out of the zarr archive until first access, so constructing a
``TissueNetConfig`` does not iterate all ~1,900 datasets when only ~285 have
marker-positivity data. Re-exported from ``deepcell_types.training.config``
for backward compatibility.
"""

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .config import TissueNetConfig


class LazyMarkerPositivityDict(dict):
    """Dict-like object that lazily loads marker positivity DataFrames on demand.

    Only datasets that actually have marker_positivity groups in the zarr archive
    will be loaded, and only when first accessed. This avoids iterating all ~1,900
    datasets at init time when only ~285 have MP data.

    Supports __contains__, __getitem__, keys(), values(), items(), __iter__, __len__,
    and list() for full dict compatibility (e.g., nimbus.py iterates keys).
    """

    def __init__(self, config: "TissueNetConfig", mp_keys: List[str]):
        """
        Args:
            config: TissueNetConfig instance (for _load_marker_positivity)
            mp_keys: List of dataset keys that have marker_positivity groups
        """
        super().__init__()
        self._config = config
        self._mp_keys = set(mp_keys)
        self._loaded_keys: set = set()  # Track which keys we've attempted to load
        self._fully_loaded = False

    def __reduce_ex__(self, protocol):
        loaded = {k: dict.__getitem__(self, k) for k in dict.keys(self)}
        state = {
            "zarr_path": (
                str(self._config.zarr_path)
                if getattr(self._config, "zarr_path", None) is not None
                else None
            ),
            "mp_keys": list(self._mp_keys),
            "loaded_keys": list(self._loaded_keys),
            "fully_loaded": self._fully_loaded,
        }
        return (self.__class__._from_pickle, (state, loaded))

    @classmethod
    def _from_pickle(cls, state, loaded):
        obj = cls.__new__(cls)
        dict.__init__(obj)
        dict.update(obj, loaded)
        obj.__setstate__(state)
        return obj

    def __setstate__(self, state):
        # Local import avoids a module-level circular import: config.py imports
        # this module, and TissueNetConfig is only needed here at unpickle time.
        from .config import TissueNetConfig

        zarr_path = state.get("zarr_path")
        self._config = TissueNetConfig(zarr_path) if zarr_path is not None else None
        self._mp_keys = set(state.get("mp_keys", []))
        self._loaded_keys = set(state.get("loaded_keys", []))
        self._fully_loaded = bool(state.get("fully_loaded", False))

    def _load_one(self, key: str):
        """Load a single dataset's marker positivity if not already loaded."""
        if key not in self._loaded_keys:
            self._loaded_keys.add(key)
            df = self._config._load_marker_positivity(key)
            if df is not None:
                super().__setitem__(key, df)

    def _load_all(self):
        """Load all MP datasets (for iteration)."""
        if self._fully_loaded:
            return
        for key in self._mp_keys:
            self._load_one(key)
        self._fully_loaded = True

    def __contains__(self, key):
        if super().__contains__(key):
            return True
        if key in self._mp_keys and key not in self._loaded_keys:
            self._load_one(key)
            return super().__contains__(key)
        return False

    def __getitem__(self, key):
        if not super().__contains__(key) and key in self._mp_keys:
            self._load_one(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        self._load_all()
        return super().keys()

    def values(self):
        self._load_all()
        return super().values()

    def items(self):
        self._load_all()
        return super().items()

    def __iter__(self):
        self._load_all()
        return super().__iter__()

    def __len__(self):
        self._load_all()
        return super().__len__()
