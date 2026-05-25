"""One-off: load the legacy pickle under numpy 2.x using a custom Unpickler
that swaps the (dtype, ndarray) tuple to (ndarray, dtype) before BUILD calls
NDArrayBacked.__setstate__. Then re-save in current format.
"""
import io
import pickle
import shutil
import numpy as np
import pandas as pd
from pandas._libs.arrays import NDArrayBacked

PICKLE = "top_5000_yf_data.pkl"
BACKUP = "top_5000_yf_data.pkl.legacy-bak"


class CompatUnpickler(pickle._Unpickler):
    def load_build(self):
        state = self.stack[-1]
        inst = self.stack[-2]
        if isinstance(inst, NDArrayBacked) and isinstance(state, tuple) and len(state) == 2:
            a, b = state
            if not isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                a, b = b, a  # swap to (ndarray, dtype)
            self.stack[-1] = (a, b, {})  # promote to 3-tuple form expected by modern pandas
        return super().load_build()

    dispatch = dict(pickle._Unpickler.dispatch)
    dispatch[pickle.BUILD[0]] = load_build


print(f"Backing up {PICKLE} -> {BACKUP}")
shutil.copy2(PICKLE, BACKUP)

print("Loading legacy pickle with CompatUnpickler...")
with open(PICKLE, "rb") as f:
    df = CompatUnpickler(f).load()
print(f"  Loaded: shape={df.shape}, index dtype={df.index.dtype}")
print(f"  Index range: {df.index.min()} -> {df.index.max()}")

print("Re-saving pickle in current pandas/numpy format...")
df.to_pickle(PICKLE)
print("Done.")
