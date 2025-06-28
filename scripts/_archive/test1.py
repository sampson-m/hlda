from pathlib import Path
import numpy as np
import pandas as pd

# ----------- 1. open the memmap -----------
MEMMAP_PATH = Path("../samples/ABCD_V1V2/6_topic_fit/A_chain.memmap")      # 👉  update to your file
SHAPE       = (100, 2000)               # 👉  plug in known dims
DTYPE       = np.uint32                       # or np.int32 / np.int64

X = np.memmap(MEMMAP_PATH, mode="r", shape=SHAPE, dtype=DTYPE)

# ----------- 2. raw count diagnostics -----------
row_sums = X.sum(axis=1)
summary  = pd.Series(row_sums).describe(percentiles=[.05,.5,.95])
print("🔍 library-size stats per doc\n", summary)

# quick checks
assert (row_sums >= 0).all(),  "negative counts detected"
assert (row_sums > 0).any(),   "all-zero rows – sampler will explode"

# ----------- 3. derive normalised matrix -----------
# (convert on-the-fly; keeps mem footprint low)
row_sums_safe = row_sums.astype(np.float32)
row_sums_safe[row_sums_safe == 0] = 1  # avoid division by zero
X_norm = X.astype(np.float32) / row_sums_safe[:, None]

# ----------- 4. verify normalisation -----------
err = np.abs(X_norm.sum(axis=1) - 1.0).max()
print(f"max |row-sum − 1| after normalisation: {err:.3e}")
assert err < 1e-6, "rows are not properly normalised"

# (optional) spot-check first few rows / columns
print("\n🎯 first 3 documents after normalisation")
print(np.round(X_norm[:3, :10], 3))
