# # Converting csv/pandas data to HF and using FAISS index for nearest-neighour identification  # noqa

# ## References
# - [Ch09 of transformers book](https://learning.oreilly.com/library/view/natural-language-processing/9781098136789/ch09.html#idm45146285976912)  # noqa
# - [HF docs on FAISS](https://huggingface.co/docs/datasets/faiss_es)
# - [HF docs on creating dataset from local tabular data](https://huggingface.co/docs/datasets/tabular_load#csv-files)  # noqa

import numpy as np
import pandas as pd
from datasets import Dataset

# Let's say we're embedding each observation into a 1D vector (i.e. basically a float).
# We can create a FAISS index on these vectors.

df = pd.DataFrame({
    'text': ['example_01', 'example_02', 'example_03'],
    'embed_1d': [[.5], [.2], [.3]]
})

data = Dataset.from_pandas(df)

data
dir(data)
data.column_names
data.data

data.add_faiss_index(column='embed_1d')


# Get nearest example:

VALUE_TO_LOOKUP = np.array([.19])

nearest_results = data.get_nearest_examples('embed_1d', VALUE_TO_LOOKUP, k=1)

type(nearest_results)
dir(nearest_results)

nearest = nearest_results.examples['embed_1d']
print(nearest)