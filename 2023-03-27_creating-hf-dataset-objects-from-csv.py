# # Converting csv/pandas data to HF and using FAISS index for nearest-neighour identification  # noqa

# ## References
# - [Ch09 of transformers book](https://learning.oreilly.com/library/view/natural-language-processing/9781098136789/ch09.html#idm45146285976912)  # noqa
# - [HF docs on FAISS](https://huggingface.co/docs/datasets/faiss_es)
# - [HF technical docs on FAISS](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.add_faiss_index)  # noqa
# - [facebook research github on FAISS](https://github.com/facebookresearch/faiss)
# - [FAISS docs](https://faiss.ai/)
# - [HF docs on creating dataset from local tabular data](https://huggingface.co/docs/datasets/tabular_load#csv-files)  # noqa

# ## Notes
# - For comparing text similarity, it is more common to use cosine similarity (which
#     ranges from -1 to 1) than L2 distance (which is unbounded). See more on this
#     [page](https://deepnote.com/blog/semantic-search-using-faiss-and-mpnet).

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


# Check that nearest neighbour to existing value is itself:

VALUE_TO_LOOKUP = np.array(df['embed_1d'][0])
nearest_results = data.get_nearest_examples('embed_1d', VALUE_TO_LOOKUP, k=1)
assert nearest_results.examples['embed_1d'][0] == VALUE_TO_LOOKUP.tolist()


# Get nearest neighbour for new example:

VALUE_TO_LOOKUP = np.array([.19])

nearest_results = data.get_nearest_examples('embed_1d', VALUE_TO_LOOKUP, k=1)

type(nearest_results)
dir(nearest_results)

nearest = nearest_results.examples['embed_1d']
nearest
nearest_array = np.array(nearest[0])

score = nearest_results.scores
l2_manual = np.sqrt(np.sum((VALUE_TO_LOOKUP - nearest_array) ** 2))
inner_prod_manual = np.dot(VALUE_TO_LOOKUP, nearest_array)


def cosine_similarity(x1, x2):
    dot_product = np.dot(x1, x2)
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)
    return dot_product / (norm_x1 * norm_x2)


cosine_manual = cosine_similarity(VALUE_TO_LOOKUP, nearest_array)

# todo: why doesn't the score match either the L2 or inner prod that we manually
#   calculated?

