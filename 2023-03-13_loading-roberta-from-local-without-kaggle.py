# # Loading a pretrained roberta model from local in a local environment

# ## Overview
# - The saved model has the following files:
#   - for the tokenizer:
#     - config.json (also used for the model)
#     - merges.txt
#     - vocab.json and dict.txt. Note that dict.txt is for human readability and is
#       not directly used.
#   - for the model:
#     - config.json
#     - pytorch_model.bin


from pathlib import Path

import torch
from transformers import RobertaTokenizer, RobertaModel


MODEL_PATH = Path(r'C:\Nayef\transformers-book\model\pretrained-roberta')

tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_PATH)

texts = ['I want to tokenize this', 'And this too', 'What about this?']
texts = [text.lower() for text in texts]
text_encoded = tokenizer(texts, padding=True)
model_inputs = tokenizer(texts, return_tensors='pt', padding=True)

model = RobertaModel.from_pretrained(pretrained_model_name_or_path=MODEL_PATH)
model

if model.training:
    print("We first set the model to eval mode, then do inference")
    model.eval()

with torch.no_grad():
    last_hidden_state = model(**model_inputs).last_hidden_state
    cls_token_embeddings = last_hidden_state[:, 0, :].numpy()

assert cls_token_embeddings.shape == (3, 768)