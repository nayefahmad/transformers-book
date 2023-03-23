# # Using a state_dict to load a saved roberta model

# ## References:
# - [hf docs](https://huggingface.co/transformers/v1.0.0/model_doc/overview.html#loading-google-ai-or-openai-pre-trained-weights-or-pytorch-dump)

from pathlib import Path
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast

STATE_DICT_PATH = Path(r'./model/roberta_model_state_dict_fine_tuned.pt')

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base')
model_02 = RobertaForSequenceClassification.from_pretrained('roberta-base')

old_state_dict = model.roberta.state_dict()
state_dict = torch.load(STATE_DICT_PATH)
assert old_state_dict.keys() == state_dict.keys()

# comparing the weights in one of the last layers:
old = old_state_dict['encoder.layer.11.output.LayerNorm.bias']
new = state_dict['encoder.layer.11.output.LayerNorm.bias']
assert not torch.equal(old, new)

# load the new state dict:
model_02.roberta.load_state_dict(state_dict)

# confirm that loading worked:
updated = model_02.roberta.state_dict()['encoder.layer.11.output.LayerNorm.bias']
assert torch.equal(updated, new)
assert not torch.equal(updated, old)

print('done')