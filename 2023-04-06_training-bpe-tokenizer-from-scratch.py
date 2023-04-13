# # Training a BPE tokenizer from scratch

from pathlib import Path
from tokenizers import Tokenizer, pre_tokenizers, models, trainers, decoders
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import Lowercase

# Create a dataset
data = [
    "I have a dog.",
    "The dog is brown.",
    "I love my dog.",
    "The cat is on the table.",
]

DST_PATH = Path(r'C:\Nayef\transformers-book\dst')
DST_FILE = 'data.txt'

# Save the dataset to a file
with open(DST_PATH.joinpath(DST_FILE), "w") as f:
    for line in data:
        f.write(line + "\n")

# Define the BPE tokenizer
tokenizer = Tokenizer(models.BPE())

# Set the pre-tokenizer to split the text into words
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Set the normalizer to lowercase the text
tokenizer.normalizer = Lowercase()

# Train the BPE tokenizer
trainer = trainers.BpeTrainer(
    vocab_size=1000,
    min_frequency=1,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
)
tokenizer.train(files=[str(DST_PATH.joinpath(DST_FILE))], trainer=trainer)

# Configure the tokenizer to add special tokens [CLS] and [SEP]
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)

# Configure the tokenizer to decode tokens back to the original text
tokenizer.decoder = decoders.BPEDecoder()

# Save the tokenizer
tokenizer.save(str(DST_PATH.joinpath("bpe_tokenizer.json")))

# Load the tokenizer
loaded_tokenizer = Tokenizer.from_file(str(DST_PATH.joinpath("bpe_tokenizer.json")))

# Test the tokenizer
text = "The cat is on the table."
text = text.lower()
encoded = loaded_tokenizer.encode(text)
print(encoded.tokens)
print(encoded.ids)

# Test the tokenizer's decoding
decoded = loaded_tokenizer.decode(encoded.ids)
print(decoded)

