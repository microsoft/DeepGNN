# DeepGNN note, TriLetterTokenizer, TriletterEmbeddings and WeightPooler were taken from Author's implementation of TwinBERT.
# Details please refer to https://arxiv.org/abs/2002.06275.

# flake8: noqa
from .encoder import TwinBERTEncoder
from .tokenization import TriLetterTokenizer, StdBertTokenizer
