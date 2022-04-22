import re
from transformers.models.bert.tokenization_bert import BertTokenizer  # type: ignore


class StdBertTokenizer(BertTokenizer):
    def __init__(self, vocab_file):
        super(StdBertTokenizer, self).__init__(vocab_file)

    def extract_from_sentence(self, sentence, max_seq_len=12):
        seq = [0] * max_seq_len
        mask = [0] * max_seq_len
        ret = self.encode(sentence, add_special_tokens=False)
        seq_len = min(max_seq_len, len(ret))
        seq[0:seq_len] = ret[0:seq_len]
        mask[0:seq_len] = [1] * seq_len
        return seq, mask


class TriLetterTokenizer:
    """Taken from TwinBERT Author's implementation."""

    def __init__(self, l3g_path):
        self._init_lg3_dict(l3g_path)
        self.invalid = re.compile("[^a-zA-Z0-9 ]")
        self.multispace = re.compile("  +")

    def _init_lg3_dict(self, l3g_path):
        self.l3g_dict = {}
        with open(l3g_path, "r", encoding="utf-8") as fin:
            for i, token in enumerate(fin):
                token = token.strip("\r\n")
                if len(token) == 0:
                    continue
                # reserve 0 as default, start from 1
                self.l3g_dict[token] = i + 1

    def extract_from_sentence(self, text, max_seq_len=12, max_n_letters=20):
        step1 = text.lower()
        step2 = self.invalid.sub("", step1)
        step3 = self.multispace.sub(" ", step2)
        step4 = step3.strip()
        words = step4.split(" ")
        return self._from_words_to_id_sequence(words, max_seq_len, max_n_letters)

    def _from_words_to_id_sequence(self, words, max_seq_len=12, max_n_letters=20):
        n_seq = min(len(words), max_seq_len)
        n_letter = max_n_letters
        feature_seq = [0] * (max_seq_len * max_n_letters)
        seq_mask = [0] * max_seq_len
        for i in range(n_seq):
            if words[i] == "":
                words[i] = "#"
            word = "#" + words[i] + "#"
            n_letter = min(len(word) - 2, max_n_letters)
            for j in range(n_letter):
                s = word[j : (j + 3)]
                if s in self.l3g_dict:
                    feature_seq[i * max_n_letters + j] = self.l3g_dict[s]
            seq_mask[i] = 1
        return feature_seq, seq_mask
