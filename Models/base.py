
"""
Contains the base Tokenizer class and a few common helper functions.
The base class also contains the (common) save/load functionality.
It would be possible to be a lot more strict about the interface and
e.g. isolating all regex/pattern parts to the RegexTokenizer, but
some concessions are made for simplicity.
"""

import unicodedata




# -----------------------------------------------------------------------------
# a few helper functions useful for both BasicTokenizer and RegexTokenizer

def get_stats(ids,count=None):
    count = {}if count is None else count
    i =1
    for pair in zip(ids,ids[i:]):
        count[pair] = count.get(pair,0) +1
    return count

def merge(ids,pair,idx):
    newids = []
    i =0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def replace_control_characters(s : str)-> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table

    char =[]
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            char.append(ch)
        else:
            char.append(f"\\u{ord(ch):04x}")
    return "".join(char)


def render_token(t: bytes) -> str: # Convert string to bytes
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s



class Tokenization:
    """Base class for Tokenizers"""

    def __init__(self):
        self.merges = {}
        self.pattern = ""
        self.special_token = {}
        self.vocab = self._build_vocab()

    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(self, text):
        # Tokenizer can encode a string into a list of integers
        raise NotImplementedError

    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError

    def _build_vocab(self):
        vocab = {idx : bytes([idx]) for idx in range(256)}
        for (p0,p1),idx in self.merges.items():
            vocab[idx] = vocab[p0]+vocab[p1]
        return vocab

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """

        model_file = file_prefix + ".model"
        with open(model_file,'w') as f:
            f.write("pmbpe v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_token)}\n")
            for special , key in self.special_token.items():
                f.write(f"{special} {key}\n")
            for idx1,idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")


        vocab_file = file_prefix +".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file,"w",encoding="utf-8") as f:
            for idx , token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)

                if idx in inverted_merges:
                    idx0 , idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """Inverse of save() but only for the model file"""

        assert model_file.endswith(".model")

        merges = {}
        special_toke = {}
        idx = 256

        with open(model_file,'r',encoding="utf-8") as f:

            Version = f.readline().strip()
            assert  Version == "pmbpe v1"

            self.pattern = f.readline().strip()

            num_special = int(f.readline().strip())

            for _ in range(num_special):
                special , special_idx = f.readline().strip().split()
                special_toke[special] = int(special_idx)

            for line in f:
                idx1,idx2 = map(int,line.split())
                merges[(idx1,idx2)] = idx
                idx+=1

            self.merges = merges
            self.special_token = special_toke
            self.vocab = self._build_vocab()