from .base import Tokenization, get_stats, merge
import unicodedata


class BasicTokenizer(Tokenization):

    def __init__(self):
        super().__init__()


    def train(self,text,vocab_size,verbose = False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256


        token = text.encode("utf-8")
        ids= list(token)

        merges = {}
        vocab = {idx : bytes([idx]) for idx in range(256)}

        for i in range(num_merges):

            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256+i

            ids = merge(ids,pair,idx)

            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"merging {pair} into a new token {idx}")
        self.merges = merges
        self.vocab = vocab


    def encode(self,text):
        ids = list(text.encode("utf-8"))
        print(len(ids))
        while len(text) >= 2:
            stats = get_stats(ids)
            pair = min(stats ,key = lambda x : self.merges.get(x,float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]

            ids = merge(ids,pair,idx)

        return ids


    def decode(self,ids):
      token = b"".join(self.vocab.get(t, b"") for t in ids)
      text = token.decode("utf-8",errors="ignore")

      return text