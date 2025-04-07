

import regex  as re
from .base import Tokenization, get_stats , merge


GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenization(Tokenization):

    def __init__(self,pattern=None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern  = re.compile(self.pattern)
        self.special_token = {}
        self.inverse_special_token = {}

    def train(self,text,vocab_size,verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size-256

        text_chunk = re.findall(GPT4_SPLIT_PATTERN,text)
        ids = [list(ch.encode("utf-8")) for ch in text_chunk ]

        merges = {}
        vocab = {idx:bytes([idx]) for idx in range(256)}

        for i in range(num_merges):

            stats = {}
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts
                get_stats(chunk_ids, stats)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            idx =256+i

            ids = [merge(chunk_id,pair,idx) for chunk_id in ids]


            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if  verbose:
                print(f"merging {pair} into a new token {idx}")

        self.merges = merges
        self.vocab = vocab
        print(list(self.vocab[32]))


    def reg_Special_Token(self,special_token):
        self.special_token  =special_token
        self.inverse_special_token = {v:k for k,v in self.special_token.items()}


    def decode(self, ids):
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_token:
                part_bytes.append(self.inverse_special_token[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode_chunk(self,text_bytes):
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats,key =lambda x : self.merges.get(x,float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]

            ids = merge(ids,pair,idx)
        return ids


    def encode_ordinary(self, text):
        text_chunks = re.findall(self.compiled_pattern,text)
        print(len(text_chunks))
        ids =[]
        for chunk in text_chunks:
            chunk_byte = chunk.encode("utf-8")
            chunk_ids  = self.encode_chunk(chunk_byte)
            ids.extend(chunk_ids)
        return ids


    def encode(self,text,allowed_special="none_raise"):

        special = None
        if allowed_special == "all":
            special = self.special_token
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert  all(t not in text for t in self.special_token)
        elif isinstance(allowed_special,set):
            special = {k:v for k,v in self.special_token.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")


        # Ordinary input - text (Not contain any sort of special token)
        if not special:
            return self.encode_ordinary(text)

        special_pattern = "("+"|".join(re.escape(k) for k in self.special_token)+")"
        special_chunks = re.split(special_pattern,text)

        ids = []

        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))

        return ids




