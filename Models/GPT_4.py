"""
Implements the GPT-4 Tokenizer as a light wrapper around the RegexTokenizer.
Note that this is a pretrained tokenizer. By default and inside init(), it
loads the pretrained tokenizer from the `cl100k_base` tokenizer of tiktoken.
"""


import tiktoken

from .regexTokenizer import RegexTokenization

# helper
def bpe(merge_rank,token,max_rank):
    parts = [bytes([b]) for b in token]

    while True:

        min_idx = None
        min_rank = None

        for i , pair in enumerate(zip(parts[:-1],parts[1:])):
            rank = merge_rank.get(pair[0]+pair[1])

            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank

        if min_rank is  None or (max_rank is not None and min_rank >= max_rank):
            break

        assert min_idx is not None

        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx+1]] + parts[min_idx+2:]

    return parts

def recover_merges(merge_rank):

    merges = {}

    for token,rank in merge_rank.items():
        if len(token) == 1:
            continue
        pair = tuple(bpe(merge_rank,token,max_rank= rank))

        assert  len(pair) == 2

        idx0 = merge_rank[pair[0]]
        idx1 = merge_rank[pair[1]]

        merges[(idx0,idx1)] = rank

    return merges

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}
class GPT_4Tokenizer(RegexTokenization):
    def __init__(self):
        super().__init__(pattern=GPT4_SPLIT_PATTERN)

        enc = tiktoken.get_encoding("cl100k_base")

        merge_rank  = enc._mergeable_ranks
        # the merges are those of gpt4, but we have to recover them
        self.merges = recover_merges(merge_rank)
        # reconstruct the vocab from the merges
        vocab = {idx: bytes([idx])for idx in range(256)}
        for (p0,p1),idx in self.merges.items():
            vocab[idx] = vocab[p0]+vocab[p1]

        self.vocab = vocab
        # now here is another tricky part.
        # for some reason, the tokens corresponding to individual bytes
        # are permuted in a different order. This is completely non-sensical
        # and probably historical, but therefore we have to deal with it here.

        self.byte_shuffle  = {idx : merge_rank[bytes([idx])]   for idx in range(256)}
        self.inverse_byte_shuffle  = {v:k for k,v in self.byte_shuffle .items()}

        # finally register the special tokens
        self.reg_Special_Token(GPT4_SPECIAL_TOKENS)


    def encode_chunk(self,text_bytes):
        text_bytes = bytes(self.byte_shuffle[b] for b in text_bytes)
        ids = super().encode_chunk(text_bytes)
        return ids

    def decode(self,ids):
        # we have to un-permute the bytes before we decode
        bytes_text = b"".join(self.vocab[idx] for idx in ids)
        bytes_text = bytes(self.inverse_byte_shuffle[b] for b in bytes_text)
        text = bytes_text.decode("utf-8" , errors="ignore")
        return text

    # this is a pretrained tokenizer, it is not intended to be trained
    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError

    # save/load would require some thought.
    # we'd have to change save/load of base to add support for byte_shuffle...
    # alternatively, we could move byte_shuffle to base class, but that would
    # mean that we're making ugly our beautiful Tokenizer just to support
    # the GPT-4 tokenizer and its weird historical quirks around byte_shuffle.
    def save(self, file_prefix):
        raise NotImplementedError("GPT4Tokenizer cannot be saved...")

    def load(self, mode_file):
        raise NotImplementedError("GPT4Tokenizer cannot be loaded...")

    def save_vocab(self,vocab_file):
        from .base import render_token

        vocab = {idx:bytes([self.inverse_byte_shuffle[idx]]) for idx in range(256)}

        for (p0,p1) , idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]


        inverted_merges =  {idx:pair for pair,idx in self.merges.items()}
        with open(vocab_file,"w",encoding="utf-8") as f:
            for idx, T in vocab.items():
                s = render_token(T)
                if idx in inverted_merges.items():

                    idx0,idx1  = inverted_merges[idx]
                    s0 = render_token(vocab[idx0])
                    s1 = render_token(vocab[idx1])

                    f.write(f"[{s0}] [{s1}] -> [{s}] {idx}")

                else:
                    f.write(f"[{s}] {idx}")




