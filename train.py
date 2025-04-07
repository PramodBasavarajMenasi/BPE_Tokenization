
from Models.regexTokenizer import RegexTokenization

# text = open("Test/data.txt" ,"r",encoding="utf-8").read()
# text = "aaabdaaabac <|endoftext|> "
#
text = "<|endoftext|> hello how r you"


# text  =    """ Minimal, clean code for the (byte-level) Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization. The BPE algorithm is "byte-level" because it runs on UTF-8 encoded strings.
#
# This algorithm was popularized for LLMs by the GPT-2 paper and the associated GPT-2 code release from OpenAI. Sennrich et al. 2015 is cited as the original reference for the use of BPE in NLP applications. Today, all modern LLMs (e.g. GPT, Llama, Mistral) use this algorithm to train their tokenizers.
#
# There are two Tokenizers in this repository, both of which can perform the 3 primary functions of a Tokenizer: 1) train the tokenizer vocabulary and merges on a given text, 2) encode from text to tokens, 3) decode from tokens to text. The files of the repo are as follows:
#
# minbpe/base.py: Implements the Tokenizer class, which is the base class. It contains the train, encode, and decode stubs, save/load functionality, and there are also a few common utility functions. This class is not meant to be used directly, but rather to be inherited from.
# minbpe/basic.py: Implements the BasicTokenizer, the simplest implementation of the BPE algorithm that runs directly on text.
# minbpe/regex.py: Implements the RegexTokenizer that further splits the input text by a regex pattern, which is a preprocessing stage that splits up the input text by categories (think: letters, numbers, punctuation) before tokenization. This ensures that no merges will happen across category boundaries. This was introduced in the GPT-2 paper and continues to be in use as of GPT-4. This class also handles special tokens, if any. """

special_tokens = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}
tokenizer = RegexTokenization()
tokenizer.reg_Special_Token(special_tokens)

# tokenizer.train(text,256+3,True)
t = tokenizer.encode(text,"all")

print(len(t))
print(t)

text_decoded = tokenizer.decode(t)
print(str(text_decoded))

if text_decoded == text:
    print("1")



