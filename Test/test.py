import pytest
import os

import tiktoken

from Models import GPT_4Tokenizer
from Models.basicTokenizer import BasicTokenizer
from Models.regexTokenizer import RegexTokenization
from train import tokenizer

# -----------------------------------------------------------------------------
# common test data

test_strings = [
    "", # empty string
    "?", # single character
    "hello world!!!? (안녕하세요!) lol123 😉", # fun small string
    "FILE:data.txt"# FILE: is handled as a special string in unpack()
]

special_token= {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

def uncap(text):
    if text.startswith("FILE:"):
        dirname = os.path.dirname(os.path.abspath(__file__))
        file_data = os.path.join(dirname ,text[5:])
        content = open(file_data,"r",encoding="utf-8").read()
        return  content
        pass
    else:
        return text

llama_text = """
<|endoftext|>The llama (/ˈlɑːmə/; Spanish pronunciation: [ˈʎama] or [ˈʝama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era.
Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5–8 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4]
The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000–12,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5]
<|fim_prefix|>In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology,<|fim_suffix|> where they come from at the end of time.[6]<|fim_middle|> llamas will return to the water springs and ponds<|endofprompt|>
""".strip()


# -----------------------------------------------------------------------------
# tests

@pytest.mark.parametrize("tokenizer_fact" , [BasicTokenizer,RegexTokenization])
@pytest.mark.parametrize("text",test_strings)
def test_encode_decode_identity(tokenizer_fact,text):
    text = uncap(text)
    tokenizers = tokenizer_fact()
    ids = tokenizers.encode(text)
    decoded = tokenizers.decode(ids)
    assert text == decoded


@pytest.mark.parametrize("text" , test_strings)
def test_gpt4_tiktoken_equality(text):
    tokenizer = GPT_4Tokenizer()
    enc = tiktoken.get_encoding("cl100k_base")
    tiktoken_ids =  enc.encode(text)
    tokenizer_ids = tokenizer.encode(text)
    assert tiktoken_ids ==  tokenizer_ids


@pytest.mark.parametrize("text" , test_strings)
def test_gpt4_tiktoken_equality_with_specialToken(text):
    tokenizer = GPT_4Tokenizer()
    enc = tiktoken.get_encoding("cl100k_base")
    tokenizer_ids = tokenizer.encode(text,allowed_special="all")
    tiktoken_ids = enc.encode(text,allowed_special="all")
    assert tiktoken_ids == tokenizer_ids

@pytest.mark.parametrize("tokenizer_fact",[BasicTokenizer,RegexTokenization])
def test_wikipedia_example(tokenizer_fact):
    """
        Quick unit test, following along the Wikipedia example:
        https://en.wikipedia.org/wiki/Byte_pair_encoding

        According to Wikipedia, running bpe on the input string:
        "aaabdaaabac"

        for 3 merges will result in string:
        "XdXac"

        where:
        X=ZY
        Y=ab
        Z=aa

        Keep in mind that for us a=97, b=98, c=99, d=100 (ASCII values)
        so Z will be 256, Y will be 257, X will be 258.

        So we expect the output list of ids to be [258, 100, 258, 97, 99]
    """

    tokenizer = tokenizer_fact()
    text = "aaabdaaabac"
    tokenizer.train(text,256+3,True)
    ids = tokenizer.encode(text)
    assert ids == [258, 100, 258, 97, 99]
    assert tokenizer.decode(tokenizer.encode(text)) == text

@pytest.mark.parametrize("special_tokens" ,[{},special_token])
def test_save_load(special_tokens):
    text = llama_text
    tokenizer = RegexTokenization()
    tokenizer.train(text,256+64)
    tokenizer.reg_Special_Token(special_tokens)

    assert  tokenizer.decode(tokenizer.encode(text,"all")) == text

    ids = tokenizer.encode(text,"all")

    tokenizer.save("test_reg_tokenizer")


    tokenizer = RegexTokenization()
    tokenizer.load("test_reg_tokenizer.model")
    s = tokenizer.special_token
    tokenizer.reg_Special_Token(s)

    assert tokenizer.decode(ids) == text
    assert tokenizer.encode(text,"all") == ids
    assert tokenizer.decode(tokenizer.encode(text,"all")) == text


    for file in ["test_reg_tokenizer.model", "test_reg_tokenizer.vocab"]:
        os.remove(file)



if __name__ == "__main__":
    pytest.main()