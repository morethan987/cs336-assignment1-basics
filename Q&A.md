## Answers for Problems

### unicode 1

1. What Unicode character does chr(0) return?
```python
>>> print(repr('a' + chr(0) + 'a'))
'a\x00a'
```
Ans: The Unicode control character U+0000 (NULL) -- invisible, but has a length of 1.

2. How does this character’s string representation (__repr__()) differ from its printed representation?
```python
>>> print("printed representation:", chr(0))
printed representation:
>>> print("string representation:", chr(0).__repr__())
string representation: '\x00'
```
Ans: string representation will show an invisible unicode control character as a visible string.

3. What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:
```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```
Ans: '\x00', (nothing), 'this is a test\x00string', this is a teststring

### unicode 2

1. What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.
Ans: There are too many 0 in UTF-16 and UTF-32 encoded bytes that represents nothing.

2. Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.
```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```
Ans: Some Unicode characters are represented by 2 or more bytes, for example, some chinese characters like '牛'. 
```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("牛".encode("utf-8"))
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe7 in position 0: unexpected end of data
```

3. Give a two-byte sequence that does not decode to any Unicode character(s).
Ans: bytes([0xc2, 0x41]), 0xc2 declares two-byte sequence that the following byte should be at lest 0x80. Actually the second byte is 0x41.

4. train_bpe and pass tests
Ans: success.

```txt
╭─ morethan@headlessArch  ~/github/cs336-assignment1-basics
╰─❯ uv run pytest tests/test_train_bpe.py
=============================================================================== test session starts ================================================================================
platform linux -- Python 3.13.14, pytest-9.0.2, pluggy-1.6.0
rootdir: /home/morethan/github/cs336-assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.9, timeout-2.4.0
collected 3 items

tests/test_train_bpe.py::test_train_bpe_speed PASSED
tests/test_train_bpe.py::test_train_bpe PASSED
tests/test_train_bpe.py::test_train_bpe_special_tokens PASSED

================================================================================ 3 passed in 1.41s =================================================================================
```

Time test for normal tokenization and parallelized tokenization. Approximately 66.28% relative advantage.

```txt
╭─ morethan@headlessArch  ~/github/cs336-assignment1-basics
╰─❯ uv run python -m timeit -s "from cs336_basics.pretokenization import pretokenize" "pretokenize('data/TinyStoriesV2-GPT4-valid.txt', [b'<|endoftext|>'])"
1 loop, best of 5: 2.86 sec per loop
╭─ morethan@headlessArch  ~/github/cs336-assignment1-basics
╰─❯ uv run python -m timeit -s "from cs336_basics.pretokenization import pretokenize_parallel" "pretokenize_parallel('data/TinyStoriesV2-GPT4-valid.txt', [b'<|endoftext|>'], 4)"
1 loop, best of 5: 1.72 sec per loop
```
