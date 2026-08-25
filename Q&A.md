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

### train_bpe

1. train_bpe and pass tests
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

Final outputs on both datasets:

```txt
vocabulary list length:  10000
special tokens:  b'<|endoftext|>'
first 20 tokens:
[b' t', b'he', b' a', b' s', b' w', b'nd', b' the', b'ed', b' b', b' to', b' and', b' h', b' f', b'in', b' T', b' wa', b're', b'it', b'ou', b' l']
last 20 tokens:
[b' meets', b' marvel', b' Rusty', b' Liza', b' Jet', b'Froggy', b' wrapper', b' Reddy', b' Hops', b' Crusty', b' whiskers', b' nicest', b' improving', b' booth', b' Land', b'Surrender', b'Rocky', b' meadows', b' imaginary', b' bold']
longest token:
 accomplishment

vocabulary list length:  32000
special tokens:  b'<|endoftext|>'
first 20 tokens:
[b' t', b' a', b'he', b'in', b're', b' the', b'on', b'er', b' s', b' w', b'at', b' o', b'en', b' c', b'it', b'is', b'an', b'or', b' b', b'es']
last 20 tokens:
[b' gradient', b' discredited', b' pg', b' merging', b' disple', b' Hz', b' Aristotle', b'ulls', b'ircraft', b'YD', b'Wik', b' partnering', b' drilled', b'hang', b' pesticide', b' latitude', b' fetish', b' Fruit', b'ivists', b'Disp']
longest token:
ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ
```

### train_bpe_tinystories

1. Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size of 10,000. Make sure to add the TinyStories <|endoftext|> special token to the vocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How much time and memory did training take? What is the longest token in the vocabulary? Does it make sense? 
Ans: Whole traing process takes 216.03s. Memory distribution:
=== memory by stage (RSS, tree-summed) ===
  init           n=    1  avg=    29.5 MB  peak=    29.5 MB
  pretokenize    n= 3100  avg=  1252.0 MB  peak=  2331.5 MB
  merge_loop     n=   15  avg=   119.1 MB  peak=   124.8 MB
  OVERALL        peak=  2331.5 MB

longest token is " accomplishment", make sense.

2. Profile your code. What part of the tokenizer training process takes the most time?
Ans:
Code profile:

```txt
         494641868 function calls (494641801 primitive calls) in 412.435 seconds

   Ordered by: cumulative time
   List reduced from 389 to 25 due to restriction <25>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    1.926    1.926  412.435  412.435 /home/morethan/github/cs336-assignment1-basics/cs336_basics/bpe_trainer.py:62(train)
        1    0.000    0.000  279.911  279.911 /home/morethan/github/cs336-assignment1-basics/cs336_basics/pretokenization.py:77(pretokenize_parallel)
        1    0.000    0.000  279.856  279.856 /home/morethan/.local/share/uv/python/cpython-3.13.14-linux-x86_64-gnu/lib/python3.13/concurrent/futures/_base.py:652(__exit__)
        1    0.000    0.000  279.856  279.856 /home/morethan/.local/share/uv/python/cpython-3.13.14-linux-x86_64-gnu/lib/python3.13/concurrent/futures/process.py:845(shutdown)
      2/1    0.000    0.000  279.851  279.851 /home/morethan/.local/share/uv/python/cpython-3.13.14-linux-x86_64-gnu/lib/python3.13/threading.py:1059(join)
      2/1    0.000    0.000  279.851  279.851 {method 'join' of '_thread._ThreadHandle' objects}
      2/1    0.000    0.000  279.851  279.851 /home/morethan/.local/share/uv/python/cpython-3.13.14-linux-x86_64-gnu/lib/python3.13/threading.py:1001(_bootstrap)
      2/1    0.000    0.000  279.851  279.851 /home/morethan/.local/share/uv/python/cpython-3.13.14-linux-x86_64-gnu/lib/python3.13/threading.py:1028(_bootstrap_inner)
        1    0.000    0.000  279.851  279.851 /home/morethan/.local/share/uv/python/cpython-3.13.14-linux-x86_64-gnu/lib/python3.13/concurrent/futures/process.py:330(run)
        1    0.000    0.000  279.851  279.851 /home/morethan/.local/share/uv/python/cpython-3.13.14-linux-x86_64-gnu/lib/python3.13/concurrent/futures/process.py:549(join_executor_internals)
        1    0.000    0.000  279.851  279.851 /home/morethan/.local/share/uv/python/cpython-3.13.14-linux-x86_64-gnu/lib/python3.13/concurrent/futures/process.py:553(_join_executor_internals)
        1    0.000    0.000  279.849  279.849 /home/morethan/.local/share/uv/python/cpython-3.13.14-linux-x86_64-gnu/lib/python3.13/multiprocessing/queues.py:145(join_thread)
        6    0.000    0.000  279.849   46.641 /home/morethan/.local/share/uv/python/cpython-3.13.14-linux-x86_64-gnu/lib/python3.13/multiprocessing/util.py:276(__call__)
        1    0.000    0.000  279.848  279.848 /home/morethan/.local/share/uv/python/cpython-3.13.14-linux-x86_64-gnu/lib/python3.13/multiprocessing/queues.py:212(_finalize_join)
        6    0.000    0.000  279.848   46.641 /home/morethan/.local/share/uv/python/cpython-3.13.14-linux-x86_64-gnu/lib/python3.13/concurrent/futures/process.py:405(wait_result_broken_or_wakeup)
       17    0.000    0.000  279.811   16.459 /home/morethan/.local/share/uv/python/cpython-3.13.14-linux-x86_64-gnu/lib/python3.13/multiprocessing/connection.py:1160(wait)
       17    0.000    0.000  279.810   16.459 /home/morethan/.local/share/uv/python/cpython-3.13.14-linux-x86_64-gnu/lib/python3.13/selectors.py:385(select)
       17  279.810   16.459  279.810   16.459 {method 'poll' of 'select.poll' objects}
     9751   88.712    0.009  129.920    0.013 {built-in method builtins.max}
490440103   41.208    0.000   41.208    0.000 /home/morethan/github/cs336-assignment1-basics/cs336_basics/bpe_trainer.py:82(<lambda>)
  2460252    0.242    0.000    0.242    0.000 {built-in method builtins.len}
   759789    0.160    0.000    0.160    0.000 {method 'add' of 'set' objects}
   759779    0.160    0.000    0.160    0.000 {method 'setdefault' of 'dict' objects}
        1    0.097    0.097    0.106    0.106 /home/morethan/github/cs336-assignment1-basics/cs336_basics/pretokenization.py:52(init_word_tokens)
        4    0.043    0.011    0.053    0.013 /home/morethan/.local/share/uv/python/cpython-3.13.14-linux-x86_64-gnu/lib/python3.13/collections/__init__.py:928(__iadd__)
```

Pre-tokenization: 279.81s; Merging: 129.92s, find the most freqent pair is the bottleneck.

### train_bpe_expts_owt

1. Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection. What is the longest token in the vocabulary? Does it make sense?
Ans: longest token is "ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ". It may cause by none-utf8 encoded text in web pages.

2. Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.
Ans: some early statge tokens are nearly same since both dataset is English corpus. OpenWebText contains some dirty text which leads to some strange token such as "ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ". Tokens generated in final stage reflect the topic of corpus.

### tokenizer
Ans: success.

```txt
╭─ morethan@headlessArch  ~/github/cs336-assignment1-basics
╰─❯ uv run pytest tests/test_tokenizer.py
=============================================================================== test session starts ================================================================================
platform linux -- Python 3.13.14, pytest-9.0.2, pluggy-1.6.0
rootdir: /home/morethan/github/cs336-assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.9, timeout-2.4.0
collected 25 items

tests/test_tokenizer.py::test_roundtrip_empty PASSED
tests/test_tokenizer.py::test_empty_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_single_character PASSED
tests/test_tokenizer.py::test_single_character_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_single_unicode_character PASSED
tests/test_tokenizer.py::test_single_unicode_character_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_ascii_string PASSED
tests/test_tokenizer.py::test_ascii_string_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_unicode_string PASSED
tests/test_tokenizer.py::test_unicode_string_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_unicode_string_with_special_tokens PASSED
tests/test_tokenizer.py::test_unicode_string_with_special_tokens_matches_tiktoken PASSED
tests/test_tokenizer.py::test_overlapping_special_tokens PASSED
tests/test_tokenizer.py::test_address_roundtrip PASSED
tests/test_tokenizer.py::test_address_matches_tiktoken PASSED
tests/test_tokenizer.py::test_german_roundtrip PASSED
tests/test_tokenizer.py::test_german_matches_tiktoken PASSED
tests/test_tokenizer.py::test_tinystories_sample_roundtrip PASSED
tests/test_tokenizer.py::test_tinystories_matches_tiktoken PASSED
tests/test_tokenizer.py::test_encode_special_token_trailing_newlines PASSED
tests/test_tokenizer.py::test_encode_special_token_double_newline_non_whitespace PASSED
tests/test_tokenizer.py::test_encode_iterable_tinystories_sample_roundtrip PASSED
tests/test_tokenizer.py::test_encode_iterable_tinystories_matches_tiktoken PASSED
tests/test_tokenizer.py::test_encode_iterable_memory_usage PASSED
tests/test_tokenizer.py::test_encode_memory_usage XFAIL (Tokenizer.encode is expected to take more memory than allotted (1MB).)

========================================================================== 24 passed, 1 xfailed in 8.68s ===========================================================================
```

### tokenizer_experiments

1. Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinyStories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), encode these sampled documents into integer IDs. What is each tokenizer’s compression ratio (bytes/token)?
Ans: TinyStory tokenizer is 4.171, OWT tokenizer is 4.478.

```txt
TinyStory compression ratio results:
['4.130', '4.439', '4.110', '4.192', '4.028', '4.305', '4.188', '4.126', '3.863', '4.327']
Average: 4.171

OpenWebText compression ratio results:
['4.781', '5.036', '4.002', '4.339', '4.642', '4.394', '4.421', '4.447', '4.134', '4.589']
Average: 4.478

```

2. What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Compare the compression ratio and/or qualitatively describe what happens.
Ans: Compression ratio is 3.370, relatively drops 19.53%. OWT document encoded with TinyStory tokenizer contains some tivial pieces such as 'W' 'om' 'en', which should be a single token 'Women', leading to a lower compression ratio.

```txt
Compression ratio results of OpenWebText encoded with TinyStory tokenizer:
['3.736', '2.906', '3.410', '3.550', '3.665', '3.642', '3.075', '2.856', '3.543', '3.001']
Average: 3.338

OpenWebText compression ratio results:
['5.261', '4.025', '4.181', '5.272', '4.263', '4.228', '3.669', '4.199', '4.456', '4.599']
Average: 4.415
Relatively drops:0.2439

One sample output:
Encoding results of OpenWebText with TinyStory tokenizer:
['W', 'om', 'en', ' can', ' live', ' at', ' the', ' res', 'idence', ' for']

Encoding results of OpenWebText with OpenWebText tokenizer:
['Women', ' can', ' live', ' at', ' the', ' residence', ' for', ' up', ' to', ' five']
```

3. Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to tokenize the Pile dataset (825GB of text)?
Ans: The throughput is 1.47 MB/s for TinyStory tokenizer, 1.35 MB/s for OWT tokenizer. Tokenizing Pile dataset need 159.64 hours.

```txt
Dataset                MB/s       tokens/s    bytes/token
------------------------------------------------------------
TinyStories            1.47        374,023          4.117
OpenWebText            1.35        324,726          4.367
```

4. Using your TinyStories and OpenWebText tokenizers, encode the respective training and development datasets into a sequence of integer token IDs. We’ll use this later to train our language model. We recommend serializing the token IDs as a NumPy array of datatype uint16. Why is uint16 an appropriate choice?
Ans: There is no negative IDs so uint16 is enough which has the same poritive range with int32.

### linear
Ans: success.

```txt
============================= test session starts ==============================
platform linux -- Python 3.13.14, pytest-9.0.2, pluggy-1.6.0
rootdir: /home/morethan/github/cs336-assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.9, timeout-2.4.0
collected 48 items / 47 deselected / 1 selected

tests/test_model.py::test_linear PASSED

======================= 1 passed, 47 deselected in 0.98s =======================
```

### embedding

Ans: success.

```txt
============================= test session starts ==============================
platform linux -- Python 3.13.14, pytest-9.0.2, pluggy-1.6.0
rootdir: /home/morethan/github/cs336-assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.9, timeout-2.4.0
collected 48 items / 47 deselected / 1 selected

tests/test_model.py::test_embedding PASSED

======================= 1 passed, 47 deselected in 0.99s =======================
```

### rmsnorm

Ans: success.

```txt
============================= test session starts ==============================
platform linux -- Python 3.13.14, pytest-9.0.2, pluggy-1.6.0
rootdir: /home/morethan/github/cs336-assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.9, timeout-2.4.0
collected 48 items / 47 deselected / 1 selected

tests/test_model.py::test_rmsnorm PASSED

======================= 1 passed, 47 deselected in 1.00s =======================
```

### positionwise_feedforward

Ans: success.

```txt
============================= test session starts ==============================
platform linux -- Python 3.13.14, pytest-9.0.2, pluggy-1.6.0
rootdir: /home/morethan/github/cs336-assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.9, timeout-2.4.0
collected 48 items / 47 deselected / 1 selected

tests/test_model.py::test_swiglu PASSED

======================= 1 passed, 47 deselected in 1.03s =======================
```

### rope

Ans: success.

```txt
============================= test session starts ==============================
platform linux -- Python 3.13.14, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/morethan/github/cs336-assignment1-basics
configfile: pyproject.toml
plugins: timeout-2.4.0, jaxtyping-0.3.11
collected 48 items / 47 deselected / 1 selected

tests/test_model.py::test_rope PASSED

======================= 1 passed, 47 deselected in 1.06s =======================
```

### softmax

Ans: success.

```txt
============================= test session starts ==============================
platform linux -- Python 3.13.14, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/morethan/github/cs336-assignment1-basics
configfile: pyproject.toml
plugins: timeout-2.4.0, jaxtyping-0.3.11
collected 48 items / 47 deselected / 1 selected

tests/test_nn_utils.py::test_softmax_matches_pytorch PASSED

======================= 1 passed, 47 deselected in 0.85s =======================
```

### scaled_dot_product_attention

Ans: success.

```txt
uv run pytest -k test_scaled_dot_product_attention

============================= test session starts ==============================
platform linux -- Python 3.13.14, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/morethan/github/cs336-assignment1-basics
configfile: pyproject.toml
plugins: timeout-2.4.0, jaxtyping-0.3.11
collected 48 items / 47 deselected / 1 selected

tests/test_model.py::test_scaled_dot_product_attention PASSED

======================= 1 passed, 47 deselected in 1.02s =======================

uv run pytest -k test_4d_scaled_dot_product_attention

============================= test session starts ==============================
platform linux -- Python 3.13.14, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/morethan/github/cs336-assignment1-basics
configfile: pyproject.toml
plugins: timeout-2.4.0, jaxtyping-0.3.11
collected 48 items / 47 deselected / 1 selected

tests/test_model.py::test_4d_scaled_dot_product_attention PASSED

======================= 1 passed, 47 deselected in 1.01s =======================
```

### multihead_self_attention

Ans: success.

```txt
============================= test session starts ==============================
platform linux -- Python 3.13.14, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/morethan/github/cs336-assignment1-basics
configfile: pyproject.toml
plugins: timeout-2.4.0, jaxtyping-0.3.11
collected 48 items / 46 deselected / 2 selected

tests/test_model.py::test_multihead_self_attention PASSED
tests/test_model.py::test_multihead_self_attention_with_rope PASSED

======================= 2 passed, 46 deselected in 1.40s =======================
```

### transformer_block

Ans: success.

```txt
============================= test session starts ==============================
platform linux -- Python 3.13.14, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/morethan/github/cs336-assignment1-basics
configfile: pyproject.toml
plugins: timeout-2.4.0, jaxtyping-0.3.11
collected 48 items / 47 deselected / 1 selected

tests/test_model.py::test_transformer_block PASSED

======================= 1 passed, 47 deselected in 1.49s =======================
```

### transformer_lm

Ans: success.

```txt
============================= test session starts ==============================
platform linux -- Python 3.13.14, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/morethan/github/cs336-assignment1-basics
configfile: pyproject.toml
plugins: timeout-2.4.0, jaxtyping-0.3.11
collected 48 items / 46 deselected / 2 selected

tests/test_model.py::test_transformer_lm PASSED
tests/test_model.py::test_transformer_lm_truncated_input PASSED

======================= 2 passed, 46 deselected in 2.14s =======================
```

### transformer_accounting

1. Consider a GPT-2 XL-sized model using our assignment architecture, which has the following configuration:

```txt
vocab_size: 50257
context_length: 1024
num_layers: 48
d_model: 1600
num_heads: 25
d_ff: 4288 (the nearest multiple of 64 to 8/3 x 1600)
```

Suppose we constructed our model using this configuration. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?
Ans: Exactly trainable parameters is 1,640,452,800, approximately 1.6B. Approximately 6.5GB.

2. Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has context_length tokens.
Ans: Approximately 2.51TFLOPs.

```json
{
  "transformer_block": {
    "multi_head_attention": 2 * 1600 * 2 * 1024 * 1024,
    "swiglu": 3 * 2 * 1024 * 1600 * 4288
  },
  "output_linear": 2 * 1024 * 1600 * 50257
}
```

3. Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium (24 layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?
Ans: Small costs 0.234TFLOPs, medium costs 0.624TFLOPs, large costs 1.3TFLOPs. With model size increases, the FFN costs much more FLOPs than other two parts.

```txt
=== GPT-2 small ===
Total FLOPs: 233,666,248,704  (233.666 GFLOPs)
Component                       FLOPs        %  Distribution
--------------------------------------------------------------------------------
MHA                    38,654,705,664   16.54%  ████████
SwiGLU/FFN            115,964,116,992   49.63%  █████████████████████████
Output Linear          79,047,426,048   33.83%  █████████████████

=== GPT-2 medium ===
Total FLOPs: 624,013,869,056  (624.014 GFLOPs)
Component                       FLOPs        %  Distribution
--------------------------------------------------------------------------------
MHA                   103,079,215,104   16.52%  ████████
SwiGLU/FFN            415,538,085,888   66.59%  █████████████████████████████████
Output Linear         105,396,568,064   16.89%  ████████

=== GPT-2 large ===
Total FLOPs: 1,303,466,475,520  (1.303 TFLOPs)
Component                       FLOPs        %  Distribution
--------------------------------------------------------------------------------
MHA                   193,273,528,320   14.83%  ███████
SwiGLU/FFN            978,447,237,120   75.07%  ██████████████████████████████████████
Output Linear         131,745,710,080   10.11%  █████
```

4. Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one forward pass change? How does the relative contribution of FLOPs of the model components change?
Ans: Total FLOPs is 46.8 times of short context one. Multi head attention becomes the major part since the complexity grow quadratically with context length.

```txt
=== GPT-2 XL (ctx 1024) ===
Total FLOPs: 2,510,136,934,400  (2.510 TFLOPs)
Component                       FLOPs        %  Distribution
--------------------------------------------------------------------------------
MHA                   322,122,547,200   12.83%  ██████
SwiGLU/FFN          2,023,332,249,600   80.61%  ████████████████████████████████████████
Output Linear         164,682,137,600    6.56%  ███

=== GPT-2 XL (ctx 16384) ===
Total FLOPs: 117,471,602,278,400  (117.472 TFLOPs)
Component                       FLOPs        %  Distribution
--------------------------------------------------------------------------------
MHA                82,463,372,083,200   70.20%  ███████████████████████████████████
SwiGLU/FFN         32,373,315,993,600   27.56%  ██████████████
Output Linear       2,634,914,201,600    2.24%  █
```

### cross_entropy

Ans: success.

```txt
============================= test session starts ==============================
platform linux -- Python 3.13.14, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/morethan/github/cs336-assignment1-basics
configfile: pyproject.toml
plugins: timeout-2.4.0, jaxtyping-0.3.11
collected 48 items / 47 deselected / 1 selected

tests/test_nn_utils.py::test_cross_entropy PASSED

======================= 1 passed, 47 deselected in 0.98s =======================
```

### learning_rate_tuning

As we will see, one of the hyperparameters that affects training the most is the learning rate. Let’s see that in practice in our toy example. Run the SGD example above with three other values for the learning rate: 1e1, 1e2, and 1e3, for just 10 training iterations. What happens with the loss for each of these learning rates? Does it decay faster, slower, or does it diverge (i.e., increase over the course of training)?

Ans: Suitable learning rates make loss decay faster (1, 10, 100), but an extremely large lr explode (1000).

```txt
=============================================
LR       | Step   | Loss           
---------------------------------------------
1.0      | 0      | 21.902155      
1.0      | 1      | 21.034828      
1.0      | 2      | 20.444082      
1.0      | 3      | 19.974670      
1.0      | 4      | 19.577175      
1.0      | 5      | 19.228535      
1.0      | 6      | 18.915817      
1.0      | 7      | 18.630915      
1.0      | 8      | 18.368368      
1.0      | 9      | 18.124271      
---------------------------------------------
10.0     | 0      | 21.902155      
10.0     | 1      | 14.017379      
10.0     | 2      | 10.333013      
10.0     | 3      | 8.084479       
10.0     | 4      | 6.548428       
10.0     | 5      | 5.429397       
10.0     | 6      | 4.578977       
10.0     | 7      | 3.912866       
10.0     | 8      | 3.379067       
10.0     | 9      | 2.943543       
---------------------------------------------
100.0    | 0      | 21.902155      
100.0    | 1      | 21.902155      
100.0    | 2      | 3.757815       
100.0    | 3      | 0.089933       
100.0    | 4      | 0.000000       
100.0    | 5      | 0.000000       
100.0    | 6      | 0.000000       
100.0    | 7      | 0.000000       
100.0    | 8      | 0.000000       
100.0    | 9      | 0.000000       
---------------------------------------------
1000.0   | 0      | 21.902155      
1000.0   | 1      | 7906.677734    
1000.0   | 2      | 1365607.625000 
1000.0   | 3      | 151909280.000000
1000.0   | 4      | 12304650240.000000
1000.0   | 5      | 776564310016.000000
1000.0   | 6      | 39866272317440.000000
1000.0   | 7      | 1715217738235904.000000
1000.0   | 8      | 63219268602298368.000000
1000.0   | 9      | 2030040962746548224.000000
---------------------------------------------
```

### adamw

Ans: success.

```txt
============================= test session starts ==============================
platform linux -- Python 3.13.14, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/morethan/github/cs336-assignment1-basics
configfile: pyproject.toml
plugins: timeout-2.4.0, jaxtyping-0.3.11
collected 48 items / 47 deselected / 1 selected

tests/test_optimizer.py::test_adamw PASSED

======================= 1 passed, 47 deselected in 1.40s =======================
```
