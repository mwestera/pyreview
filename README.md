# PyReview

Wrapper around Huggingface  `transformers` for language generation with Large Language Models, to easily request feedback on beginner Python code. 

## Install

```bash
$ pip/pipx install pyreview
```

## Use

```bash
$ pyreview some_code.py
``` 

This will write feedback to `some_code.md`; add the option `--force` to overwrite if it already exists. You can also apply it to all files in the current working directory:

```bash
$ pyreview *.py --force
``` 

Or feed code through `stdin`, and it'll print feedback to `stdout`:

```bash
$ cat some_code.py | pyreview
``` 

A sensible default model will be used. You can specify any huggingface instruct model and it should work (though this particular model didn't give great results):

```bash
$ cat some_code.py | pyreview --model Qwen/CodeQwen1.5-7B-Chat
```

It works on both .py and .ipynb files (though only the notebook's code is fed into the LLM), and you can include `--withcode` to repeat the raw code in the feedback files, for easy reference:

```bash
pyreview *.py *.ipynb --withcode
```

You can add some further nudges to the basic prompt:

```bash
$ cat some_code.py | pyreview --nudge "Start your feedback with 'Dear human overlord'" "Format your feedback as a haiku please."
``` 

And you can add a custom prefix to the feedback files (default: a disclaimer about LLM reliability).

```bash
$ cat some_code.py | pyreview --prefix "WARNING: The following was written by a robot."
``` 


## Which model to use?

Consider some of the 'instruct' models here:

https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard
