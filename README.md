# PyReview

Wrapper around Huggingface  `transformers` for language generation, to easily request feedback on beginner Python code. 

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

You can specify any huggingface instruct model and it should work:

```bash
$ cat some_code.py | pyreview --model jwnder/codellama_CodeLlama-70b-Instruct-hf-bnb-4bit
```

Lastly, you can add some further nudges to the basic prompt:

```bash
$ cat some_code.py | pyreview --nudge "Start your feedback with 'Dear human overlord'" "Format your feedback as a haiku please."
``` 
