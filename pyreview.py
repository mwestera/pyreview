import json

from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import argparse
import os
import functools
import logging
import nbformat


PROMPT = """
Can you give me some feedback on my Python code?

```python
{code}
```

- Please give me AT MOST 3 points of improvement.
- No fluff and clichés; refer to specific parts of my code! 
- DON'T write the corrected code for me.
- Don't recommend any package outside the Python standard library.
{nudge}
Thanks so much!
""".strip()

MAX_NEW_TOKENS = 1024

PREFIX = """
This feedback was generated by a large language model (LLM): {model}. 

- LLMs are often wrong; they may not understand the intent of your code.
- The model was not given the assignment, only your code.
- The model was not given information about your current knowledge of Python or the course contents.

##################################################
"""


def main():

    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    programs = []

    for file in args.files:
        content = file.read()

        if os.path.splitext(file.name)[-1] == '.py':
            program = content
        else:   # .ipynb or stdin
            try:
                notebook = nbformat.reads(content, as_version=4)
                program = extract_notebook_code(notebook)
            except nbformat.ValidationError:
                program = content

        programs.append(program)

    prompt_format = functools.partial(PROMPT.format, nudge=args.nudge)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model_inputs = build_model_inputs(programs, tokenizer, prompt_format)
    model_outputs = model.generate(model_inputs.input_ids, max_new_tokens=MAX_NEW_TOKENS)
    generated_ids = (output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, model_outputs))
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    if len(args.files) == 1 and args.files[0] == sys.stdin:
        if args.prefix:
            print(args.prefix)
        print(responses[0])
        if args.withcode:
            print(f'\n## Your submitted code:\n\n```python\n{programs[0]}\n```\n')
    else:
        for file, program, response in zip(args.files, programs, responses):
            outpath = os.path.splitext(file.name)[0] + '.md'
            if os.path.exists(outpath) and not args.force:
                raise FileExistsError(f'Feedback file exists: {outpath}! Use --force to overwrite.')
            with open(outpath, 'w') as outfile:
                if args.prefix:
                    outfile.write(args.prefix + '\n')
                outfile.write(response + '\n')
                if args.withcode:
                    outfile.write(f'\n## Your submitted code:\n\n```python\n{program}\n```\n')

                logging.info(f'Feedback written to {outfile.name}.')


def parse_args():
    argparser = argparse.ArgumentParser(description='Auto-review Python code for beginners.')
    argparser.add_argument('files', nargs='*', default=[sys.stdin], type=argparse.FileType('r'), help='Specify one or more .py or .ipynb files; default stdin')
    argparser.add_argument('--model', nargs='?', default="jwnder/codellama_CodeLlama-70b-Instruct-hf-bnb-4bit", type=str)
    argparser.add_argument('--force', required=False, action='store_true', help='To force overwriting if feedback .md files already exist.')
    argparser.add_argument('--nudge', nargs='*', type=str, help='To tweak the default prompt by adding one or more nudges.')
    argparser.add_argument('--prefix', required=False, type=str, default=PREFIX, help='To prefix some text in each feedback file; default is a disclaimer about LLMs.')
    argparser.add_argument('--withcode', required=False, action='store_true', help='To include the input code in the feedback file.')

    args = argparser.parse_args()
    if args.prefix and '{model}' in args.prefix:
        args.prefix = args.prefix.format(model=args.model)
    if args.nudge:
        args.nudge = ''.join(f'- {nudge}\n' for nudge in args.nudge)

    return args


def extract_notebook_code(notebook):

    code_cell_contents = [cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code']

    return '\n\n'.join(code_cell_contents)


def build_model_inputs(programs, tokenizer, prompt_format):
    texts = []

    for program in programs:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_format(code=program).strip()}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        texts.append(text)

    return tokenizer(texts, return_tensors="pt")


if __name__ == '__main__':
    main()