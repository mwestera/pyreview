from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import argparse
import os


PROMPT = """
I've done only a basic Python course and am still learning. Can you give some feedback on my code? 

```python
{code}
```

- Please give me AT MOST 3 points of improvement.
- Avoid 'fluff' and clichés.
- With specific reference to my code! 

Thanks so much!
"""



def main():

    argparser = argparse.ArgumentParser(description='Auto-review Python code for beginners.')
    argparser.add_argument('files', nargs='*', default='-', type=str)
    argparser.add_argument('--model', nargs='?', default="Qwen/CodeQwen1.5-7B-Chat", type=str)
    argparser.add_argument('--force', required=False, action='store_true')

    args = argparser.parse_args()

    if args.files == '-':
        programs = [sys.stdin.read()]
    else:
        programs = []
        for path in args.files:
            with open(path, 'r') as file:
                programs.append(file.read())

    device = "cuda"  # the device to load the model onto
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto"
    )
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model_inputs = build_model_inputs(programs, tokenizer).to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = (output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids))

    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    if args.files == '-':
        print(responses[0])
    else:
        for path, response in zip(args.files, responses):
            outpath = path.replace('.py', '.md')
            if os.path.exists(outpath) and not args.force:
                raise FileExistsError(f'Feedback file exists: {outpath}! Use --force to overwrite.')
            with open(outpath, 'w') as outfile:
                outfile.write(response)


def build_model_inputs(programs, tokenizer):
    texts = []

    for program in programs:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": PROMPT.format(code=program)}
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