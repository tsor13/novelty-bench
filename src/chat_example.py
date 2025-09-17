import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import tqdm
import json
import os

def get_infinite_chat_eval():
    # liweijiang/infinite-chats-eval
    dataset = load_dataset("liweijiang/infinite-chats-eval")
    prompts = dataset['train']['query']
    return prompts

def get_novelty_bench_curated():
    # yimingzhang/novelty-bench
    dataset = load_dataset("yimingzhang/novelty-bench")
    prompts = dataset['curated']['prompt']
    return prompts

def debug_prompts():
    prompts = [
        "What is the capital of France?",
        "Tell me a joke about Texas.",
        "What are the Norse Gods?",
        "What would you say is the best kind of food?",
        "Explain why someone would choose to be a priest in three sentences",
        "Explain the 2008 financial crisis to me in about one paragraph.",
        "What color should I paint my car? Respond with just a color",
        "Pick a number between 1 and 100",
        "Write a python function for doing binary search on a sorted list.",
        "Write a poem about a shark.",
    ]
    return prompts

splits = {
    "infinite-chats-eval": get_infinite_chat_eval,
    "novelty-bench-curated": get_novelty_bench_curated,
    "debug": debug_prompts,
}

# from Llama-3.1-70B-Instruct
llama_examples = [
    {"prompt": "What is the capital of France?", "response": "The capital of France is Paris."},
    {"prompt": "What color should I paint my car? Respond with just a color", "response": "Navy blue"},
    {"prompt": "Explain the 2008 financial crisis to me in about one paragraph.", "response": "The 2008 financial crisis, also known as the Global Financial Crisis (GFC), was a severe economic downturn that began in the United States in 2007 and spread globally, lasting through 2009. The crisis was triggered by a housing market bubble bursting in the US, fueled by lax lending practices and excessive speculation. Many homeowners, especially those with subprime mortgages, were unable to afford their mortgage payments, leading to widespread defaults and foreclosures. This created a crisis in the financial markets, as many banks and financial institutions had invested heavily in mortgage-backed securities (MBS) that were now worthless. As a result, these institutions found themselves facing massive losses, which in turn led to a credit crisis, stock market crash, and a global recession. The crisis was exacerbated by other factors, including over-reliance on complex financial instruments, inadequate regulation, and a global economic imbalance. The crisis led to unprecedented government interventions, including massive bailouts and stimulus packages, to stabilize the financial system and stimulate economic growth."},
    {"prompt": "Tell me a joke about Texas.", "response": "Why did the Texan bring a ladder to the party? \n\nBecause he heard the drinks were on the house."},
]

# from Qwen3-32B
qwen_examples = [
    {"prompt": "What is the capital of France?", "response": "The capital of France is **Paris**."},
    {"prompt": "What color should I paint my car? Respond with just a color", "response": "Crimson Red"},
    {"prompt": "Explain the 2008 financial crisis to me in about one paragraph.", "response": "The 2008 financial crisis was triggered by a housing market bubble in the United States that burst, leading to a severe credit crunch and global economic downturn. Starting in the early 2000s, lax lending standards and speculative investment fueled a rapid rise in home prices. Banks and financial institutions began offering high-risk subprime mortgages to borrowers with poor credit, then bundled these loans into complex financial products like mortgage-backed securities (MBS) and collateralized debt obligations (CDOs), which were widely traded. When housing prices peaked and began to fall in 2006-2007, many borrowers defaulted on their mortgages, causing the value of these financial instruments to plummet. This led to massive losses for banks and investors, triggering a wave of bankruptcies and failures, including major institutions like Lehman Brothers. The crisis spread globally as financial systems were interconnected, resulting in a deep recession with soaring unemployment, foreclosures, and a widespread loss of consumer and business confidence. In response, governments and central banks implemented massive bailouts and stimulus measures to stabilize the economy."},
    {"prompt": "Tell me a joke about Texas.", "response": "Sure! Here's a Texas-themed joke for you:\n\nWhy don't cowboys in Texas ever get cold?\n\nBecause the only *draft* they know is from the beer and the wind... and even that's a *light beer breeze*!"},
]

# from gemma-3-27b-it
gemma_examples = [
    {"prompt": "What is the capital of France?", "response": "The capital of France is **Paris**. \n\nIt's known for its iconic landmarks like the Eiffel Tower, Louvre Museum, and Arc de Triomphe, as well as its fashion, cuisine, and culture."},
    {"prompt": "What color should I paint my car? Respond with just a color", "response": "Forest green."},
    {"prompt": "Explain the 2008 financial crisis to me in about one paragraph.", "response": "The 2008 financial crisis was a severe worldwide economic downturn triggered by the collapse of the US housing market. Years of low interest rates and lax lending standards led to a boom in subprime mortgages and loans given to borrowers with poor credit. These mortgages were packaged into complex financial products called mortgage-backed securities and sold to investors globally. When housing prices began to fall in 2006-2007, many borrowers defaulted, causing these securities to lose value. This triggered a cascading effect: banks and financial institutions holding these assets faced massive losses, credit markets froze, businesses couldn't get loans, and ultimately, major institutions like Lehman Brothers collapsed. The crisis spread globally, leading to a severe recession, job losses, and government bailouts to stabilize the financial system."},
    {"prompt": "Tell me a joke about Texas.", "response": "Why did the Texan bring a ladder to the rodeo? \n\n...He heard the steaks were high! \n\n😂\n\n\n\nHope that gave you a little chuckle! Let me know if you'd like to hear another one."},
]

example_map = {
    "llama": llama_examples,
    "qwen": qwen_examples,
    "gemma": gemma_examples,
}


if __name__ == "__main__":
    # get model name
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    # optionally, alternate tokenizer
    parser.add_argument("--tokenizer_name", type=str, required=False)
    parser.add_argument("--split", type=str, required=True)
    # flag for explicit
    parser.add_argument("--explicit", action="store_true", required=False)
    # explicit description (default: "You are a helpful AI assistant.")
    parser.add_argument("--explicit_description", type=str, required=False, default="You are a helpful AI assistant.")
    # flag for short_system_prompt
    parser.add_argument("--short_system_prompt", action="store_true", required=False)
    # n gens (default 4)
    parser.add_argument("--n_gens", type=int, required=False, default=4)
    # flag for add_examples
    parser.add_argument("--add_examples", action="store_true", required=False)
    args = parser.parse_args()

    prompts = splits[args.split]()

    # get tokenizer and model
    tokenizer_path = args.model_name
    if args.tokenizer_name:
        tokenizer_path = args.tokenizer_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", trust_remote_code=True)

    stop_strings = [tokenizer.eos_token, "<|im_end|>", "<end_of_turn>", "<|eot_id|>"]

    examples = None
    if args.add_examples:
        # iterate and if the example key is in model name, set examples to the example map[model name]
        for key, value in example_map.items():
            if key in args.model_name.lower():
                examples = value
                break
        if examples is None:
            raise ValueError(f"No examples found for model name: {args.model_name}. Either add the examples to the example map or remove the --add_examples flag.")

    all_gens = {}
    # for prompt in prompts:
    for prompt in tqdm.tqdm(prompts):
        inputs = None
        if args.explicit:
            icl_messages = [
                {"role": "description", "content": args.explicit_description},
            ]
            # add examples, making sure that the prompt is at the end
            if args.add_examples:
                # get the 

                # if example in examples, don't add it
                if prompt in [example["prompt"] for example in examples]:
                    continue
                for example in examples:
                    icl_messages.append({"role": "input", "content": example["prompt"]})
                    icl_messages.append({"role": "output", "content": example["response"]})
            icl_messages.append({"role": "input", "content": prompt})
            # use the model's chat template
            input_text = tokenizer.messages_to_text(icl_messages, start_generation=True)
            inputs = tokenizer(input_text, return_tensors="pt")
        else:
            chat_messages = [
                {"role": "user", "content": prompt},
            ]
            if args.short_system_prompt:
                chat_messages.insert(0, {"role": "system", "content": "Respond with just the requested output."})
            additional_args = {}
            # additional_args = {"enable_thinking": False}
            if "qwen" in args.model_name.lower():
                additional_args = {"enable_thinking": False}
            input_text = tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True, return_tensors="pt", **additional_args)
            # tokenize
            inputs = tokenizer(input_text, return_tensors="pt")
            # if two bos tokens, remove the first one (for gemma)
            # if inputs.input_ids[0, 0] == tokenizer.bos_token_id and inputs.input_ids[0, 1] == tokenizer.bos_token_id:
            while inputs.input_ids[0, 0] == tokenizer.bos_token_id and inputs.input_ids[0, 1] == tokenizer.bos_token_id:
                inputs.input_ids = inputs.input_ids[:, 1:]
                inputs.attention_mask = inputs.attention_mask[:, 1:]
        print(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False))
        inputs = inputs.to(model.device)
        print(inputs['input_ids'][0])
        input_len = inputs.input_ids.shape[1]
        gens = []
        for _ in range(args.n_gens):
            outputs = model.generate(**inputs, max_new_tokens=256, stop_strings=stop_strings, tokenizer=tokenizer, top_p=1, do_sample=True, temperature=1, use_cache=True)
            outputs = outputs[0, input_len:]
            generation = tokenizer.decode(outputs, skip_special_tokens=True)
            print(generation)
            gens.append(generation)
        all_gens[prompt] = gens

    # save as jsonl to gens/split/model_name.jsonl
    save_path = f"eval_scripts/gen_human_diverse/gens/{args.split}/{args.model_name}.jsonl"
    # make dirs if not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        for prompt, gens in all_gens.items():
            f.write(json.dumps({"prompt": prompt, "gens": gens}) + "\n")

