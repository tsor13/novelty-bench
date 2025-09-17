import argparse
import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import cohere
import torch
from aiofiles import open as aio_open
from anthropic import AsyncAnthropicVertex
from datasets import load_dataset
from google import genai
from google.auth import default, transport
from google.genai import types
from openai import AsyncOpenAI
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
import json

from src.common import oai_client


class InferenceService(ABC):
    @abstractmethod
    async def generate(
        self, model: str, messages: list[dict[str, str]], **kwargs
    ) -> list[str]: ...

    def cleanup(self):
        print("Done!")


class OpenAIService(InferenceService):
    def __init__(self):
        self.client = oai_client()

    async def generate(
        self, model: str, messages: list[dict[str, str]], **kwargs
    ) -> list[str]:
        resp = await self.client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        return [c.message.content for c in resp.choices]


class TogetherService(OpenAIService):
    def __init__(self):
        with open("together-api-key") as file:
            self.client = AsyncOpenAI(
                api_key=file.read().strip(), base_url="https://api.together.xyz/v1"
            )


class VLLMService(OpenAIService):
    def __init__(self, model: str):
        port = int(os.environ["VLLM_PORT"])
        self.client = AsyncOpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")


class CohereService(InferenceService):
    def __init__(self):
        with open("cohere-api-key") as file:
            self.client = cohere.AsyncClientV2(file.read().strip())

    async def generate(
        self, model: str, messages: list[dict[str, str]], n=1, **kwargs
    ) -> list[str]:
        responses = []
        for _ in range(n):  # Cohere's API does not support parallel generation
            resp = await self.client.chat(model=model, messages=messages, **kwargs)
            responses.append(resp.message.content[0].text)
        return responses


class GeminiService(InferenceService):
    def __init__(self):
        with open("gemini-api-key") as file:
            self.client = genai.Client(api_key=file.read().strip())

    async def generate(
        self, model: str, messages: list[dict[str, str]], n=1, max_tokens=512, **kwargs
    ) -> list[str]:
        contents = [
            types.Content(
                parts=[types.Part(text=msg["content"])],
                role="user" if msg["role"] == "user" else "model",
            )
            for msg in messages
        ]
        responses = []
        for _ in range(n):
            resp = await self.client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=max_tokens, **kwargs
                ),
            )
            if resp.candidates:
                responses.append(resp.candidates[0].content.parts[0].text)
            else:
                responses.append("[Blocked]")

        return responses


class AnthropicService(InferenceService):
    def __init__(self):
        self.client = AsyncAnthropicVertex(region="us-east5", project_id="GOOGLE-CLOUD-PROJECT-ID")

    async def generate(
        self, model: str, messages: list[dict[str, str]], n=1, **kwargs
    ) -> list[str]:
        responses = []
        for _ in range(n):
            if messages[0]["role"] == "system":
                resp = await self.client.messages.create(
                    system=messages[0]["content"],
                    model=model,
                    messages=messages[1:],
                    **kwargs,
                )
                responses.append(resp.content[0].text)
            else:
                resp = await self.client.messages.create(
                    model=model, messages=messages, **kwargs
                )
                responses.append(resp.content[0].text)
        return responses


class VertexService(InferenceService):
    def __init__(self):
        self.client, self.last_refreshed = self.refresh_client()

    def refresh_client(self):
        model_location = "us-central1"
        project_id = "GOOGLE-CLOUD-PROJECT-ID"
        credentials, _ = default()
        auth_request = transport.requests.Request()
        credentials.refresh(auth_request)

        client = AsyncOpenAI(
            base_url=f"https://{model_location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{model_location}/endpoints/openapi/chat/completions?",
            api_key=credentials.token,
        )
        return client, time.time()

    async def generate(
        self, model: str, messages: list[dict[str, str]], n=1, **kwargs
    ) -> list[str]:
        responses = []
        for _ in range(n):
            if time.time() - self.last_refreshed > 1800:
                self.client, self.last_refreshed = self.refresh_client()
            resp = await self.client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
            responses.append(resp.choices[0].message.content)
        return responses


class DeepSeekService(OpenAIService):
    def __init__(self):
        with open("openrouter-api-key") as file:
            self.client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1", api_key=file.read().strip()
            )


class TransformersService(InferenceService):
    template_modes = {"default", "colon", "explicit-description", "explicit-assistant"}

    def __init__(
        self,
        model: str,
        stop_tokens: list[str] | None = None,
        template_mode: str = "default",
        # fewshot_messages: Optional[list[dict[str, Any]]] = None,
        # prompt_overrides: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model
        print(f"Loading tokenizer and model for {model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if stop_tokens is None:
            stop_tokens = ["<|end_of_text|>", "<eos>", "<end_of_turn>", "<|eot_id|>", "<|im_end|>'", "User:", "Assistant:", self.tokenizer.eos_token] # need to be overridden for other models
        warnings.warn(f"Using default stop sequences: {stop_tokens}. Consider modifying the 'stop' field if your model expects different stop tokens.")
        self.stop_tokens = stop_tokens

        mode = (template_mode or "default").lower()
        if mode not in self.template_modes:
            raise ValueError(f"Unsupported prompt mode '{template_mode}'. Expected one of {sorted(self.template_modes)}")
        self.template_mode = mode
        # self.prompt_overrides: Dict[str, Any] = dict(prompt_overrides or {})
        # self.fewshot_messages = self._normalize_messages(fewshot_messages)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model, 
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2"  # Use flash attention if available
            )
        except:
            print("Flash attention not available, falling back to eager attention")
            self.model = AutoModelForCausalLM.from_pretrained(
                model, 
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="eager",
                # stop=stop_tokens
            )
        
        # Set pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"Model loaded on device: {self.model.device}")
        print(f"Model memory footprint: {self.model.get_memory_footprint() / 1e9:.2f} GB")
        print(f"Model loaded successfully!")

    @staticmethod
    def _normalize_messages(messages: Optional[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        if not messages:
            return []
        normalized: list[dict[str, Any]] = []
        for idx, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"Few-shot message at index {idx} must be a dict with 'role' and 'content' keys")
            if "role" not in message or "content" not in message:
                raise ValueError(
                    f"Few-shot message at index {idx} missing required 'role'/'content' keys: {message}"
                )
            normalized.append(dict(message))
        return normalized

    @staticmethod
    def _copy_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [dict(msg) for msg in messages]

    @staticmethod
    def _strip_markers_from_message(
        message: dict[str, Any], markers: list[str]
    ) -> dict[str, Any]:
        content = message.get("content")
        if isinstance(content, str):
            updated = content
            for marker in markers:
                updated = updated.replace(marker, "")
            if updated != content:
                new_message = dict(message)
                new_message["content"] = updated
                return new_message
        return dict(message)

    def _apply_prompt_template(
        self,
        messages: list[dict[str, Any]],
        template_mode: Optional[str] = None,
        # prompt_overrides: Optional[Dict[str, Any]] = None,
    ):
        mode = (template_mode or self.template_mode or "default").lower()
        if mode not in self.template_modes:
            raise ValueError(f"Unknown prompt mode '{mode}'")
        # if default, just run through the tokenizer
        if mode == "default":
            additional_args = {}
            if "qwen" in self.tokenizer.name_or_path:
                additional_args = {"enable_thinking": False}
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                **additional_args)
        elif mode == "colon":
            if len(messages) <= 2:
                raise ValueError("Colon prompt mode requires at least 3 messages to work for ICL")
            prompt = ""
            for msg in messages:
                prompt += f"{msg['role'].capitalize()}: {msg['content']}\n\n"
            prompt += "Assistant:"
            return self.tokenizer(prompt, return_tensors="pt")['input_ids']
        elif mode == "explicit-description":
            if len(messages) > 1:
                raise ValueError("Explicit-description prompt mode requires exactly 1 message")
            explicit_messages = [{"role": "description", "content": messages[0]["content"]}]
            input_text = self.tokenizer.messages_to_text(explicit_messages, start_generation=True)
            return self.tokenizer(input_text, return_tensors="pt")['input_ids']
        elif mode == "explicit-assistant":
            explicit_messages = [
                {"role": "description", "content": "You are a helpful AI assistant."},
            ]
            for msg in messages:
                # if role is user, make input, if role is assistant, make output
                if msg["role"] == "user":
                    explicit_messages.append({"role": "input", "content": msg["content"]})
                else:
                    explicit_messages.append({"role": "output", "content": msg["content"]})
            input_text = self.tokenizer.messages_to_text(explicit_messages, start_generation=True)
            return self.tokenizer(input_text, return_tensors="pt")['input_ids']

            


        # overrides: Dict[str, Any] = dict(self.prompt_overrides)
        # if prompt_overrides:
        #     overrides.update(prompt_overrides)

        # template_kwargs: Dict[str, Any] = {
        #     "add_generation_prompt": overrides.get(
        #         "add_generation_prompt", False if mode == "nothink" else True
        #     ),
        #     "return_tensors": overrides.get("return_tensors", "pt"),
        #     "padding": overrides.get("padding", False),
        #     "truncation": overrides.get("truncation", True),
        #     "max_length": overrides.get("max_length", 4000),
        # }

        extra_template_kwargs = overrides.get("template_kwargs")
        if extra_template_kwargs:
            template_kwargs.update(extra_template_kwargs)

        chat_messages = self._copy_messages(messages)

        if mode == "fewshot":
            fewshot_messages = overrides.get("fewshot_messages")
            normalized_shots = (
                self._normalize_messages(fewshot_messages)
                if fewshot_messages is not None
                else self.fewshot_messages
            )
            if normalized_shots:
                chat_messages = self._copy_messages(normalized_shots) + chat_messages
            else:
                warnings.warn(
                    "Few-shot prompt mode selected but no few-shot messages provided.",
                    stacklevel=2,
                )
        elif mode == "nothink":
            drop_system = overrides.get("drop_system_message", True)
            if drop_system:
                chat_messages = [msg for msg in chat_messages if msg.get("role") != "system"]
            strip_markers = overrides.get("strip_markers")
            if strip_markers:
                markers = (
                    [strip_markers]
                    if isinstance(strip_markers, str)
                    else list(strip_markers)
                )
                chat_messages = [
                    self._strip_markers_from_message(msg, markers)
                    for msg in chat_messages
                ]

        return self.tokenizer.apply_chat_template(
            chat_messages,
            **template_kwargs,
        )

    async def generate(
        self, model: str, messages: list[dict[str, str]], n=1, max_tokens=512, temperature=1.0, stop=None, **kwargs
    ) -> list[str]:
        # Run the actual generation in a thread to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_sync, messages, n, max_tokens, temperature, self.stop_tokens, kwargs)
    
    def _generate_sync(self, messages, n, max_tokens, temperature, stop, kwargs):
        generate_kwargs = dict(kwargs) if kwargs else {}
        template_mode_override = generate_kwargs.pop("template_mode", None)
        # prompt_overrides = generate_kwargs.pop("prompt_overrides", None)

        # Apply chat template to convert messages to a prompt
        try:
            inputs = self._apply_prompt_template(
                messages,
                template_mode=template_mode_override,
                # prompt_overrides=prompt_overrides,
            )
        except Exception as e:
            print(f"Chat template failed: {e}, using fallback")
            raise Exception(f"Chat template failed: {e}")

        inputs = inputs.to(self.model.device)

        # Use batch generation for efficiency if n > 1
        if n > 1 and hasattr(self.model, 'generate') and temperature > 0:
            # print(f"Generating {n} responses in batch...")
            with torch.no_grad():
                input_ids = inputs.repeat(n, 1)
                attention_mask = torch.ones_like(input_ids)
                # Generate all responses in a single batch call
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    # max_tokens=max_tokens,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,  # Enable KV cache for faster generation
                    stop_strings=stop,
                    tokenizer=self.tokenizer,
                    **generate_kwargs
                )
                
                # Decode all responses
                responses = []
                input_length = input_ids.shape[1]
                for i in range(n):
                    generated_tokens = outputs[i][input_length:]
                    response = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
                    
                    # Apply stop sequences
                    if stop:
                        for stop_seq in stop:
                            if stop_seq in response:
                                response = response.split(stop_seq)[0]
                    
                    response = response.strip()
                    print(response)
                    responses.append(response)
        else:
            # Sequential generation for n=1 or when batch generation isn't suitable
            responses = []
            for i in range(n):
                print(f"Generating response {i+1}/{n}...")
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=True if temperature > 0 else False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,  # Enable KV cache
                        **generate_kwargs
                    )
                    
                    # Decode only the generated part
                    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                    response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    
                    # Apply stop sequences
                    if stop:
                        for stop_seq in stop:
                            if stop_seq in response:
                                response = response.split(stop_seq)[0]
                    
                    response = response.strip()
                    responses.append(response)
                    print(f"Generated: {response[:100]}...")
        
        return responses
    
    def cleanup(self):
        # Clean up GPU memory
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()
        print("Done!")

# sample_path = None
examples = []

async def run_generation(
    service: InferenceService,
    model: str,
    prompt: str,
    prompt_paraphrases: list[str] | None,
    num_generations: int,
    sampling: str,
    max_retries: int = 10,
) -> list[str]:
    responses = []
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(max_retries):
        try:
            if sampling == "regenerate":
                # parallel generation w/o context
                responses = await service.generate(
                    model=model,
                    messages=messages,
                    max_tokens=512,
                    temperature=1.0,
                    n=num_generations,
                )

            elif sampling == "in-context":
                while len(responses) < num_generations:
                    response = await service.generate(
                        model=model,
                        messages=messages,
                        max_tokens=512,
                        temperature=1.0,
                    )
                    new_response = response[0]
                    responses.append(new_response)
                    messages.append({"role": "assistant", "content": new_response})
                    messages.append(
                        {
                            "role": "user",
                            "content": "Can you generate a different answer?",
                        }
                    )

            elif sampling == "paraphrase":
                assert prompt_paraphrases and len(prompt_paraphrases) == num_generations
                while len(responses) < num_generations:
                    messages = [
                        {"role": "user", "content": prompt_paraphrases[len(responses)]}
                    ]
                    response = await service.generate(
                        model=model,
                        messages=messages,
                        max_tokens=512,
                        temperature=1.0,
                    )
                    new_response = response[0]
                    responses.append(new_response)

            elif sampling == "system-prompt":
                messages = [
                    {
                        "role": "system",
                        "content": "You are a producer of unique answers, and you strive to tell each user a novel answer to their question.",
                    },
                    {"role": "user", "content": prompt},
                ]
                responses = await service.generate(
                    model=model,
                    messages=messages,
                    max_tokens=512,
                    temperature=1.0,
                    n=num_generations,
                )
            elif sampling == "fewshot":
                if len(examples) == 0:
                    raise Exception("Sample path is not set")
                # examples = load_dataset("jsonl", data_files=sample_path, split="train")
                # read in jsonl
                fewshot_messages = []
                for example in examples:
                    fewshot_messages.append({"role": "user", "content": example["prompt"]})
                    fewshot_messages.append({"role": "assistant", "content": example["response"]})
                messages = fewshot_messages + messages
                responses = await service.generate(
                    model=model,
                    messages=messages,
                    max_tokens=512,
                    temperature=1.0,
                    n=num_generations,
                )

            else:
                raise Exception("Unknown mode " + sampling)

            return responses

        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                print(
                    f"Error generating response for prompt '{prompt}' after {max_retries} attempts: {e}",
                    flush=True,
                )
                return []

            # Exponential backoff
            wait_time = min(5 * 2**attempt, 60)  # 5, 10, 20, 40, 60, 60, ... seconds
            print(
                f"Attempt {attempt + 1} failed, retrying in {wait_time} seconds...",
                flush=True,
            )
            await asyncio.sleep(wait_time)


async def process_prompts(
    prompts,
    service,
    model,
    output_file,
    num_generations,
    concurrent_requests,
    sampling,
):
    """Processes all prompts concurrently and writes results to a file."""
    async with aio_open(output_file, "a", buffering=1) as f:
        semaphore = asyncio.Semaphore(concurrent_requests)

        async def process_single_prompt(prompt):
            async with semaphore:
                generations = await run_generation(
                    service,
                    model,
                    prompt["prompt"],
                    prompt.get("prompt_paraphrases"),
                    num_generations,
                    sampling,
                )
                return {
                    "id": prompt["id"],
                    "prompt": prompt["prompt"],
                    "model": model,
                    "generations": generations,
                }

        tasks = [process_single_prompt(prompt) for prompt in prompts]
        for task in tqdm(asyncio.as_completed(tasks), total=len(prompts)):
            result = await task
            await f.write(json.dumps(result) + "\n")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "vllm",
            "openai",
            "together",
            "cohere",
            "gemini",
            "anthropic",
            "vertex",
            "deepseek",
            "transformers",
        ],
        required=True,
        help="Inference service provider (vllm for local server, openai for API, transformers for local HF models, etc.)",
    )
    parser.add_argument("--model", required=True, help="Model to run inference with")
    parser.add_argument(
        "--eval-dir", help="Directory to save evaluation results", required=True
    )
    parser.add_argument(
        "--data",
        default="curated",
        choices=["curated", "wildchat"],
        help="Source of prompts",
    )
    parser.add_argument(
        "--sampling",
        choices=["regenerate", "in-context", "paraphrase", "system-prompt", "fewshot"],
        default="regenerate",
    )
    parser.add_argument(
        "--fewshot-file",
        help="Path to JSON file containing few-shot examples.",
    )
    parser.add_argument(
        "--template-mode",
        # choices=["default", "fewshot", "nothink"],
        choices=["default", "colon", "explicit-description", "explicit-assistant"],
        default="default",
        help="Prompt construction strategy for transformers mode.",
    )
    # parser.add_argument(
    #     "--fewshot-file",
    #     help="JSON file containing chat messages to prepend when prompt mode is fewshot.",
    # )
    # parser.add_argument(
    #     "--nothink-keep-system",
    #     action="store_true",
    #     help="Keep system messages when using the nothink prompt mode.",
    # )
    # parser.add_argument(
    #     "--nothink-strip-markers",
    #     nargs="*",
    #     help="Markers to strip from message content when using nothink prompt mode (defaults to <think> tags).",
    # )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=10,
        help="Number of generations per prompt",
    )
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=10,
        help="Number of concurrent requests",
    )
    args = parser.parse_args()

    # set sample path
    global examples
    fewshot_file = args.fewshot_file
    if fewshot_file:
        with open(fewshot_file, "r") as f:
            examples = [json.loads(line) for line in f]

    dataset = load_dataset("yimingzhang/novelty-bench", split=args.data)
    eval_dir = (
        args.eval_dir if args.eval_dir else os.path.join(f"{args.data}-evals", args.model)
    )
    os.makedirs(eval_dir, exist_ok=True)
    output_file = os.path.join(eval_dir, "generations.jsonl")

    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        dataset_keys = set(dataset["id"])
        existing_output = load_dataset("json", data_files=output_file, split="train")
        existing_output = existing_output.filter(
            lambda x: len(x["generations"]) == args.num_generations
            and x["id"] in dataset_keys
        )

        # Save filtered dataset back to output file
        with open(output_file, "w") as f:
            for item in existing_output:
                f.write(json.dumps(item) + "\n")

        existing_keys = set(existing_output["id"])
        # Filter dataset to only include missing or invalid items
        dataset = dataset.filter(lambda x: x["id"] not in existing_keys)

        if len(dataset) == 0:
            print("All prompts have valid generations. Skipping.")
            return
        else:
            print(f"Generating {len(dataset)} missing or invalid entries.")

    concurrent_requests = args.concurrent_requests
    if args.mode == "vllm":
        service = VLLMService(args.model)
    elif args.mode == "openai":  # openai mode
        service = OpenAIService()
    elif args.mode == "together":
        service = TogetherService()
    elif args.mode == "cohere":
        service = CohereService()
    elif args.mode == "gemini":
        service = GeminiService()
    elif args.mode == "anthropic":
        service = AnthropicService()
    elif args.mode == "vertex":
        service = VertexService()
    elif args.mode == "deepseek":
        service = DeepSeekService()
    elif args.mode == "transformers":
        # prompt_overrides: Dict[str, Any] = {}
        # fewshot_messages = None
        # if args.template_mode == "fewshot":
        #     if args.fewshot_file:
        #         with open(args.fewshot_file) as fh:
        #             try:
        #                 fewshot_messages = json.load(fh)
        #             except json.JSONDecodeError as exc:
        #                 raise ValueError(
        #                     f"Failed to parse few-shot file '{args.fewshot_file}': {exc}"
        #                 ) from exc
        #     else:
        #         warnings.warn(
        #             "Few-shot prompt mode selected but no --fewshot-file provided; proceeding without extra shots.",
        #             stacklevel=2,
        #         )
        # if args.template_mode == "nothink":
        #     prompt_overrides["drop_system_message"] = not args.nothink_keep_system
        #     strip_markers = args.nothink_strip_markers
        #     if strip_markers is None:
        #         strip_markers = ["<think>", "</think>"]
        #     prompt_overrides["strip_markers"] = strip_markers

        service = TransformersService(
            args.model,
            template_mode=args.template_mode,
            # fewshot_messages=fewshot_messages,
            # prompt_overrides=prompt_overrides,
        )
        # Reduce concurrent requests for local inference to avoid memory issues
        concurrent_requests = 1
    else:
        raise Exception(f"unknown service {args.mode}")
    try:
        await process_prompts(
            dataset,
            service,
            args.model,
            output_file,
            args.num_generations,
            concurrent_requests,
            args.sampling,
        )

    finally:
        service.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
