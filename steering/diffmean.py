import argparse
import os
from pathlib import Path
from types import MethodType

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("-l", "--lang", type=str, default="en")
    parser.add_argument(
        "-s", "--save",
        type=str,
        default="llama",
        choices=["llama", "aya"],
        help="Select the token data directory: data_llama or data_aya",
    )
    parser.add_argument(
        "--target_tokens",
        type=int,
        default=10_000_000,
    )
    args = parser.parse_args()

    steering_dir = Path(__file__).resolve().parent
    data_dir = steering_dir / f"data_{args.save}"

    input_path = data_dir / f"flores_{args.lang}.pt"
    if not input_path.exists():
        raise FileNotFoundError(f"Token file not found: {input_path}")

    if torch.cuda.device_count() == 0:
        raise RuntimeError("No CUDA GPU is available.")

    print(f"Model: {args.model}")
    print(f"Input: {input_path}")
    print(f"Output directory: {data_dir}")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
    )

    hf_config = llm.llm_engine.model_config.hf_config
    num_layers = hf_config.num_hidden_layers
    hidden_size = hf_config.hidden_size
    max_length = llm.llm_engine.model_config.max_model_len

    # Find the underlying model and decoder layers.
    worker = (
        llm.llm_engine.model_executor
        .driver_worker.model_runner.model
    )

    if "gemma-3" in args.model.lower():
        layers = worker.model.language_model.model.layers
    else:
        layers = worker.model.layers

    if len(layers) != num_layers:
        raise RuntimeError(
            f"Found {len(layers)} layers, expected {num_layers}. "
            "Check the model architecture before proceeding."
        )

    device = torch.device("cuda")
    activation_sums = torch.zeros(
        num_layers, hidden_size, dtype=torch.float32, device=device
    )
    token_counts = torch.zeros(
        num_layers, dtype=torch.int64, device=device
    )

    def factory(idx):
        def forward_hook(self, positions, hidden_states, residual):
            output_hidden_states, output_residual = self._original_forward(
                positions, hidden_states, residual
            )

            if "aya" in args.model.lower():
                final_output = output_hidden_states
            else:
                final_output = output_hidden_states + output_residual

            final_output = final_output.detach().float()

            if final_output.dim() == 3:
                activation_sums[idx] += final_output.sum(dim=(0, 1))
                token_counts[idx] += (
                    final_output.shape[0] * final_output.shape[1]
                )
            elif final_output.dim() == 2:
                activation_sums[idx] += final_output.sum(dim=0)
                token_counts[idx] += final_output.shape[0]
            else:
                raise RuntimeError(
                    f"Unexpected activation shape: {final_output.shape}"
                )

            return output_hidden_states, output_residual

        return forward_hook

    for i, layer in enumerate(layers):
        layer._original_forward = layer.forward
        layer.forward = MethodType(factory(i), layer)

    try:
        print(f"Loading token IDs from {input_path}")
        token_ids = torch.load(input_path, map_location="cpu")

        if args.target_tokens > 0:
            token_ids = token_ids[:args.target_tokens]

        print(f"Loaded {len(token_ids):,} tokens")

        # Keep only complete sequences.
        usable_tokens = (
            len(token_ids) // max_length
        ) * max_length

        if usable_tokens == 0:
            raise ValueError(
                f"Only {len(token_ids)} tokens available, but "
                f"model max length is {max_length}."
            )

        input_ids = token_ids[:usable_tokens].reshape(
            -1, max_length
        )

        print(
            f"Processing {input_ids.size(0)} sequences "
            f"of length {max_length}"
        )

        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0,
        )

        # vLLM executes the model and triggers the patched forwards.
        llm.generate(
            prompt_token_ids=input_ids.tolist(),
            sampling_params=sampling_params,
        )

        if torch.any(token_counts == 0):
            raise RuntimeError(
                "Some layers captured no activations. "
                "Check the vLLM model path and forward hooks."
            )

        mean_activations = (
            activation_sums / token_counts.unsqueeze(1).float()
        )

        result = {
            "steering_vectors": mean_activations.cpu(),
            "token_counts": token_counts.cpu(),
            "num_tokens": usable_tokens,
            "language": args.lang,
            "model": args.model,
            "method": "mean_activation",
        }

        output_path = data_dir / f"mean.{args.lang}.pt"
        torch.save(result, output_path)

        print(f"Saved mean activations to: {output_path}")
        print(f"Tensor shape: {mean_activations.shape}")

    finally:
        # Restore original layer forwards.
        for layer in layers:
            if hasattr(layer, "_original_forward"):
                layer.forward = layer._original_forward
                del layer._original_forward


if __name__ == "__main__":
    main()
