import argparse
import shlex
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a local OpenAI-compatible vLLM server for LLMxCPG-Q."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--model-id",
        type=str,
        help="HF/local model path when using a merged/full model.",
    )
    mode.add_argument(
        "--lora-adapter",
        type=str,
        help="HF/local LoRA adapter path for LLMxCPG-Q.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-Coder-32B-Instruct",
        help="Base model used with --lora-adapter.",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default="LLMxCPG-Q",
        help="Model name exposed by the OpenAI API server.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=9001, help="Server port.")
    parser.add_argument("--dtype", type=str, default="auto", help="vLLM dtype.")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Maximum context length for vLLM.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for multi-GPU runs.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Target GPU memory utilization for vLLM.",
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=64,
        help="Maximum LoRA rank when using --lora-adapter.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass --trust-remote-code to vLLM.",
    )
    parser.add_argument(
        "--extra-vllm-args",
        type=str,
        default="",
        help="Extra raw arguments appended to the vLLM command.",
    )

    args = parser.parse_args()

    if args.lora_adapter and not args.base_model:
        parser.error("--base-model is required when --lora-adapter is used.")

    return args


def build_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--served-model-name",
        args.served_model_name,
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]

    if args.trust_remote_code:
        cmd.append("--trust-remote-code")

    if args.lora_adapter:
        cmd.extend(
            [
                "--model",
                args.base_model,
                "--enable-lora",
                "--max-lora-rank",
                str(args.max_lora_rank),
                "--lora-modules",
                f"{args.served_model_name}={args.lora_adapter}",
            ]
        )
    else:
        cmd.extend(["--model", args.model_id])

    if args.extra_vllm_args.strip():
        cmd.extend(shlex.split(args.extra_vllm_args))

    return cmd


def main() -> None:
    args = parse_args()
    cmd = build_command(args)

    print("Starting local LLMxCPG-Q endpoint...")
    print("Endpoint:", f"http://{args.host}:{args.port}/v1/chat/completions")
    print("Model name for --llm-endpoint:", args.served_model_name)
    print()
    print("Run generate_and_run_queries.py with:")
    print(
        "python generate_and_run_queries.py "
        "-d /path/to/dataset.json "
        "-o /path/to/output_dir "
        "--llm-model-type vLLM "
        f"--llm-endpoint {args.served_model_name} "
        f"--llm-port {args.port}"
    )
    print()
    print("vLLM command:")
    print(" ".join(shlex.quote(part) for part in cmd))

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
