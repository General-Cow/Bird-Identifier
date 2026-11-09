from inference.vllm_backend import VLLMBackend


def main():

    backend = VLLMBackend(
        model_name="Qwen/Qwen3-4B-AWQ",
        base_url="http://localhost:8000/v1",
    )

    messages = [
        {
            "role": "user",
            "content": (
                "/no_think\n"
                "Explain in three sentences how a transformer "
                "neural network uses attention."
            ),
        }
    ]

    print("Sending request to vLLM...")

    result = backend.generate(
        messages=messages,
        max_tokens=300,
        temperature=0.2,
    )

    print("\n" + "=" * 60)
    print("MODEL RESPONSE")
    print("=" * 60)

    print(result)


if __name__ == "__main__":
    main()