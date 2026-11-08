
from openai import OpenAI
import time

# Connect to the local vLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

MODEL = "Qwen/Qwen3-4B-AWQ"


def main():
    prompt = (
        "/no_think\n"
        "Explain in three sentences how a transformer neural network "
        "uses attention."
    )

    print(f"Sending request to {MODEL}...")
    start = time.time()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )

    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print("MODEL RESPONSE")
    print("=" * 60)

    print(response.choices[0].message.content)

    print("\n" + "=" * 60)
    print("INFERENCE DETAILS")
    print("=" * 60)

    print(f"Model: {response.model}")
    print(f"Time: {elapsed:.2f} seconds")
    print(f"Prompt tokens: {response.usage.prompt_tokens}")
    print(f"Completion tokens: {response.usage.completion_tokens}")
    print(f"Finish reason: {response.choices[0].finish_reason}")


if __name__ == "__main__":
    main()