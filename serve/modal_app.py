import modal

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER    = "glen-louis/llama-3.2-3b-sql-qlora"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.7.2",
        "transformers==4.48.3",
        "fastapi[standard]",
        "huggingface_hub[hf_transfer]",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",  # avoid flashinfer JIT (needs nvcc)
    })
)

app = modal.App("sql-qlora", image=image)

SYSTEM_PROMPT = (
    "You are a SQL expert. Given a natural language question and a database schema, "
    "write a SQL query that answers the question. Return only the SQL query with no explanation."
)


def build_prompt(question: str, schema: str) -> str:
    # Llama 3 chat template — hardcoded to avoid tokenizers version issues
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{question}\n\nSchema:\n{schema}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


@app.cls(gpu="T4", secrets=[modal.Secret.from_name("huggingface-secret")], scaledown_window=300)
class SQLModel:

    @modal.enter()
    def load(self):
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        self.llm = LLM(
            model=BASE_MODEL,
            enable_lora=True,
            max_lora_rank=16,
            dtype="half",
            max_model_len=2048,
        )
        self.lora_request = LoRARequest("sql-adapter", 1, ADAPTER)

    @modal.method()
    def generate(self, question: str, schema: str) -> str:
        from vllm import SamplingParams

        prompt  = build_prompt(question, schema)
        params  = SamplingParams(temperature=0, max_tokens=256)
        outputs = self.llm.generate([prompt], params, lora_request=self.lora_request)
        return outputs[0].outputs[0].text.strip()

    @modal.fastapi_endpoint(method="POST")
    def api(self, request: dict) -> dict:
        question = request.get("question", "").strip()
        schema   = request.get("schema", "").strip()
        if not question or not schema:
            return {"error": "question and schema required"}
        return {"sql": self.generate.local(question, schema)}


@app.local_entrypoint()
def main(question: str, schema: str):
    sql = SQLModel().generate.remote(question, schema)
    print(sql)
