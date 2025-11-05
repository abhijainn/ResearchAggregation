from types import SimpleNamespace

import torch


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and hasattr(mps_backend, "is_available") and mps_backend.is_available():
        return "mps"
    return "cpu"


CONFIG = SimpleNamespace(
    llm=SimpleNamespace(
        model_name="gpt-5-nano",
    ),
    rerank=SimpleNamespace(
        cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    ),
    paths=SimpleNamespace(
        data_dir="data",
    ),
    faiss=SimpleNamespace(
        index_name="faiss",
    ),
    search=SimpleNamespace(
        topk_initial=50,
        topk_show=15,
    ),
    vector=SimpleNamespace(
        embed_model_name="allenai/specter2_base",
        adapter_name="allenai/specter2",
    ),
    prompts=SimpleNamespace(
        abstract_system_prompt=(
            "You are a meticulous scientific writing assistant.\n"
            "Write plausible research paper abstracts in 2-3 complete sentences.\n"
            "Mirror the tone of peer-reviewed scientific literature, focusing on objectives, methodology, and key findings.\n"
            "Align your response with the user's claim. Do not attempt to correct the user if their claim is wrong.\n"
            "Avoid citations, hedging, and unnecessary background context.\n"
            "Do not invent overly specific experimental details."
        ),
        abstract_user_prompt_template=(
            "User Claim:\n{claim}\n\n"
            "Write a 1-2 sentence abstract that aligns with the chosen stance."
        ),
    ),
    runtime=SimpleNamespace(
        device=_detect_device(),
    ),
)
