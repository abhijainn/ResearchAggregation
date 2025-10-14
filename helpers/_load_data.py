import torch
import torch.nn.functional as F
from adapters import AutoAdapterModel
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer
from typing import List

try:
    from app.config import CONFIG  # type: ignore
except ImportError:
    from config import CONFIG  # type: ignore

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

TOPK_INITIAL = CONFIG.search.topk_initial
TOPK_SHOW = CONFIG.search.topk_show

DATA_DIR = CONFIG.paths.data_dir
INDEX_NAME = CONFIG.faiss.index_name
EMBED_MODEL_NAME = CONFIG.vector.embed_model_name
ADAPTER_NAME = CONFIG.vector.adapter_name

class Specter2Embeddings:
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 16):
        if device is None:
            device = CONFIG.runtime.device
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoAdapterModel.from_pretrained(model_name)
        self.model.load_adapter(ADAPTER_NAME, source='hf', load_as='proximity', set_active=True)
        self.model.to(self.device)
        self.model.eval()

    def _prepare_inputs(self, texts: List[str]):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=False,
            max_length=512,
        )

    def _embed_batch(self, inputs):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        normalized = F.normalize(cls_embeddings, p=2, dim=1)
        return normalized

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings: List[List[float]] = []
        total = len(texts)
        use_progress_bar = tqdm is not None and total > self.batch_size
        progress_bar = tqdm(total=total, desc='Embedding texts', unit='doc') if use_progress_bar else None
        log_progress = tqdm is None and total > self.batch_size
        log_step = max(self.batch_size, total // 10 or 1) if log_progress else None
        next_log = log_step if log_progress else None
        if log_progress:
            print(f'Embedding {total} texts...')
        processed = 0
        try:
            with torch.inference_mode():
                for start_idx in range(0, total, self.batch_size):
                    batch = texts[start_idx:start_idx + self.batch_size]
                    inputs = self._prepare_inputs(batch)
                    normalized = self._embed_batch(inputs)
                    embeddings.extend(normalized.cpu().tolist())
                    batch_size = len(batch)
                    processed += batch_size
                    if progress_bar is not None:
                        progress_bar.update(batch_size)
                    elif log_progress and processed >= next_log:
                        print(f'Embedded {processed}/{total} texts')
                        next_log += log_step
        finally:
            if progress_bar is not None:
                progress_bar.close()
        if log_progress:
            if processed < total:
                print(f'Embedded {processed}/{total} texts')
            print('Embedding complete.')
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed_texts([text])[0]


    def __call__(self, texts):
        if isinstance(texts, str):
            return self.embed_query(texts)
        try:
            iterable = list(texts)
        except TypeError as exc:
            raise TypeError('Expected iterable of strings or a single string.') from exc
        return self.embed_documents(iterable)