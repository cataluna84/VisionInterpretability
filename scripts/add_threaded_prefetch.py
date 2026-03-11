"""Add 2-thread prefetch to the DataLoader for overlapping download + GPU.

Architecture:
    Thread 0 (bg): shard 0,2,4,...  -> decode -> preprocess -> queue.put()
    Thread 1 (bg): shard 1,3,5,...  -> decode -> preprocess -> queue.put()
    Main thread:   queue.get() -> GPU inference -> score

Uses threading (not multiprocessing) to avoid Windows pickle issues.
"""
import json
from pathlib import Path

NOTEBOOK_PATH = (
    Path(__file__).parent.parent
    / "notebooks"
    / "Segment_3_canonical.ipynb"
)

NEW_DATASET_CELL = r'''"""ImageNet-1k streaming dataset via WebDataset with 2-thread prefetch."""
import sys
import urllib.request
import threading
import queue

import webdataset as wds
from torch.utils.data import DataLoader
from torchvision import transforms

# Preprocessing (ImageNet normalization)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def _is_not_none(x: tuple | None) -> bool:
    """Filter predicate: returns True if sample is not None."""
    return x is not None


def to_tensor(sample: dict) -> tuple | None:
    """Convert a WDS sample to (tensor, key), or None if no image.

    Tries multiple image extensions for compatibility with different
    WebDataset shards.

    Args:
        sample: WebDataset sample dictionary.

    Returns:
        Tuple of (preprocessed_tensor, sample_key), or None.
    """
    img = None
    for ext in ("jpg", "jpeg", "JPEG", "png"):
        img = sample.get(ext)
        if img is not None:
            break
    if img is None:
        return None
    key = sample.get("__key__", "")
    return preprocess(img), key


# ── Monkey-patch gopen for authenticated HuggingFace URLs ──
# In webdataset 1.0, the gopen module is accessed via sys.modules
# because `wds.gopen` resolves to the function, not the module.
_gopen_module = sys.modules["webdataset.gopen"]
_original_gopen = _gopen_module.gopen


def _gopen_with_hf_auth(url, mode="rb", bufsize=8192, **kw):
    """Open URLs with HuggingFace Bearer auth (no curl needed).

    For huggingface.co URLs, uses urllib with an Authorization header.
    For all other URLs, falls back to the original gopen.
    """
    if isinstance(url, str) and "huggingface.co" in url:
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
        )
        return urllib.request.urlopen(req, timeout=120)
    return _original_gopen(url, mode, bufsize, **kw)


_gopen_module.gopen = _gopen_with_hf_auth
print("Patched webdataset.gopen with HF auth opener.")


# Build shard URLs (direct HTTPS, no pipe:curl)
_BASE = "https://huggingface.co/datasets/timm/imagenet-1k-wds/resolve/main/"
_URLS_TRAIN = [
    f"{_BASE}imagenet1k-train-{i:04d}.tar"
    for i in range(1024)
]

# ── Threaded Prefetch Configuration ──
NUM_DOWNLOAD_THREADS = 2
PREFETCH_QUEUE_SIZE = 4  # max batches buffered ahead


class ThreadedPrefetchLoader:
    """DataLoader that prefetches batches using background threads.

    Splits shards across N threads, each running its own WebDataset
    pipeline. Decoded + preprocessed batches are put into a shared
    queue consumed by the main thread for GPU inference.

    Architecture:
        Thread 0: shards [0, 2, 4, ...]  -> decode -> queue
        Thread 1: shards [1, 3, 5, ...]  -> decode -> queue
        Main:     queue.get() -> GPU inference

    Args:
        urls: List of shard URLs.
        batch_size: Number of images per batch.
        num_threads: Number of background download threads.
        queue_size: Max number of batches to buffer ahead.
    """

    _SENTINEL = object()  # Signals "thread is done"

    def __init__(
        self,
        urls: list[str],
        batch_size: int,
        num_threads: int = NUM_DOWNLOAD_THREADS,
        queue_size: int = PREFETCH_QUEUE_SIZE,
    ):
        self.urls = urls
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.queue_size = queue_size

    def _worker(
        self,
        shard_urls: list[str],
        out_queue: queue.Queue,
        worker_id: int,
    ) -> None:
        """Background worker: stream shards, decode, batch, enqueue."""
        try:
            dataset = (
                wds.WebDataset(
                    shard_urls,
                    handler=wds.handlers.warn_and_continue,
                    empty_check=False,
                )
                .decode("pil", handler=wds.handlers.warn_and_continue)
                .map(to_tensor, handler=wds.handlers.warn_and_continue)
                .select(_is_not_none)
            )
            dl = DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=0,
                pin_memory=False,  # pinning done in main thread
            )
            for batch in dl:
                out_queue.put(batch)  # blocks if queue is full
        except Exception as e:
            print(f"  Worker {worker_id} error: {type(e).__name__}: {e}")
        finally:
            out_queue.put(self._SENTINEL)

    def __iter__(self):
        """Yield batches from background threads via shared queue."""
        out_queue = queue.Queue(maxsize=self.queue_size)

        # Split shards round-robin across threads
        shard_splits = [[] for _ in range(self.num_threads)]
        for i, url in enumerate(self.urls):
            shard_splits[i % self.num_threads].append(url)

        # Start background threads
        threads = []
        for tid in range(self.num_threads):
            t = threading.Thread(
                target=self._worker,
                args=(shard_splits[tid], out_queue, tid),
                daemon=True,
            )
            t.start()
            threads.append(t)

        # Consume batches from queue
        done_count = 0
        while done_count < self.num_threads:
            item = out_queue.get()
            if item is self._SENTINEL:
                done_count += 1
                continue
            yield item

        # Wait for all threads to finish
        for t in threads:
            t.join(timeout=5)


def make_dataloader(batch_size: int = CFG["BATCH_SIZE"]) -> ThreadedPrefetchLoader:
    """Create a threaded prefetch loader (2 download threads + main GPU).

    Args:
        batch_size: Number of images per batch.

    Returns:
        A ThreadedPrefetchLoader that yields (imgs, keys) batches.
    """
    return ThreadedPrefetchLoader(
        urls=_URLS_TRAIN,
        batch_size=batch_size,
        num_threads=NUM_DOWNLOAD_THREADS,
        queue_size=PREFETCH_QUEUE_SIZE,
    )


print(f"Dataset ready: {len(_URLS_TRAIN)} shards, batch_size={CFG['BATCH_SIZE']}")
print(f"Prefetch: {NUM_DOWNLOAD_THREADS} download threads, "
      f"queue_size={PREFETCH_QUEUE_SIZE}")
'''


def main():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Find the dataset cell (contains make_dataloader)
    target_idx = None
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        joined = "".join(cell["source"])
        if "def make_dataloader" in joined:
            target_idx = idx
            break

    if target_idx is None:
        print("ERROR: Could not find the dataset cell.")
        return

    # Split the new cell content into lines for the JSON source array
    lines = NEW_DATASET_CELL.split("\n")
    source_lines = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            source_lines.append(line + "\n")
        elif line:  # last line, only add if non-empty
            source_lines.append(line)

    nb["cells"][target_idx]["source"] = source_lines
    nb["cells"][target_idx]["outputs"] = []
    nb["cells"][target_idx]["execution_count"] = None

    # Also update the markdown cell before it
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "markdown":
            continue
        joined = "".join(cell["source"])
        if "ImageNet-1k via WebDataset" in joined:
            cell["source"] = [
                "## 4. Dataset: ImageNet-1k via WebDataset Streaming\n",
                "\n",
                "Streams directly from HuggingFace Hub using a Python-native URL opener\n",
                "with Bearer token auth -- **no curl or local download required**.\n",
                "\n",
                "### Threaded Prefetch Architecture\n",
                "\n",
                "| Thread | Role |\n",
                "|--------|------|\n",
                "| **Thread 0** (bg) | Download shards 0,2,4,... -> decode -> preprocess -> queue |\n",
                "| **Thread 1** (bg) | Download shards 1,3,5,... -> decode -> preprocess -> queue |\n",
                "| **Main** | queue.get() -> GPU inference -> score |\n",
                "\n",
                "This overlaps network I/O with GPU compute,\n",
                "so the GPU is never starved for data.\n",
                "\n",
                "> **Note**: WebDataset iterators are **single-use**. The `make_dataloader()`\n",
                "> factory function creates a fresh loader for each pipeline pass.\n",
            ]
            break

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print(f"Replaced dataset cell (index {target_idx}) with threaded prefetch.")
    print("  - 2 background threads split shards round-robin")
    print("  - Shared queue (maxsize=4) feeds main GPU thread")
    print("  - Uses threading (not multiprocessing) - no pickle issues")


if __name__ == "__main__":
    main()
