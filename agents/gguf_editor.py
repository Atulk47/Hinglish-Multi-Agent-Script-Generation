"""
agents/gguf_editor.py  -  the slang editor as a 4-bit GGUF model on llama.cpp (CPU).

Same trained model (Qwen2.5-1.5B + the merged LoRA adapter), same prompt, greedy decoding - but quantized to 4 bits and run by llama.cpp's
optimised CPU kernels instead of PyTorch's emulated bfloat16. Typically 10-20x faster per edit on a laptop CPU with a few GB less RAM.
Produced by  tools/export_gguf.py.
"""
from __future__ import annotations
import os


class GGUFEditor:
    # (context, batch) ladder: llama.cpp allocates a (batch x vocab x 4 bytes) logits buffer - 150 MB at batch 256 - which is the first thing to fail on
    # a nearly full machine. A prompt of ~250 tokens needs a context of 512 at most.
    LADDER = ((1024, 256), (768, 128), (512, 64), (512, 32))

    def __init__(self, path: str, n_threads: int | None = None, n_ctx: int = 1024):
        from llama_cpp import Llama
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.path = path
        last = None
        for ctx, batch in self.LADDER:
            try:
                self.llm = Llama(model_path=path, n_ctx=min(ctx, n_ctx), n_threads=n_threads or max(4, (os.cpu_count() or 8) - 2),
                                 n_batch=batch, verbose=False, seed=7, use_mmap=True)
                self.config = {"n_ctx": min(ctx, n_ctx), "n_batch": batch}
                if (ctx, batch) != self.LADDER[0]:
                    print(f"GGUFEditor: low memory - loaded with context {ctx}, batch {batch}", flush=True)
                return
            except Exception as e:                               # MemoryError / "Unable to allocate" -> try a lighter setting
                last = e
                import gc
                gc.collect()
        raise RuntimeError(f"GGUF editor could not be loaded even with the smallest settings: {last}")

    def n_tokens(self, text: str) -> int:
        return len(self.llm.tokenize(text.encode("utf-8"), add_bos=False))

    def generate(self, system: str, user: str, max_new: int) -> str:
        r = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0, max_tokens=int(max_new), stop=["<|im_end|>", "<|endoftext|>"])
        return (r["choices"][0]["message"]["content"] or "").strip()
