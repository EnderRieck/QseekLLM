"""外部基线模型(现成 HF 目录,如 Qwen)在 heldout 过程集上的一次性评测。
复用 async_eval.eval_step 全套(vLLM 生成/判分/dump/md/metrics),
仅把"verl→HF 转换"替换为 symlink(模型已是 HF 格式)。

用法:
  PYTHONPATH=. .venv/bin/python eval/baseline_heldout.py \
    --model /data/zilu/fastrl/checkpoints/external/qwen3_1_7b_base --device 4
产出: <model>/eval_dumps/{metrics.jsonl, step_0/heldout.{jsonl,md}}
"""
import argparse, json, os
import eval.async_eval as ae

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF 模型目录")
    ap.add_argument("--device", type=int, required=True, help="GPU 卡号(PCI 序)")
    ap.add_argument("--heldout", default="eval/heldout.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--out-dir", default="", help="默认 <model>/eval_dumps")
    a = ap.parse_args()

    # 转换步替换为 symlink;keep_hf=True 防 rmtree,评完手动解链
    def _link(step_dir, base, hf_out):
        if os.path.islink(hf_out):
            os.unlink(hf_out)
        os.symlink(step_dir, hf_out)
    ae.convert_to_hf = _link

    args = argparse.Namespace(backend="vllm", max_new_tokens=a.max_new_tokens, chunk_size=256,
                              gpu_mem_util=0.85, progress_every=30, md_per_class=3,
                              keep_hf=True, batch_size=32, token_budget=12000)
    heldout = [json.loads(l) for l in open(a.heldout, encoding="utf-8")]
    out_dir = a.out_dir or os.path.join(a.model, "eval_dumps")
    os.makedirs(out_dir, exist_ok=True)
    m = ae.eval_step(0, a.model, a.model, heldout, out_dir, [a.device], args)
    link = os.path.join(out_dir, "step_0", "_vllm_hf")
    if os.path.islink(link):
        os.unlink(link)
    print(json.dumps(m, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
