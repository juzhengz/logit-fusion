#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from transformers import AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="Qwen/Qwen2.5-Math-1.5B")
    ap.add_argument("--b", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--outdir", default="tokenizer_alignment_report")
    ap.add_argument("--trust_remote_code", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tok_a = AutoTokenizer.from_pretrained(args.a, trust_remote_code=args.trust_remote_code)
    tok_b = AutoTokenizer.from_pretrained(args.b, trust_remote_code=args.trust_remote_code)

    vocab_a = tok_a.get_vocab()  # token -> id
    vocab_b = tok_b.get_vocab()

    set_a = set(vocab_a.keys())
    set_b = set(vocab_b.keys())

    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)

    common = set_a & set_b
    id_mismatch = []
    for t in common:
        ia, ib = vocab_a[t], vocab_b[t]
        if ia != ib:
            id_mismatch.append((t, ia, ib))
    id_mismatch.sort(key=lambda x: (abs(x[1]-x[2]), x[0]))

    # Special tokens comparison
    special_a = {
        "special_tokens_map": tok_a.special_tokens_map,
        "all_special_tokens": tok_a.all_special_tokens,
        "all_special_ids": tok_a.all_special_ids,
    }
    special_b = {
        "special_tokens_map": tok_b.special_tokens_map,
        "all_special_tokens": tok_b.all_special_tokens,
        "all_special_ids": tok_b.all_special_ids,
    }

    report = {
        "model_a": args.a,
        "model_b": args.b,
        "vocab_size_a": len(vocab_a),
        "vocab_size_b": len(vocab_b),
        "num_only_in_a": len(only_a),
        "num_only_in_b": len(only_b),
        "num_common": len(common),
        "num_id_mismatch_in_common": len(id_mismatch),
        "only_in_a_sample": only_a[:50],
        "only_in_b_sample": only_b[:50],
        "id_mismatch_sample": id_mismatch[:50],
        "special_a": special_a,
        "special_b": special_b,
    }

    (outdir / "tokenizer_diff.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Full mismatch TSV
    with (outdir / "tokenizer_id_mismatch.tsv").open("w", encoding="utf-8") as f:
        f.write("token\tid_a\tid_b\n")
        for t, ia, ib in id_mismatch:
            f.write(f"{t}\t{ia}\t{ib}\n")

    # Full only-in lists
    (outdir / "only_in_a.txt").write_text("\n".join(only_a), encoding="utf-8")
    (outdir / "only_in_b.txt").write_text("\n".join(only_b), encoding="utf-8")

    print(f"[OK] Wrote report to: {outdir.resolve()}")
    print(f"vocab_size: {len(vocab_a)} vs {len(vocab_b)}")
    print(f"only_in_a: {len(only_a)}, only_in_b: {len(only_b)}, id_mismatch: {len(id_mismatch)}")

if __name__ == "__main__":
    main()