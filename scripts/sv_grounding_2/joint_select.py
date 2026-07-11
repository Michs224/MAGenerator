"""R3 GATE — seleksi joint multi-seed by per-lang VAL (maximin), penentu lanjut/stop stage-2.

KONTEKS: R2 ladder membuktikan ace di E5 two-stage DI-ANCHOR joint stage-1 (ace joint seed-42 = 79.71,
1-seed = satu lemparan dadu). R3 = tutup variance stage-1: latih joint seed tambahan (joint_multiseed.py),
lalu pilih joint terbaik BY-VAL (bukan test) -> kalau anchor ace membaik, stage-2 di-rerun di atasnya.

ATURAN (PRE-REGISTERED, jangan diubah setelah lihat angka):
1. Winner = joint seed dengan MAXIMIN per-lang val-F1 (worst-language val tertinggi; SATU aturan global,
   bukan pilih-by-ace — ace biasanya jadi worst-lang sehingga tercakup otomatis).
2. GATE LANJUT stage-2 hanya jika: val-ace winner >= val-ace seed-42 + 0.5 (perbaikan anchor nyata).
   Kalau tidak -> R3 STOP: caveat "joint 1-seed" tertutup, verdict E5 final (temuan pendukung), drop06 champion.
3. CATATAN noise: val = 100 contoh/bahasa (SE ~3-4pp) -> seleksi noisy, disclose di thesis. Keputusan tetap
   by-val (test HARAM buat seleksi).

Jalankan dari root (setelah joint_multiseed.py):  uv run python scripts/sv_grounding_2/joint_select.py
"""
import os
import glob
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
import safetensors.torch as st
from transformers import AutoConfig, BertTokenizerFast, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import LoraConfig, get_peft_model

MODEL_CHECKPOINT = "LazarusNLP/NusaBERT-large"
SCOPE = ["ace", "ban", "bjn", "jav", "mad", "min", "sun"]
JOINT_ROOT = "outputs/joint-multilingual-champion"
BASE_SEED = 42            # anchor pembanding (joint yang dipakai E5)
GATE_MARGIN = 0.5         # val-ace winner harus >= val-ace seed42 + margin ini
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAXLEN = 128
# resep joint (harus sama dgn joint_multilingual.py utk rekonstruksi): champion vanilla LoRA
LORA_R, LORA_ALPHA = 16, 32
LORA_TARGET = ["query", "key", "value"]


class MeanPoolClf(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.bert = BertModel(cfg, add_pooling_layer=False)
        self.classifier = torch.nn.Linear(cfg.hidden_size, 3)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kw):
        out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return SequenceClassifierOutput(logits=self.classifier(pooled))


def build_model(cfg):
    m = MeanPoolClf(cfg)
    n = cfg.num_hidden_layers
    lc = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET, lora_dropout=0.1,
                    bias="none", layers_to_transform=list(range(n)), layers_pattern="layer",
                    init_lora_weights=True)
    m.bert = get_peft_model(m.bert, lc)
    return m


@torch.no_grad()
def predict(model, tok, texts, bs=64):
    logits = []
    for i in range(0, len(texts), bs):
        b = tok(list(texts[i:i + bs]), padding=True, truncation=True, max_length=MAXLEN,
                return_tensors="pt").to(DEVICE)
        logits.append(model(**b).logits.float().cpu().numpy())
    return np.concatenate(logits, 0)


def main():
    seeds = sorted(int(os.path.basename(os.path.dirname(os.path.dirname(p))).split("_")[1])
                   for p in glob.glob(f"{JOINT_ROOT}/seed_*/best/model.safetensors"))
    print(f"Joint seeds dengan bobot di disk: {seeds}")
    if BASE_SEED not in seeds:
        raise SystemExit(f"seed {BASE_SEED} (anchor E5) tidak ada bobotnya — cek {JOINT_ROOT}")
    if len(seeds) < 2:
        raise SystemExit("Baru 1 joint seed — jalankan dulu: uv run python scripts/sv_grounding/p1_sweep/joint_multiseed.py")

    tok = BertTokenizerFast.from_pretrained(MODEL_CHECKPOINT)
    cfg = AutoConfig.from_pretrained(MODEL_CHECKPOINT)
    cfg.num_labels = 3
    cfg.hidden_dropout_prob = 0.25
    cfg.attention_probs_dropout_prob = 0.25
    model = build_model(cfg).to(DEVICE).eval()

    # data val per bahasa
    val = {}
    for lang in SCOPE:
        df = pd.read_csv(f"data/nusax_senti/{lang}/valid.csv")
        l2i = {v: i for i, v in enumerate(sorted(df["label"].unique()))}
        val[lang] = (df["text"].tolist(), df["label"].map(l2i).to_numpy())

    rows = {}
    for s in seeds:
        sd = st.load_file(f"{JOINT_ROOT}/seed_{s}/best/model.safetensors")
        model.load_state_dict(sd, strict=True)
        rows[s] = {}
        for lang in SCOPE:
            txt, y = val[lang]
            rows[s][lang] = f1_score(y, predict(model, tok, txt).argmax(1), average="macro") * 100
        del sd

    print(f"\n{'seed':>5} | " + " ".join(f"{l:>6}" for l in SCOPE) + " |  worst | worst-lang")
    stats = {}
    for s in seeds:
        worst_lang = min(SCOPE, key=lambda l: rows[s][l])
        worst = rows[s][worst_lang]
        stats[s] = (worst, worst_lang)
        print(f"{s:>5} | " + " ".join(f"{rows[s][l]:6.2f}" for l in SCOPE) + f" | {worst:6.2f} | {worst_lang}")

    winner = max(seeds, key=lambda s: stats[s][0])            # maximin per-lang val
    base_ace = rows[BASE_SEED]["ace"]
    win_ace = rows[winner]["ace"]
    print(f"\nMAXIMIN winner: seed {winner} (worst-lang val {stats[winner][0]:.2f} @ {stats[winner][1]})")
    print(f"val-ace: winner(seed {winner}) = {win_ace:.2f} vs anchor(seed {BASE_SEED}) = {base_ace:.2f} "
          f"(Δ {win_ace - base_ace:+.2f}, gate butuh >= +{GATE_MARGIN})")

    passed = (winner != BASE_SEED) and (win_ace >= base_ace + GATE_MARGIN)
    if passed:
        ckpt = f"{JOINT_ROOT}/seed_{winner}/best/model.safetensors"
        print(f"\n✅ GATE PASS -> lanjut stage-2 di atas joint seed {winner}.")
        print(f"   Lapor ke asisten: flip JOINT_CKPT twostage.py ke: {ckpt}")
        print(f"   (arm pre-registered: lora_lr2e5ep10, 5-seed x 7 bahasa, OUTPUT_ROOT baru)")
    else:
        print(f"\n❌ GATE STOP -> tidak ada joint dengan anchor-ace lebih baik (+{GATE_MARGIN}). "
              f"R3 selesai: caveat 'joint 1-seed' TERTUTUP, verdict E5 final, drop06 tetap champion.")

    json.dump({"per_lang_val": rows, "winner": winner, "base_seed": BASE_SEED,
               "gate_pass": bool(passed)}, open(f"{JOINT_ROOT}/joint_select_verdict.json", "w"), indent=2)
    print(f"\nverdict -> {JOINT_ROOT}/joint_select_verdict.json")


if __name__ == "__main__":
    main()
