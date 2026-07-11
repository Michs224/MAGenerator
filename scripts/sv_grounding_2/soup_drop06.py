"""Model soup antar-seed di atas pissa-drop06-full (weight averaging -> 1 model deploy).

KONTEKS: drop06 = config pertama lolos SEMUA 7 scope (F1>=80 & gap<10) di MEAN 5-seed, TAPI per-seed
rapuh (cuma seed 2 lolos single-seed; deploy by-median-val=seed42 -> ban gap 10.77 lewat tipis). Ensemble
5-model DITOLAK (5x inference utk SV live). Soup = rata-rata BOBOT 5 seed -> 1 model fisik (1x inference)
yang ngejar benefit variance-reduction ensemble sekaligus MENGHAPUS dilema seleksi-seed.

MEKANISME: checkpoint = state_dict BertLinearProbe (LoRA unmerged, init pissa). Backbone residual (base_layer)
IDENTIK antar seed (PiSSA full-SVD deterministik + backbone beku) -> soup valid ASAL merge per-seed dulu
(rata-rata lora_A/lora_B mentah SALAH krn B@A nonlinear). Per modul QKV: W = base_layer + 2.0*(B@A), scaling=alpha/r.
Head classifier ikut dirata-rata.

Yang dihitung per bahasa: individual per-seed (verifikasi vs logits ter-dump), seed-mean, median-val,
ensemble mean-softmax (upper-bound analisis), UNIFORM soup (rata-rata semua 5), GREEDY soup (Wortsman: nambah
seed satu-satu kalau naikin val-F1), + basin-check (val-F1 midpoint top-2 seed). Gate deploy: semua 7 >=80 & gap<10.

Jalankan dari root:  uv run python scripts/sv_grounding_2/soup_drop06.py
"""
import os
import gc
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
import safetensors.torch as st
from transformers import AutoConfig, BertTokenizerFast, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

MODEL_CHECKPOINT = "LazarusNLP/NusaBERT-large"
SCOPE = ["ace", "ban", "bjn", "jav", "mad", "min", "sun"]
SEEDS = [42, 0, 1, 2, 3]
ROOT = "outputs/pissa-drop06-full"
OUT_DIR = "outputs/soup-drop06"
SCALING = 32.0 / 16.0        # alpha / r (drop06: alpha=32, r=16, no rslora)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAXLEN = 128
GATE_F1 = 80.0
GATE_GAP = 10.0


def ckpt_path(seed, lang):
    return f"{ROOT}/seed_{seed}/nusabert-large-{lang}/best/model.safetensors"


def dumped_logits_dir(seed, lang):
    return f"{ROOT}/seed_{seed}/nusabert-large-{lang}"


class BertLinearProbe(torch.nn.Module):
    """Replika PERSIS champion_p1_base.BertLinearProbe (plain, tanpa PEFT) buat eval bobot merged."""
    def __init__(self, config, num_labels):
        super().__init__()
        self.config = config
        self.bert = BertModel(config, add_pooling_layer=False)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kw):
        out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return SequenceClassifierOutput(logits=self.classifier(pooled))


def merge_to_plain(peft_sd):
    """PEFT-wrapped BertLinearProbe state_dict -> plain merged state_dict (fp32, CPU).
    QKV: weight = base_layer.weight + SCALING*(lora_B @ lora_A). Strip 'base_model.model.' prefix."""
    out = {}
    for k, v in peft_sd.items():
        if ".lora_A." in k or ".lora_B." in k:
            continue
        if k.endswith(".base_layer.weight"):
            mod = k[: -len(".base_layer.weight")]
            A = peft_sd[mod + ".lora_A.default.weight"].float()
            B = peft_sd[mod + ".lora_B.default.weight"].float()
            nk = mod.replace("bert.base_model.model.", "bert.") + ".weight"
            out[nk] = v.float() + SCALING * (B @ A)
        elif k.endswith(".base_layer.bias"):
            mod = k[: -len(".base_layer.bias")]
            nk = mod.replace("bert.base_model.model.", "bert.") + ".bias"
            out[nk] = v.float()
        elif k.startswith("classifier"):
            out[k] = v.float()
        else:
            out[k.replace("bert.base_model.model.", "bert.")] = v.float()
    return out


def average_sds(sds):
    """Rata-rata list state_dict (semua key sama)."""
    keys = sds[0].keys()
    return {k: torch.stack([sd[k] for sd in sds], 0).mean(0) for k in keys}


def build_eval_model(config, num_labels):
    m = BertLinearProbe(config, num_labels)
    m.to(DEVICE).eval()
    return m


@torch.no_grad()
def predict_logits(model, sd, tokenizer, texts, bs=64):
    """Load sd ke model (strict -> raise kalau key mismatch), kembalikan logits (np)."""
    model.load_state_dict(sd, strict=True)
    logits = []
    for i in range(0, len(texts), bs):
        batch = tokenizer(list(texts[i:i + bs]), padding=True, truncation=True,
                          max_length=MAXLEN, return_tensors="pt").to(DEVICE)
        logits.append(model(**batch).logits.float().cpu().numpy())
    return np.concatenate(logits, 0)


def macro_f1(logits, y):
    return f1_score(y, logits.argmax(1), average="macro") * 100.0


def load_lang_data(lang):
    d = f"data/nusax_senti/{lang}"
    dfs = {s: pd.read_csv(f"{d}/{s}.csv") for s in ["train", "valid", "test"]}
    label_list = sorted(dfs["train"]["label"].unique().tolist())
    l2i = {v: i for i, v in enumerate(label_list)}
    for s in dfs:
        dfs[s]["y"] = dfs[s]["label"].map(l2i)
    return dfs, label_list


def ensemble_meansoftmax_test(lang):
    """Baseline upper-bound: rata-rata softmax logits test 5 seed yang SUDAH ke-dump."""
    probs, y = None, None
    for s in SEEDS:
        lg = np.load(f"{dumped_logits_dir(s, lang)}/test_logits.npy")
        y = np.load(f"{dumped_logits_dir(s, lang)}/test_labels.npy")
        p = torch.softmax(torch.tensor(lg), -1).numpy()
        probs = p if probs is None else probs + p
    return macro_f1(probs, y)


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_CHECKPOINT)
    config = AutoConfig.from_pretrained(MODEL_CHECKPOINT)
    config.num_labels = 3
    config.hidden_dropout_prob = 0.06
    config.attention_probs_dropout_prob = 0.06
    eval_model = build_eval_model(config, 3)

    all_results = {}
    print(f"CUDA={torch.cuda.is_available()} device={DEVICE} scaling={SCALING} scope={SCOPE}")

    for li, lang in enumerate(SCOPE):
        dfs, label_list = load_lang_data(lang)
        tr_txt, tr_y = dfs["train"]["text"].tolist(), dfs["train"]["y"].to_numpy()
        va_txt, va_y = dfs["valid"]["text"].tolist(), dfs["valid"]["y"].to_numpy()
        te_txt, te_y = dfs["test"]["text"].tolist(), dfs["test"]["y"].to_numpy()

        # --- merge 5 seed -> plain state dict + verifikasi vs logits ter-dump ---
        merged = {}
        indiv = {}
        base_ref = None
        for s in SEEDS:
            psd = st.load_file(ckpt_path(s, lang))
            # cek base_layer identik antar seed (validasi asumsi soup)
            bl = psd["bert.base_model.model.encoder.layer.0.attention.self.query.base_layer.weight"].float()
            if base_ref is None:
                base_ref = bl
                base_maxdiff = 0.0
            else:
                base_maxdiff = max(base_maxdiff, (bl - base_ref).abs().max().item())
            msd = merge_to_plain(psd)
            merged[s] = msd
            del psd
            # verifikasi: model merged reproduksi val/test logits ter-dump
            va_lg = predict_logits(eval_model, msd, tokenizer, va_txt)
            te_lg = predict_logits(eval_model, msd, tokenizer, te_txt)
            va_f1, te_f1 = macro_f1(va_lg, va_y), macro_f1(te_lg, te_y)
            dumped_te = np.load(f"{dumped_logits_dir(s, lang)}/test_logits.npy")
            repro_diff = float(np.abs(te_lg - dumped_te).max()) if dumped_te.shape == te_lg.shape else -1.0
            indiv[s] = {"val_f1": va_f1, "test_f1": te_f1, "repro_maxdiff": repro_diff}
            # FAIL-FAST: merged model HARUS reproduksi logits ter-dump (validasi merge math + key mapping)
            if li == 0 and repro_diff > 0.5:
                raise SystemExit(f"MERGE SALAH: {lang} seed{s} repro_maxdiff={repro_diff:.3f} (logits merged != ter-dump). "
                                 "Cek scaling/key-mapping sebelum lanjut.")

        seed_mean_test = float(np.mean([indiv[s]["test_f1"] for s in SEEDS]))
        median_val_seed = sorted(SEEDS, key=lambda s: indiv[s]["val_f1"])[len(SEEDS) // 2]

        def eval_full(sd):
            tr = macro_f1(predict_logits(eval_model, sd, tokenizer, tr_txt), tr_y)
            va = macro_f1(predict_logits(eval_model, sd, tokenizer, va_txt), va_y)
            te_lg = predict_logits(eval_model, sd, tokenizer, te_txt)
            te = macro_f1(te_lg, te_y)
            p, r, f, sup = precision_recall_fscore_support(
                te_y, te_lg.argmax(1), labels=[0, 1, 2], zero_division=0)
            per_class = {label_list[i]: {"P": float(p[i]), "R": float(r[i]), "F1": float(f[i])} for i in range(3)}
            return {"train_f1": tr, "val_f1": va, "test_f1": te, "gap": tr - te, "per_class": per_class}

        # --- UNIFORM soup (semua 5) ---
        uniform_sd = average_sds([merged[s] for s in SEEDS])
        uniform = eval_full(uniform_sd)

        # --- GREEDY soup (Wortsman): urut by val-F1, nambah kalau naikin val ---
        order = sorted(SEEDS, key=lambda s: -indiv[s]["val_f1"])
        soup_seeds = [order[0]]
        best_val = indiv[order[0]]["val_f1"]
        greedy_log = [{"seed": order[0], "action": "start", "val_f1": best_val}]
        for s in order[1:]:
            cand = average_sds([merged[j] for j in soup_seeds + [s]])
            v = macro_f1(predict_logits(eval_model, cand, tokenizer, va_txt), va_y)
            if v >= best_val:
                soup_seeds.append(s)
                best_val = v
                greedy_log.append({"seed": s, "action": "ADD", "val_f1": v})
            else:
                greedy_log.append({"seed": s, "action": "skip", "val_f1": v})
        greedy_sd = average_sds([merged[s] for s in soup_seeds])
        greedy = eval_full(greedy_sd)

        # --- basin check: midpoint 2 seed val-terbaik ---
        top2 = order[:2]
        mid_sd = average_sds([merged[top2[0]], merged[top2[1]]])
        mid_val = macro_f1(predict_logits(eval_model, mid_sd, tokenizer, va_txt), va_y)
        basin = {"top2_seeds": top2, "endpoint_val": [indiv[top2[0]]["val_f1"], indiv[top2[1]]["val_f1"]],
                 "midpoint_val": mid_val}

        ens_test = ensemble_meansoftmax_test(lang)

        all_results[lang] = {
            "individual": indiv, "seed_mean_test": seed_mean_test, "median_val_seed": median_val_seed,
            "median_val_test": indiv[median_val_seed]["test_f1"],
            "ensemble_meansoftmax_test": ens_test,
            "uniform_soup": uniform, "greedy_soup": greedy, "greedy_seeds": soup_seeds,
            "greedy_log": greedy_log, "basin": basin, "base_layer_maxdiff_across_seeds": base_maxdiff,
        }
        print(f"\n[{lang}] base_layer maxdiff antar-seed={base_maxdiff:.2e} (harus ~0 = PiSSA det + beku)")
        print(f"  indiv test: " + " ".join(f"s{s}={indiv[s]['test_f1']:.2f}(repro{indiv[s]['repro_maxdiff']:.1e})" for s in SEEDS))
        print(f"  seed-mean={seed_mean_test:.2f} | median-val(seed{median_val_seed})={indiv[median_val_seed]['test_f1']:.2f} | ensemble={ens_test:.2f}")
        print(f"  UNIFORM soup: test={uniform['test_f1']:.2f} gap={uniform['gap']:.2f}")
        print(f"  GREEDY soup {soup_seeds}: test={greedy['test_f1']:.2f} gap={greedy['gap']:.2f}")
        print(f"  basin midpoint val={mid_val:.2f} vs endpoints {basin['endpoint_val'][0]:.2f}/{basin['endpoint_val'][1]:.2f}")

        del merged, uniform_sd, greedy_sd, mid_sd
        gc.collect()

    with open(f"{OUT_DIR}/soup_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # --- ringkasan gate ---
    print("\n" + "=" * 78)
    for method, key in [("median-val (deploy skrg)", "median"), ("UNIFORM soup", "uniform_soup"),
                        ("GREEDY soup", "greedy_soup"), ("ensemble (analisis)", "ensemble")]:
        print(f"\n### {method}")
        print(f"{'lang':5} {'test':>7} {'gap':>7} {'>=80':>5} {'gap<10':>7}")
        n_pass = 0
        for lang in SCOPE:
            r = all_results[lang]
            if key == "median":
                te, gp = r["median_val_test"], None
            elif key == "ensemble":
                te, gp = r["ensemble_meansoftmax_test"], None
            else:
                te, gp = r[key]["test_f1"], r[key]["gap"]
            ok80 = "Y" if te >= GATE_F1 else "N"
            okgap = ("Y" if gp < GATE_GAP else "N") if gp is not None else "-"
            passed = te >= GATE_F1 and (gp is None or gp < GATE_GAP)
            n_pass += passed
            print(f"{lang:5} {te:7.2f} {(f'{gp:7.2f}' if gp is not None else '      -')} {ok80:>5} {okgap:>7}")
        print(f"  -> {n_pass}/7 lolos")
    print(f"\nJSON -> {OUT_DIR}/soup_results.json")


if __name__ == "__main__":
    run()
