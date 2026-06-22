"""P0 runner — jalankan SELURUH eksperimen P0 berurutan dengan SATU command.

Urutan: FT multi-seed -> Champion multi-seed -> Aggregate.
Tiap tahap = proses terpisah (isolasi memori GPU). Berhenti kalau ada yang gagal.
Resumable: (seed,bahasa) yang sudah selesai otomatis di-skip.

Jalankan:  uv run python scripts/run_p0.py
"""
import subprocess
import sys
import os

STEPS = [
    ("FT multi-seed",        "scripts/p0_ft_multiseed.py"),
    ("Champion multi-seed",  "scripts/p0_champion_multiseed.py"),
    ("Aggregate + vonis",    "scripts/p0_aggregate.py"),
]


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for i, (name, script) in enumerate(STEPS, 1):
        print(f"\n>>> [{i}/{len(STEPS)}] {name} ({script})", flush=True)
        rc = subprocess.run([sys.executable, script], cwd=root).returncode
        if rc != 0:
            print(f"\n!!! GAGAL di '{name}' (exit {rc}). Berhenti. Perbaiki lalu jalankan lagi (resumable).")
            sys.exit(rc)
    print("\nP0 SELESAI SEMUA. Lihat outputs/p0-aggregate/p0_summary.json")


if __name__ == "__main__":
    main()
