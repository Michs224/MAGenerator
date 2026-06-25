"""P0 (b) — varian DoRA dari champion.

Recipe PERSIS champion (LP-FT + all-layer LoRA r16/α32 Q/K/V, patience 5, dst), cuma
USE_DORA=True. Reuse seluruh logika `p0_champion_multiseed` (DRY) — cuma override 2 hal:
  USE_DORA=True  +  OUTPUT_ROOT terpisah (biar vanilla champion tidak ketimpa).

Seed 42 TIDAK punya artefak DoRA (champion kanonik = vanilla LoRA, BUKAN DoRA) → DoRA latih
SEMUA 5 seed (42,0,1,2,3) FRESH, biar jumlah seed adil dgn vanilla-CH & FT. Banding via
p0_aggregate.py (blok DoRA otomatis muncul kalau folder ada).

Jalankan dari root:  uv run python scripts/p0_multiseed/p0_champion_dora_multiseed.py
"""
import p0_champion_multiseed as champ

# Override 3 knob; sisanya identik champion
champ.USE_DORA = True
champ.OUTPUT_ROOT = "outputs/p0-champion-dora-multiseed"
champ.SEEDS = [42, 0, 1, 2, 3]   # DoRA tak punya artefak 42 utk reuse -> latih semua 5 fresh

if __name__ == "__main__":
    print(f"[DoRA variant] USE_DORA={champ.USE_DORA} | out={champ.OUTPUT_ROOT} | seeds={champ.SEEDS}")
    champ.main()
