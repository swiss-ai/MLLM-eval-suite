Native-head (symmetric 266k vocab) text runs of the released 8B weights —
CONTROL DATA, not column data. The Hub release ships lm_head truncated to the
131,072 text rows; the A/B on identical weights showed the truncated head is
functionally better for generation (gsm8k 79.8 vs 74.2 — the symmetric head
lets visual/audio tokens compete in the argmax and derail chains of thought).
Dashboard text columns use the release presentation (textview extraction).
