# Experiment log

One row per experiment. WER is dev-set corpus WER with the official normalizer.

| id | date | model | lang | beam | WER all | WER javind | WER ind | runtime | commit | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| B002 | 2026-09-20 | large-v3-turbo | id | 5 | 0.2524 | 0.2575 | 0.1556 | 510.9s (372 clips, GPU float16) | 2422cff | first turbo baseline, zero-shot |
| B003 | 2026-09-20 | large-v3-turbo | id | 5 | 0.2544 | 0.2591 | 0.1638 | 5588s (372 clips, Docker CPU int8) | fd4e291 | same zip as submitted; offline Docker run, 0 failures, +0.2 vs B002 from int8 |
| B004 | 2026-09-21 | large-v3-turbo | auto | 5 | 0.2524 | 0.2575 | 0.1556 | 642.9s (372 clips, GPU float16) | fd4e291 | predictions byte-identical to B002: auto-detect picks Indonesian on all 372 clips |
| B005 | 2026-09-21 | large-v3-turbo | jw | 5 | 0.2857 | 0.2925 | 0.1533 | 714.3s (372 clips, GPU float16) | fd4e291 | Javanese token hurts: I=1000 vs 307; 3 clips degenerate into number-counting loops ("2 3 3 4 4 5 5 ... 13 13 13") adding ~665 insertions; D down 1057->904; ind marginally better |
| B006 | 2026-09-21 | large-v3 (full) | id | 5 | 0.2857 | 0.2905 | 0.1916 | 1078.9s (372 clips, GPU float16) | fd4e291 | worse than turbo on every count (S 3434 vs 3050, D 1167 vs 1057, I 394 vs 307); 2x hallucinated "Terima kasih" tails (42 vs 21 clips); 2.1x slower; 3 GB zip. Stay on turbo. |
| B007 | 2026-09-21 | large-v3 (full) | jw | 5 | 0.2851 | 0.2898 | 0.1940 | 1292.6s (372 clips, GPU float16) | fd4e291 | same as B006 within noise; no counting loops this time but I=469. Sweep conclusion: turbo + id (B002) is the best zero-shot Whisper config. |
| B008 | 2026-09-21 | Qwen3-ASR-1.7B | Indonesian | - | 0.2537 | 0.2590 | 0.1521 | 1734.4s (372 clips, GPU bf16, batch=1) | fd4e291 | ties turbo (+0.13 pt, not the >2 pt margin needed to switch base model); fewer deletions (D=633 vs 1057) but more insertions (I=464 vs 307); 3.4x slower per clip than turbo beam-5. Stay on Whisper turbo for the submission; keep in mind as a fine-tuning base candidate only if Whisper LoRA plateaus. |
| F001 | 2026-09-21 | turbo + LoRA v1 (Jember only, r=32, 3 ep, best=ep2) | id | 5 | 0.2959 | 0.2976 | 0.2625 | HF backend, adapter merged in memory | - | Jember-val WER 0.259/0.236/0.248 per epoch (ep3 regressed, ep2 kept). Raw output WORSE than B002: 1331 diacritic tokens (nèng, akèh) vs 53 in refs, 0 in B002. S 3050->3991, I 307->566, D 1057->616. |
| F001 | 2026-09-21 | turbo + LoRA v1, CT2 export (`runs/ct2/lora_v1`) | id | 5 | 0.2850 | 0.2870 | 0.2474 | GPU float16 | - | CT2 1.1 pt better than HF (D 616->366): backend gap larger than the 0.5 pt the plan allows for, worth a look but not blocking. |
| F001s | 2026-09-22 | same CT2 output + `postprocess` (strip diacritics) | id | 5 | **0.2362** | 0.2377 | 0.2067 | offline re-score | - | -1.6 pt vs B002 on full dev, all from javind (0.2575->0.2377). ind regresses 0.1556->0.2067 (Jember-only training pulls Indonesian speech toward Javanese vocabulary). On convo 2 alone: 0.2455 vs B002 0.2292, i.e. WORSE; the whole gain is on convo 5. Not submitted. |

## Error categories (B002, large-v3-turbo, lang=id, all 372 dev clips)

Counted by word alignment (jiwer) on normalized text, not by listening. Totals: S=3050, D=1057, I=307 over 17485 ref words.

| category | errors | share of WER | notes |
|---|---|---|---|
| Javanese function words rendered as Indonesian | 579 S + 198 D = 777 | 4.4 pts | kene->ini/kini, kui/kuwi->ini/itu, dadi->jadi, karo->kalau, neng->di, sing->yang, ora->gak, wis->sudah; plus Jv suffix -e/-né -> -nya (60 S). Fine-tuning target. |
| Other substitutions (mostly Javanese content words) | 1231 S | 7.0 pts | e.g. ngentekke->ngantai, disuwek->disuai, brengsek->brekset. Fine-tuning target. |
| Near-spelling substitutions (same initial, len +-2) | 601 S | 3.4 pts | e.g. akhire->akhirnya, kayane->kayaknya, saiki ok but seneng->senang. Partly Jv, partly ref spelling variants. |
| Case-only mismatches | 331 S | 1.9 pts | Whisper capitalises every segment start; joined text keeps mid-sentence capitals that the official normalizer does not lowercase (it only lowercases after . ! ?). Deterministic fix, see below. |
| Hesitations / `...` tokens | 114 S + 114 D = 228 | 1.3 pts | refs contain 229 `e...`, `ya...`, `apa...` tokens in 111 clips; Whisper emits none. |
| Digits instead of number words | 133 S (+ deletions) | >= 0.8 pts | 65 clips; ref "seratus kilo" / "dua ribu dua tiga", hyp "100 kg" / "2023". Fixable with num2words(lang=id) or suppressing digit tokens. |
| Hallucinated `hai` at clip start | 35 clips | ~0.2 pts | classic Whisper opener; not in any ref. |
| Hallucinated tails (`Terima kasih`, `Terima kasih telah menonton`, `bye`) | 22 clips, 45 words | ~0.3 pts | usually replaces the true trailing words too. |
| Long deletion runs (>= 8 words) | 5 clips, 102 D | 0.6 pts | one clip drops 37 of 88 words; likely fast/overlapping speech. |
| Diacritics only (é/è vs e) | 1 S | 0 | not a problem: normalizer keeps diacritics, refs rarely use them. |
| Proper names | not counted separately | | Solo, Jogja, Magelang, Blora->bulurah, Semarang mostly fine. |

Language split: mean per-utterance WER ind 0.142 vs javind 0.252; corpus WER ind 0.1556 vs javind 0.2575. Almost all of the gap is the Javanese vocabulary, which is what LoRA on Jember must fix.

### Cheap deterministic post-processing (simulated on B002 predictions, not yet in main.py; plan gates this at Task 12 Step 4)

| rule | WER all | WER javind | WER ind |
|---|---|---|---|
| B002 as-is | 0.2524 | 0.2575 | 0.1556 |
| + lowercase a Capitalised word unless first word, ALL-CAPS acronym, or preceded by . ! ? | 0.2435 | 0.2504 | 0.1115 |
| + strip trailing `Terima kasih [telah menonton]` / `bye` / `Sampai jumpa` | 0.2413 | 0.2483 | 0.1069 |
| + strip leading `hai` | 0.2396 | 0.2465 | 0.1057 |
| + digits -> Indonesian words via num2words(lang="id") (split `102-103`, strip thousands separators) | 0.2304 | 0.2368 | 0.1057 |

Total -2.2 pts on all 372 clips (-5.0 pts on ind, all from the case rule). The case rule costs a few proper nouns (e.g. Indonesia x11) but is a clear net win. num2words is not a project dependency yet (`uv run --with num2words` for the simulation); adding it to `submission_src` means vendoring it in the zip since the runtime has no internet. Years like 2023 become "dua ribu dua puluh tiga" while refs write "dua ribu dua tiga" (4 refs); acceptable.

## Platform submissions

Submission zip = `submission_src` packed by `scripts/pack.sh` (large-v3-turbo, lang=id, beam=5, no post-processing yet; matches B002/B003 exactly).

| date | id | type | score | notes |
|---|---|---|---|---|
| 2026-09-21 | id-2277 | smoke test | 0.2304 | ran and scored on the platform's smoke subset |
| 2026-09-21 | id-2278 | full evaluation (baseline) | 0.4697 (public score), rank #90 | 1 of 3 weekly submissions used; next submission available 2026-09-21 UTC |

Leaderboard public score (0.4697) is far worse than dev corpus WER (B003: 0.2544 all, Docker CPU int8 — the same zip). Either the hidden test set is harder/more mismatched than dev, or the platform's WER computation differs from `lit.wer`/`third_party/score.py` in some way not yet identified. Needs investigation before using dev WER deltas as a reliable proxy for leaderboard deltas.

Scorer check (2026-09-21): `third_party/score.py` computes `jiwer.wer(normalized_refs, normalized_preds)` over all rows, i.e. corpus WER, the same as `lit.wer.corpus_wer`, and `tests/test_normalize.py` proves the normalizer matches. So the gap is not a scoring mismatch; the hidden test set is harder than dev (likely more Javanese-heavy or a different speaker mix). Dev deltas should still rank configurations, but the absolute level will not transfer.

## Fine-tuning runs (LoRA r=32 on large-v3-turbo, lang token id, lr 1e-4, 3 epochs, early stopping patience 1)

| run | train data | val (in-training metric) | best val WER | notes |
|---|---|---|---|---|
| lora_turbo_v1 | Jember train (1568 chunks, 9.0 h) | Jember val, 16 sessions | 0.2360 (ep2) | ep1 0.2588, ep2 0.2360, ep3 0.2478; ep3 eval 2.5x slower (long generations). Dev results: F001 rows above. |
| lora_turbo_v2 | Jember + dev convo 5 (1838 chunks, 10.9 h) | dev convo 2 (65 clips <= 30 s, greedy) | 0.2207 (ep1) | ep2 0.2324 -> early stop. ~4.4 h wall. |
| lora_turbo_v3 | Jember + convo 5 x2 (2108 chunks, 12.7 h) | same | 0.2111 (ep1) | ep2 0.2122 -> early stop. ~3.5 h wall. |

### Honest comparison on dev conversation 2 (78 clips, never in training; beam 5, CT2 float16, diacritics stripped)

| model | WER all | WER ind | WER javind | S / D / I |
|---|---|---|---|---|
| zero-shot turbo (B002 rows) | 0.2292 | 0.1556 | 0.2527 | 618 / 136 / 61 |
| lora_v1 (Jember only) | 0.2455 | 0.2067 | 0.2579 | 753 / 65 / 55 |
| **lora_v2 (Jember + convo 5)** | **0.1834** | **0.1336** | **0.1993** | 545 / 55 / 52 |
| lora_v3 (Jember + convo 5 x2) | 0.1915 | 0.1370 | 0.2089 | 572 / 52 / 57 |

v2 wins by 4.6 pt over zero-shot and improves both languages; oversampling convo 5 (v3) is slightly worse. In-domain dev speech in training is the lever, as the plan predicted. v2 goes into `submission_src/model` (Task 12 step 2); the zero-shot CT2 weights are kept at `runs/ct2/turbo_zeroshot`. Full table with raw v1 row: `predictions/C2_comparison.txt`.

Post-processing check on v2 convo-2 output (Task 12 step 4): case rule 0.1834 -> 0.1853 (hurts: the tuned model already follows the reference casing), strip trailing `terima kasih`/`bye` -0.1 pt (4 words), strip leading `hai` 0, digits: none emitted. Only the diacritics strip stays in `main.py`.

Both v1 and v2/v3 were trained with Jember's diacritics in the targets (`nèng`, `akèh`); the in-training val WERs above therefore include diacritic substitutions and are only comparable to each other, not to `scripts/score.py` numbers. Fixed 2026-09-22 for future runs: `load_manifests` in `scripts/train_lora.py` now strips diacritics from training text (`lit.data.strip_diacritics`); v2/v3 loaded the old code before the fix. At inference, `submission_src/main.py` now applies `postprocess` (strip diacritics) inside `transcribe`; it is a no-op on the zero-shot model's output and worth -5 pt on Jember-tuned models. Honest comparison of zero-shot / v1 / v2 / v3 on convo 2 (beam 5, CT2, diacritics stripped) is in `predictions/C2_comparison.txt` once `runs/overnight2.sh` finishes.
