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
| 2026-09-21 | id-2277 | smoke test | 0.2304 | ran and scored on the platform's smoke subset (zero-shot baseline zip) |
| 2026-09-21 | id-2278 | full evaluation (baseline) | 0.4697 (public score), rank #90 | 1 of 3 weekly submissions used; next submission available 2026-09-21 UTC |
| 2026-09-22 | id-2350 | smoke test | 0.1609 | lora_v2 (Jember + convo 5) zip; does not count against the weekly quota |
| 2026-09-22 | id-2353 | full evaluation (lora_v2, Jember + convo 5) | 0.2780 (public score), rank #35 | 2 of 3 weekly submissions used; 1 left, next available 2026-09-22 UTC |
| 2026-09-23 | id-2448 | smoke test | 0.1043 | lora_v5 (convo 5 + TTS Central Javanese, no Jember) zip; does not count against the weekly quota |
| 2026-09-23 | id-2451 | full evaluation (lora_v5, convo 5 + Central Javanese) | 0.2600 (public score), rank #22 | 3 of 3 weekly submissions used; quota empty until 2026-09-28 UTC |

Leaderboard public score (0.4697) is far worse than dev corpus WER (B003: 0.2544 all, Docker CPU int8 — the same zip). Either the hidden test set is harder/more mismatched than dev, or the platform's WER computation differs from `lit.wer`/`third_party/score.py` in some way not yet identified. Needs investigation before using dev WER deltas as a reliable proxy for leaderboard deltas.

lora_v2's full evaluation (0.2780) is -0.1917 vs the baseline (0.4697) and moves rank #90 -> #35. The smoke-test score improved by roughly the same margin (0.2304 -> 0.1609, -0.0695), so the smoke subset direction at least agrees with the full leaderboard result, even though its absolute level is not close to the full-evaluation score.

lora_v5's full evaluation (0.2600, rank #22) is -0.0180 vs v2, i.e. -6.5% relative. **The smoke subset badly over-predicted that gain and convo 2 slightly under-predicted it**, so the smoke score must not be used to size an improvement:

| model | convo 2 | smoke | full test | full / smoke |
|---|---|---|---|---|
| zero-shot baseline | 0.2292 | 0.2304 | 0.4697 | 2.04 |
| lora_v2 | 0.1834 | 0.1609 | 0.2780 | 1.73 |
| lora_v5 | 0.1760 | 0.1043 | 0.2600 | 2.49 |
| v5 vs v2, relative | -4.0% | -35.2% | **-6.5%** | |

The full/smoke ratio moves between 1.73 and 2.49 across three models, so the smoke subset is a pass/fail gate on the zip and a direction check only; its magnitude carries no information about the leaderboard. Held-out convo 2 is the honest predictor: -4.0% there against -6.5% on the test set, the right order of magnitude and conservative. The test set still rewards the Central Javanese data more than convo 2 does (-6.5% vs -4.0%), which is the dialect signal, but by roughly 1.6x, not the 9x the smoke score suggested. Use convo-2 deltas to rank candidates and expect the leaderboard delta to be about 1.5x larger in relative terms.

Scorer check (2026-09-21): `third_party/score.py` computes `jiwer.wer(normalized_refs, normalized_preds)` over all rows, i.e. corpus WER, the same as `lit.wer.corpus_wer`, and `tests/test_normalize.py` proves the normalizer matches. So the gap is not a scoring mismatch; the hidden test set is harder than dev (likely more Javanese-heavy or a different speaker mix). Dev deltas should still rank configurations, but the absolute level will not transfer.

## Fine-tuning runs (LoRA r=32 on large-v3-turbo, lang token id, lr 1e-4, 3 epochs, early stopping patience 1)

| run | train data | val (in-training metric) | best val WER | notes |
|---|---|---|---|---|
| lora_turbo_v1 | Jember train (1568 chunks, 9.0 h) | Jember val, 16 sessions | 0.2360 (ep2) | ep1 0.2588, ep2 0.2360, ep3 0.2478; ep3 eval 2.5x slower (long generations). Dev results: F001 rows above. |
| lora_turbo_v2 | Jember + dev convo 5 (1838 chunks, 10.9 h) | dev convo 2 (65 clips <= 30 s, greedy) | 0.2207 (ep1) | ep2 0.2324 -> early stop. ~4.4 h wall. |
| lora_turbo_v3 | Jember + convo 5 x2 (2108 chunks, 12.7 h) | same | 0.2111 (ep1) | ep2 0.2122 -> early stop. ~3.5 h wall. |
| lora_turbo_v4 | Jember + convo 5 + Central Javanese (3582 chunks, 18.85 h) | same | 0.1846 (ep2) | ep1 0.1927, ep2 0.1846, ep3 0.1890. 2 h 22 min wall at 12.8 s/step (v2's 4.4 h must have shared the GPU with something). Diacritics stripped from targets (post-fix code). |
| lora_turbo_v5 | convo 5 + Central Javanese, no Jember (2014 chunks, 9.81 h) | same | 0.1798 (ep2) | ep1 0.2166, ep2 0.1798, ep3 0.1813. 1 h 21 min wall. |

### Honest comparison on dev conversation 2 (78 clips, never in training; beam 5, CT2 float16, diacritics stripped)

| model | WER all | WER ind | WER javind | S / D / I |
|---|---|---|---|---|
| zero-shot turbo (B002 rows) | 0.2292 | 0.1556 | 0.2527 | 618 / 136 / 61 |
| lora_v1 (Jember only) | 0.2455 | 0.2067 | 0.2579 | 753 / 65 / 55 |
| lora_v2 (Jember + convo 5) | 0.1834 | **0.1336** | 0.1993 | 545 / 55 / 52 |
| lora_v3 (Jember + convo 5 x2) | 0.1915 | 0.1370 | 0.2089 | 572 / 52 / 57 |
| lora_v4 (Jember + convo 5 + Central Javanese) | 0.1797 | 0.1487 | 0.1896 | 535 / 57 / 47 |
| **lora_v5 (convo 5 + Central Javanese, no Jember)** | **0.1760** | 0.1359 | **0.1889** | 497 / 78 / 51 |

v2 wins by 4.6 pt over zero-shot and improves both languages; oversampling convo 5 (v3) is slightly worse. In-domain dev speech in training is the lever, as the plan predicted. v2 went into `submission_src/model` on 2026-09-22 (Task 12 step 2) and was replaced by v5 on 2026-09-23 (see the decision below); the zero-shot CT2 weights are kept at `runs/ct2/turbo_zeroshot`. Full table with raw v1 row: `predictions/C2_comparison.txt`.

**Autoresearch proxy validation (2026-09-24):** the 500-chunk proxy scored v1 0.2081, v2 0.1862, v3 0.1808, and v2/seed-1 0.1831 in 543-549 s per run (noise 0.0031). A definitive 1000-chunk rerun scored v1 0.2101, v2 0.1943, v3 0.1887, and v2/seed-1 0.1907 in 938-946 s (noise 0.0036). Both sizes reliably identify the clearly bad v1 mix, but both rank v3 ahead of v2, opposite the full-run truth (v2 0.1834 < v3 0.1915), and at 1000 chunks the wrong-way gap (0.0056) exceeds seed noise. Conclusion: keep the cheaper 500-chunk proxy only as a coarse regression screen; it cannot gate or rank close configurations, so the overnight loop must rely on decode/post-processing plus 2-4 hypothesis-driven full runs.

Central Javanese runs (2026-09-22 overnight, table `predictions/C2_comparison_v4.txt`): adding the MDC Central Javanese read-speech corpus helps the Javanese side in both runs (javind 0.1993 -> 0.1896 / 0.1889, about -1.0 pt, substitutions 451 -> 430 / 403). Keeping Jember in the mix (v4) costs 1.5 pt on ind (0.1336 -> 0.1487), the same Indonesian regression Jember caused in v1, so v4 misses the decision rule. Dropping Jember (v5) removes that regression (ind 0.1359, +2 errors on 861 words, i.e. noise) and gives the best overall result, 0.1760, -0.74 pt vs v2. v5 trades substitutions for deletions (S 545 -> 497, D 55 -> 78), worth a look in the error report. Conclusion: the dialect hypothesis holds; Jember (East Javanese) is net negative once Central Javanese data is available. v5 passes the overall (>= 0.5 pt) and javind criteria; its ind is 0.0023 above v2, within noise but strictly outside the "not worse on either language" clause, so replacing v2 is a judgment call rather than automatic.

**Decision (2026-09-23): v5 replaces v2 in `submission_src/model`.** Per-clip check of v5 against v2 on the 78 clips: 32 clips better, 16 equal, 30 worse; total errors 652 -> 626; the extra deletions are diffuse (largest per-clip increase +3 words, largest per-clip deletion count 5, no empty or truncated outputs), so it is not a hallucination-stop or truncation failure mode. The ind gap is 115 vs 117 errors on 20 clips with per-clip deltas scattered in both directions. Paired bootstrap over clips (5,000 resamples) puts the v5 - v2 corpus WER difference at -0.0073 with 95% CI [-0.0195, +0.0043], P(v5 worse) = 0.11. The strict "not worse on either language" clause was written to catch v1/v4-style regressions of 1.5 to 5 pt, not a 2-word delta, so it was relaxed for this call; the platform smoke test and full evaluation are the real gate, with v2's CT2 weights kept at `runs/ct2/lora_v2_submitted` (byte-identical to `runs/ct2/lora_v2`) and v2's zip kept as `~/repos/lost-in-transcription-runtime/submission/submission_lora_v2_4103bd10.zip` for re-upload if v5 scores no better. **Gate passed 2026-09-23:** v5's full evaluation scored 0.2600 against v2's 0.2780 and moved rank #38 -> #22, so relaxing the `ind` clause was the right call and the v2 fallback is no longer needed.

What the v4/v5 pair adds to what we know: (1) the dialect hypothesis holds on held-out speech, not just on marker counts: Central Javanese read speech moves javind by -1.0 pt from 8 h of data, and Jember's contribution is now negative on ind and neutral on javind, so East Javanese data is not worth its training time for this dev set; (2) read TTS speech with no code-switching still transfers to conversational Javanese, as long as convo 5 supplies the register; (3) training throughput with an idle GPU is 12.8 s/step, so an 18.85 h mix takes 2 h 22 min and a 9.8 h mix 1 h 21 min, which leaves room for one or two more runs before the 2026-09-30 final freeze; (4) v5 reaches its best in-training val at epoch 2 with epoch 3 flat, so 3 epochs with patience 1 remains the right budget.

Post-processing check on v2 convo-2 output (Task 12 step 4): case rule 0.1834 -> 0.1853 (hurts: the tuned model already follows the reference casing), strip trailing `terima kasih`/`bye` -0.1 pt (4 words), strip leading `hai` 0, digits: none emitted. Only the diacritics strip stays in `main.py`.

### Decoding sweep on v2 (Task 12 step 3, convo 2, CT2)

| config | WER all | ind | javind | S/D/I |
|---|---|---|---|---|
| beam 1 | 0.1912 | 0.1405 | 0.2074 | 566/60/54 |
| beam 5 (kept, `config.json` default) | 0.1834 | 0.1336 | 0.1993 | 545/55/52 |
| beam 8 | 0.1839 | 0.1278 | 0.2019 | 546/54/54 |
| beam 5 + condition_on_previous_text=true | 0.1960 | 0.1289 | 0.2174 | 540/52/105 |

Beam 5 stays; beam 8 ties within noise at 60% more decode time, and `condition_on_previous_text=true` nearly doubles insertions (52->105, cross-segment echo/drift). `config.json` unchanged. `submission_src/model` now holds `lora_v2`'s CT2 weights (moved zero-shot weights to `runs/ct2/turbo_zeroshot` first).

## Freeze (Task 13, 2026-09-22)

Final submission = `submission_src` as of this commit: large-v3-turbo + `lora_turbo_v2` merged, CT2 float16, `config.json` unchanged (lang `id`, beam 5, no conditioning, no VAD), `postprocess` strips diacritics. Expected dev WER 0.1834 on convo 2 (in-training convo 5 makes full-dev numbers meaningless for this model).

Validation before upload: `uv run pytest -q` 24 passed; `scripts/pack.sh --run --small 10` VALIDATION PASSED, zip 1.6 GB (sha256 prefix `4103bd1066ffbaf8`), official image on CPU int8 wrote 10 rows, 0 failures, 119 s. Local Docker cannot use the GPU: the image is CUDA 13 and the host Windows driver 566 (CUDA 12.7) is refused by the NVIDIA container hook; with the check bypassed CTranslate2 still sees 0 CUDA devices. The GPU float16 path is only exercised by the platform smoke test, so check its log for `on cuda (float16)` before the full run.

| date | id | type | score | notes |
|---|---|---|---|---|
| 2026-09-22 | id-2350 | smoke test | 0.1609 | uploaded `~/repos/lost-in-transcription-runtime/submission/submission.zip`; completed in 1h17min |
| 2026-09-22 | id-2353 | full evaluation (lora_v2) | 0.2780 (public score), rank #35 | -0.1917 vs baseline 0.4697; completed in 31min, well under 2h; tag `final-v1` |

## Second freeze (2026-09-23, model lora_turbo_v5)

Final submission candidate = `submission_src` with `lora_turbo_v5` (convo 5 + TTS Central Javanese, no Jember; best epoch 2) merged into large-v3-turbo, CT2 float16, `config.json` unchanged (lang `id`, beam 5, no conditioning, no VAD), `postprocess` strips diacritics. Expected dev WER 0.1760 on convo 2 (ind 0.1359, javind 0.1889). The previous frozen model (v2, zip sha256 prefix `4103bd1066ffbaf8`, tag `final-v1`) stays available as described in the decision paragraph above.

Validation before upload (2026-09-23): `uv run pytest -q` 27 passed; 10-clip GPU round-trip through `submission_src/model` byte-identical to `predictions/C2_lora_v5.csv` (10 of 10); `scripts/pack.sh --run --small 10` VALIDATION PASSED, zip 1.6 GB (sha256 prefix `99c7a8383c5cc016`), official image on CPU int8 wrote 10 rows, 0 failures, 127 s (v2 took 119 s on the same clips). Check the platform smoke log for `on cuda (float16)` before the full run.

| date | id | type | score | notes |
|---|---|---|---|---|
| 2026-09-23 | id-2448 | smoke test | 0.1043 | zip sha256 prefix `99c7a838`; smoke history baseline 0.2304 -> v2 0.1609 -> v5 0.1043. v2's rank had drifted from #35 to #38 by this date. |
| 2026-09-23 | id-2451 | full evaluation (lora_v5) | 0.2600 (public score), rank #22 | -0.0180 vs v2's 0.2780, -0.2097 vs baseline; rank #38 -> #22. Quota now 3 of 3 used, empty until 2026-09-28 UTC. Tag `final-v2`. |

Both v1 and v2/v3 were trained with Jember's diacritics in the targets (`nèng`, `akèh`); the in-training val WERs above therefore include diacritic substitutions and are only comparable to each other, not to `scripts/score.py` numbers. Fixed 2026-09-22 for future runs: `load_manifests` in `scripts/train_lora.py` now strips diacritics from training text (`lit.data.strip_diacritics`); v2/v3 loaded the old code before the fix. At inference, `submission_src/main.py` now applies `postprocess` (strip diacritics) inside `transcribe`; it is a no-op on the zero-shot model's output and worth -5 pt on Jember-tuned models. Honest comparison of zero-shot / v1 / v2 / v3 on convo 2 (beam 5, CT2, diacritics stripped) is in `predictions/C2_comparison.txt` once `runs/overnight2.sh` finishes.

## Dialect mismatch and the Central Javanese corpus (2026-09-22)

Marker-word counts on the transcripts show the dev set is Central Javanese while Jember is East Javanese (Pandhalungan). Counts over dev (18k tokens) vs Jember (68k tokens): `kui/kuwi` 71 vs 311, `iku` 9 vs 2687, `neng` 24 vs 3, `nang` 0 vs 236, `wae` 20 vs 74, `ae` 0 vs 188, `kate/katene/sampeyan` 0 vs 211. Dev speakers also name Solo, Jogja, Magelang, Blora and Semarang. This is the likely reason v1 (Jember only) lost to zero-shot on convo 2 and v2's whole gain came from in-domain convo 5. Plan: `docs/superpowers/plans/central-javanese-data-2026-09-22.md`.

Jember is hosted on MDC (Universitas Gadjah Mada, CC-BY-NC-SA-4.0), so the DrivenData reminder of 2026-09-22 about publishing non-MDC training data does not apply to the current submission.

**Central Javanese corpus** = MDC "TTS Central Javanese" (`mozilladatacollective.com/datasets/cml5bn4k900aame07u0rwidcg`, CC-BY-SA-4.0, Semarang-dialect ngoko read speech, everyday topics). Downloaded 2026-09-22 as `RECORDING TTS.tar.gz` (461 MB, a tar of one zip of 92 zips). Layout after extraction: 92 folders `recordings (N)`, each with `mapping.tsv` (columns `audio_filename`, `sentence`) and about 50 Opus WEBM clips (48 kHz mono, about 6.5 s each) named by hash; one nested zip in folder 11 was a byte-identical duplicate of folder 3 and was deleted. Inventory: 4,579 mapping rows, 4,549 clips on disk, 30 rows without a file, 60 rows with empty text; all kept rows resolve, no duplicate ids, 59 duplicate sentences (recorded by different readers). Flattened into `data/central_jv/audio/` (hardlinks) plus one `data/central_jv/mapping.tsv` with 4,519 rows, so `prepare_tts.py --tsv data/central_jv/mapping.tsv --audio-dir data/central_jv/audio --file-col audio_filename --text-col sentence` works without layout code.

Manifest built 2026-09-22 with `uv run scripts/prepare_tts.py --tsv data/central_jv/mapping.tsv --audio-dir data/central_jv/audio --file-col audio_filename --text-col sentence` (defaults: max 20 s per chunk, 0.25 s silence between packed sentences, session label `cjv`). All 4,519 clips decoded, 0 failures. Result: **1,744 chunks, 7.97 h**, mean chunk 16.5 s, max 20.0 s, 2.6 sentences per chunk, longest target 53 words. Spot-checked chunks are 16 kHz with rms 0.07-0.09 and peak below 0.8. Packing arithmetic verified on a 20-clip dry run: 132.6 s packed = 129.6 s raw + 12 joins x 0.25 s. Disk: 880 MB `data/central_jv/wav` (chunks) + 864 MB `wav_raw` (per-sentence cache, safe to delete after training).

Training mixes this enables, via `load_manifests`:

| mix | chunks | hours |
|---|---|---|
| Jember train | 1568 | 9.04 |
| dev convo 5 | 270 | 1.84 |
| Central Javanese | 1744 | 7.97 |
| v4 = Jember + convo 5 + Central Javanese | 3582 | 18.85 |
| v5 = convo 5 + Central Javanese | 2014 | 9.81 |

Text conventions: plain ASCII, no diacritics, sentence-case with final `.`/`?`/`!` on 99% of rows, mean 13 words (max 37), no `...` tokens, 41 rows with digits. Dialect check over 61k tokens: Central markers `iku` 1922, `ning` 1713, `ora` 966, `kuwi` 783, `yo` 528, `wae` 255, `opo` 246; East markers `nang` 1, `ae` 0, `kate` 0, `sampeyan` 0. Semarang spellings with `o` (`okeh`, `seko`, `mergo`, `utowo`, `wes`). Indonesian code-mixing is light (`dan` 10, `juga` 21). So it matches the dev dialect but not its register: no hesitations, no Indonesian stretches, one or few readers. Expect it to help vocabulary and spelling, not acoustics; convo 5 stays in training for register.

## SLR35 screening (2026-09-26/27)

Spec `docs/superpowers/specs/2026-09-26-slr35-data-scale-design.md`. Rationale: every real gain so far came from data, the leaderboard/dev ratio shrinks as the model learns Javanese (2.05 zero-shot, 1.48 v5), and on convo 2 the same model scores 0.19 on speaker 5 vs 0.15 on speaker 7, so speaker variety is a direct target. Clip length is not a factor (clips > 30 s score no worse than short ones).

**OpenSLR SLR35** (Google / Reykjavik University / UGM, CC BY-SA 4.0; allowed by the rules provided the data is published to MDC after the competition): 185,076 utterances, 1,019 speakers, 16 shards. Shard 0 downloaded 2026-09-26 (1.2 GB, 11,654 FLAC): `scripts/prepare_slr35.py` maps ids to text (sorted by speaker so packed chunks are single-speaker, a period appended because SLR35 is sentence-case without punctuation), then `prepare_tts.py` packs to `data/slr35/train.csv`: **4,155 chunks, 19.27 h, 829 speakers**, mean chunk 16.7 s / 19.6 words. Four single-utterance chunks of 20-28 s (mostly leading silence) were dropped; typical edge silence is 8%. Text is encyclopaedic Central Javanese (`iku` 14k, `kuwi` 4.8k, `ora` 3.8k) with krama register; expect vocabulary and acoustic gains, not register.

**Screening night** (`runs/overnight4.sh`, decoder-only r32, `experiment.py train --full`, dry run t022 ok): control seed 2 on the v5 mix (seeds 0/1 = t013/t014, mean 0.1723); mix A = v5 mix + SLR35, seeds 0-2; mix B = A with convo 5 twice, seeds 0-2. Decision rule: a mix is adopted only if its three-seed mean beats the control mean by > 0.005 (twice the measured seed noise). Submitted models then use the all-scope recipe.

Also found on MDC (no publication obligation, all released Sep 2026, access-gated, requests sent 2026-09-26): Klaten Regency speech corpus (5 h, dev-dialect area, code-mixed, CC BY-NC-SA), Purworejo (10 h, CC BY-NC-SA), Jepara (5 h, CC BY-SA). They join as a follow-up mix once approved. Rejected: Common Voice Spontaneous 5.0 Javanese (1.4 h, 3 speakers, untranscribed), Multidialect TTS (forbids LLM training), Banyumasan (permission clause, divergent dialect), East Javanese TTS sets (Jember lesson).

**Screening result (2026-09-27, all decoder-only r32, convo 2):** control t013/t014/t023 = 0.1707/0.1738/0.1735 (mean 0.1727, sd 0.0017); mix A (+SLR35) t024/t026/t028 = 0.1687/0.1710/0.1769 (mean 0.1722, sd 0.0042); mix B (+SLR35, convo 5 ×2) t025/t027/t029 = 0.1693/0.1760/0.1707 (mean 0.1720, sd 0.0035). No mix clears the 0.005 bar; SLR35 is not adopted on this evidence. Caveat that decides the next step: decoder-only LoRA freezes the encoder, so the screen tested only SLR35's text (encyclopaedic, off-register) and not its 829 speakers, which was the reason to add it. `t030` runs mix A with the all-scope recipe to test that, against the all-scope control arm t018/t019/t020 (mean 0.1688, sd 0.0026).

**Submission-1 fallback:** `t020` (all-scope r32 on the v5 mix, seed 2), 0.1670, and 0.1659 with `tail` (`p008`). Its arm mean is 0.1688 against the submitted lora_v5's 0.1760, so the gain holds without best-of-three selection.

**MDC Purworejo** (arrived 2026-09-27; CC BY-NC-SA, Central Java, read speech, everyday topics): 115 archives (zip and tar.gz), 2,812 transcript rows; 2,780 resolve to audio with text, all unique by content. 54 rows dropped because their filenames repeat across recording sets with different audio (the packer caches by filename); 19 clips in set 63a are corrupt webm and are skipped by ffmpeg. Quote marks stripped from 7 sentences (the scorer deletes `"` anyway). Text is Central Javanese ngoko (`sing` 1780, `ora` 738, `kuwi` 434, `wis` 195), no East markers, no Indonesian, mean 16 words, 1.2 words/s. Packed with `prepare_tts.py --session pwj`: **2,317 chunks, 9.72 h** (4 chunks > 30 s are dropped at training time). Queued as `runs/overnight5.sh`: all-scope v5 mix + Purworejo, seeds 0 and 1, after `t030`.

**All-scope data-mix results (2026-09-27/28, convo 2, control = all-scope v5 mix t018/t019/t020, mean 0.1688, 0.1674 with `tail`):** `t030` (v5 mix + SLR35, 29.1 h, seed 0) hit the runner's 5 h full-run limit at step ~1,110 of 1,158 and is logged as `timeout`; its checkpoints were kept but it was not exported. In-training val (65 clips, greedy) was 0.2225 at epoch 1 and 0.1828 at epoch 2, against the controls' best epochs 0.1772/0.1658/0.1724, so with the encoder adapted SLR35 looks about 0.011 worse, not better. `t031` (v5 mix + Purworejo, 19.5 h, seed 0) scored 0.1673 (ind 0.1359, javind 0.1774, S/D/I 475/76/44), 0.1654 with `tail`: -0.002 against the control on one seed, inside the noise. Purworejo seed 1 was not run: even a matching second seed could not clear the 0.005 adoption bar. **Neither corpus is adopted; the v5 mix (convo 5 + Central Javanese) stays.** Mixes above ~25 h do not fit the runner's 5 h budget at ~12.5 s/step.

## Third submission (2026-09-28): soup of t018-t020 at scale 0.7 — worse on the leaderboard

Model: uniform soup of the three all-scope v5-mix seeds (`scripts/soup.py`, exact merged-delta average) with the delta scaled by 0.7 toward the base, CT2 float16, `postprocess: ["diacritics", "tail"]`, timestamps on (zip sha256 prefix `daed3a36`). The scale was chosen on convo 2, where the soup's scale curve (1.2 / 1.0 / 0.9 / 0.8 / 0.7 / 0.6 / 0.5 / 0.4 = 0.1783 / 0.1639 / 0.1592 / 0.1564 / 0.1566 / 0.1504 / 0.1555 / 0.1583 with `tail`) put 0.5-0.8 in a flat minimum.

| date | id | type | score | notes |
|---|---|---|---|---|
| 2026-09-28 | id-2992 | smoke test | 0.1435 | v5 had 0.1043 on the same smoke subset |
| 2026-09-28 | id-3000 | full evaluation (soup s0.7) | **0.2718** (public), best stays v5's 0.2600, rank #27 | +0.0118 vs v5 (+4.5% relative) although convo 2 said -0.0194 (0.1566 vs 0.1760). Quota 3 of 3 used; next  2026-09-29 UTC |

**Convo 2 got the direction wrong for the first time.** Every earlier submission moved the leaderboard the way convo 2 predicted (v2 -4.0% dev / -6.5% test, v5 likewise). The scale-0.7 soup moved it the opposite way. Shrinking the LoRA delta toward the base is what the two signals disagree on: on 60 convo-5 clips (trained on) the same models score v5 0.1533, soup s1.0 0.1585, soup s0.7 0.1810, and the smoke subset (drawn from the training set) ranked s0.7 worse than v5 as well. The hidden test set evidently rewards full-strength adaptation: the LoRA was always worth more there than on convo 2 (zero-shot -> v5: -44.6% on the leaderboard vs -23.2% on convo 2), and convo 2's scale curve reflects one conversation, two speakers and a comparatively Indonesian-heavy mix. Conclusion: scale < 1 is dropped; convo 2 alone must not tune anything that trades Javanese adaptation against base-model behaviour, and changes of that kind need a second held-out conversation (the reverse fold, train on convo 2 / score convo 5) before they reach the leaderboard.

## Smoke test of the scale-1.0 soup (2026-09-29): not sent to full evaluation

Model: the same three-seed soup of t018-t020 at scale 1.0 (`runs/ct2/soup_allscope`, convo 2 0.1639 with `tail`, d007), `postprocess: ["diacritics", "tail"]`, timestamps on (zip sha256 prefix `2ca09ac5`).

| date | id | type | score | notes |
|---|---|---|---|---|
| 2026-09-29 | id-3085 | smoke test | 0.1261 | v5 0.1043, soup s0.7 0.1435 on the same smoke subset |

**Not sent to full evaluation.** The smoke subset has ranked all four earlier submissions in the same order as the full test (zero-shot, v2, soup s0.7, v5), including submission 1, where convo 2 got the direction wrong. Scaling today's smoke gap by how s0.7's gap carried over predicts about 0.266 on the full test, worse than v5's 0.2600. The final ranking uses the best submission, so the unused  is kept for a candidate that beats v5 on smoke.

Reading: averaging seeds partly cancels each seed's delta, so a soup acts like a milder scale < 1. The order v5 < s1.0 < s0.7 on smoke, on convo 5 and on the full test (where measured) fits "more shrinkage toward the base, worse test score". Next candidates, both smoke-tested before any full evaluation: the 5-seed all-dev soup, and one single all-dev model at full strength. Caveat: the smoke subset is drawn from the dev data, so a model trained on those clips gets an optimistic smoke score.

## All-dev candidates on the smoke subset (2026-09-29)

All-dev = all-scope r32, v5 mix + convo 2, `--no-select --epochs 2`, fixed code (step 6); `diacritics,tail`, timestamps on.

| date | id | type | score | notes |
|---|---|---|---|---|
| 2026-09-29 | id-3104 | smoke test | 0.1217 | 5-seed all-dev soup t034/t035/t036/t041/t042 (zip `c8199674`) |
| 2026-09-29 | id-3108 | smoke test | 0.1348 | single all-dev model t034, seed 0 (zip `0cd3fb36`) |

Smoke ranking so far: v5 0.1043 < all-dev soup 0.1217 < soup s1.0 0.1261 < all-dev single 0.1348 < soup s0.7 0.1435. The single model is worse than the soup, so "soups shrink the delta and shrinking loses" does not explain v5's lead.

**lora_v5 is the same recipe as t018:** the same code (later commits only added flags whose defaults reproduce it), the same mix, all-scope, seed 0, 3 epochs with best-epoch selection. The one difference is the LoRA init, which was unseeded before the step-6 fix. v5's smoke and test lead is most likely a lucky draw of that recipe, not a better recipe.
| 2026-09-29 | id-3110 | full evaluation (5-seed all-dev soup) | **0.2567** (public), **new best**, rank #25 | -0.0033 vs v5's 0.2600 (-1.3% relative). Quota 3 of 3 used; next slot Sep 30 UTC |

**The all-dev soup beat v5, and the smoke subset got the direction wrong.** Smoke ranked it 0.1217 against v5's 0.1043; the full test ranked it 0.2567 against 0.2600. The smoke subset probably favours models that fit convo 5 closely, so it is no longer used to screen candidates. What the result supports: more in-domain speakers (convo 2) plus seed averaging beat the lucky single seed, in line with every earlier real gain coming from data.

## Last submission candidates (2026-10-01): acoustic augmentation passed its gate

**omniASR check (Sep 30, zero-shot on convo 2, honest):** CTC_1B_v2 0.4213 (p011), LLM_1B_v2 `jav_Latn` 0.2570 (p013), LLM_3B_v2 does not fit the laptop. Both outputs are lowercase without punctuation. Far from the 0.14 bar; dropped. `scripts/transcribe_omni.py`.

**Acoustic augmentation** (`lit/augment.py`, `--augment-acoustic`): with p = 0.6, 1–3 of MP3 re-encode 16–64 kbps, babble from other training clips at 5–20 dB SNR, synthetic reverb RT60 0.2–0.8 s, gain ±6 dB, band-pass 300–3400 Hz. Motivation: test WER is about 1.5× dev WER, dev clips are 64 kbps MP3 voice notes, and the Central Javanese corpus is clean WAV. Noisy convo 2 for the gate: `scripts/make_noisy_dev.py` (babble 5–15 dB + MP3 16–32 kbps, seed 0).

**Gate (all-scope v5 mix + 34 convo-5 long pieces, seed 0, 3 epochs with selection; neither run sees convo 2; `diacritics,tail`):**

| run | clean convo 2 | noisy convo 2 |
|---|---|---|
| t046 control | 0.1670 | 0.5711 |
| t047 + augmentation | 0.1654 | **0.4218** |

Passed: noisy −0.149 (need > 0.01), clean −0.0017 (allowed up to +0.005).

**All-dev + long clips** (`scripts/split_long_clips.py`: 27 of 37 > 30 s dev clips recovered as 54 verified pieces, +14 min): seeds t043/t044/t045 = 0.1271/0.1316/0.1277 (contaminated). **+ augmentation:** t048/t049/t050 = 0.1375/0.1367/0.1367 (contaminated).

| soup (scale 1.0, `diacritics,tail`, timestamps on) | clean convo 2 (contaminated) | noisy convo 2 | zip |
|---|---|---|---|
| `soup_alldev_long` t043–t045 (safe) | 0.1265 (d020) | 0.5416 | `submission_alldev_long_04c70d99.zip` |
| **`soup_alldev_long_aug` t048–t050 (final)** | 0.1367 (d019) | **0.3802** | `submission_alldev_long_aug_4ba9721b.zip` |

The clean contaminated number favours the safe soup by 0.01, but it mostly measures memorisation of convo 2; the honest gate says augmentation is neutral on clean audio and far better on degraded audio. Decision rule fixed on Sep 29: augmented soup if the gate passes. Both zips: `model.bin` byte-identical to the CT2 model, GPU 10-clip round-trip 10/10 identical, `pack.sh` check passed.
