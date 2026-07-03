# Arabic document triage: OCR and translation benchmarks on consumer hardware

Sixteen notebooks answering one question: **can you turn a photo of an Arabic document into a usable German gist translation on an 8 GB laptop GPU, and know when to distrust the output?**

This research grew out of a feasibility case study with [lingomatch GmbH](https://www.lingomatch.de/) (Gov/Justice-Tech, Saarbrücken) through Saarland University's Triathlon program. Everything here runs on public datasets (KITAB-Bench, FLORES-200, Arabic Wikipedia) and open-weight models; no client data. The interesting parts are the failure modes.

## What's in here, findings first

**Translation: the licence, not the benchmark, picks your production model.**
- NLLB-1.3B translates well but is CC-BY-NC licensed, which rules it out for commercial deployment. Notebook 08 benchmarks the Apache-2.0 alternative: Opus-MT-ar-de matches it on short sentences (BLEU 43.8 vs 42.4) at one ninth of the memory, running comfortably on CPU.
- That short-text result does **not** generalize: on long legal text (notebook 09), Opus-MT truncates and inverts meaning where NLLB stays coherent. Same models, different input length, opposite conclusion. This is why you benchmark on realistic inputs.
- TranslateGemma 4B (notebooks 10-11) produces the best German of everything tested, while scoring the *lowest* BLEU, a nice demonstration that n-gram metrics punish good paraphrase. Its end-to-end image→German mode silently hallucinated an entire plausible document from an image it couldn't read, which is disqualifying for a triage tool regardless of quality; a cascade (OCR → text translation) keeps every stage auditable at 35x less compute.

**OCR: measure the confidence, not just the accuracy.**
- Surya OCR evaluated on 50 KITAB-Bench samples (notebook 07) with per-sample CER/WER: 31/50 samples reach usable quality (CER ≤ 10%), with failures concentrated in degraded scans and stylized fonts.
- Surya's own confidence score actually predicts its failures (notebook 12): a document-level confidence router flagged 4 of 4 catastrophic OCR failures, the basis for an automatic "call a human interpreter" escalation path.
- Defense in depth for the failures confidence misses: perplexity scoring with a diacritic confound found and fixed (13), script-ratio and repetition heuristics that catch hallucination loops (14), and PaddleOCR as an independent second engine for disagreement-based flagging (15). One adversarial sample evades all four signals, which is the empirical argument that human review stays mandatory.
- Phone photos in the field are skewed and shadowed: projection-profile deskew plus CLAHE contrast recovery (notebook 16) wins back roughly a third of the OCR quality lost to degradation (CER 7.6% → 4.9% on a multi-line Arabic page).

## Notebook index

| # | Notebook | One-line result |
|---|----------|-----------------|
| 01 | translation_hello_world | NLLB-600M runs locally, Arabic→German works |
| 02 | ocr_hello_world | Surya reads clean Arabic |
| 03-05 | chained_pipeline | image → OCR → translation end to end, sentence reassembly fixes chunking errors |
| 06 | reassembly_sanity_check | regex reassembly holds up on messy inputs |
| 07 | kitab_bench_ocr_eval | Surya on 50 KITAB-Bench samples: CER/WER + confidence per sample |
| 08 | opus_mt_comparison | Opus-MT-ar-de matches NLLB quality, 1/9 memory, Apache 2.0 |
| 09 | opus_mt_long_text | the short-text result does not generalize to long legal text |
| 10 | translategemma | best quality, lowest BLEU; end-to-end mode hallucinates |
| 11 | cascade_translategemma | Surya→TranslateGemma: best long-text quality, 35x slower |
| 12 | confidence_router | document-level router flags 4/4 catastrophic OCR failures |
| 13 | perplexity_signal | second safety signal; diacritic confound found and fixed |
| 14 | heuristic_guardrails | script-ratio + repetition heuristics patch the remaining blind spots |
| 15 | paddleocr | independent second engine, model-disagreement flagging |
| 16 | degradation_preprocessing | deskew + CLAHE: CER 7.6% → 4.9% on degraded input |

`surya_50_outputs.json` and `paddle_50_outputs.json` hold both engines' raw outputs on the same 50 KITAB-Bench samples, so the disagreement analysis is reproducible without re-running either engine.

## Hardware and setup

Everything ran on a laptop RTX 4060 (8 GB VRAM). Models that fit: NLLB-600M/1.3B (FP16), Opus-MT (CPU-friendly, ~300 MB), Surya (~2 GB), TranslateGemma 4B quantised. PaddleOCR ran CPU-only in a separate conda env after a PaddlePaddle/oneDNN fight documented in notebook 15.

```sh
pip install -r requirements.txt
jupyter lab
```

Run notebooks in order; later ones load cached outputs from earlier ones where possible.

## Honest limitations

Handwritten Arabic remains out of reach for every engine tested; the honest recommendation is typed-document triage with human escalation for handwriting. BLEU against FLORES references understates LLM-style translators (see notebook 10). The 50-sample KITAB-Bench subset is large enough to expose failure modes, not to certify accuracy. And one adversarial sample beats all four safety signals, so nothing here supports removing the human from the loop, which is the design conclusion, not a caveat.

---

Akshay Ashok · [axwolf13.github.io](https://axwolf13.github.io/) · akshay57ax@gmail.com
