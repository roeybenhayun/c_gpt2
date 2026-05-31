# Paper validation tests

Test cases for *Language Models are Unsupervised Multitask Learners*
(Radford et al., 2019) — the GPT-2 paper. See
`docs/papers/language_models_are_unsupervised_multitask_learners.pdf`.

Every case here has a **validated full prompt**: either the paper printed the
full prompt itself (translation, CoQA), or we recovered the full source
article from a public dataset / web fetch and rebuilt the prompt via the
GPT-2 tokenizer (Table 12, Table 14). Cases where the source 768-token
context could not be located (Tables 7-11) are NOT shipped — see
"What's intentionally missing" below.

## Layout

Each test case is a directory containing:

| File                       | Purpose                                                                                                                                                                                  |
|----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `prompt.txt`               | Input prompt fed to the model (loaded by the harness and passed via `gpt2 --prompt`).                                                                                                    |
| `expected.txt`             | Reference output from the paper. **For qualitative comparison only** — local RNG diverges from the paper's run, so byte-equal match is not expected even when the prompt matches exactly.|
| `metadata.json`            | Test conditions and provenance (table, page, sampling params, source notes).                                                                                                              |
| `build_info.json` *(opt.)* | Present for cases whose prompt was rebuilt from an external source — records token math + anchor used.                                                                                    |
| `prompt_pdf_truncated.txt` *(opt.)* | Original visible-chunk transcription from the PDF (before swap to validated full prompt). Kept for reference.                                                                  |

Run all cases with `scripts/run_paper_tests.sh`. The harness invokes the
`gpt2_{small,medium,large}` binary once per case, passes every parameter from
`metadata.json` on the CLI, and captures the generated text into
`<case>/generated_<target>_<size>.txt`.

## Cases (10 total)

| Dir / case                                 | Source                                | Conditions                                       |
|--------------------------------------------|---------------------------------------|--------------------------------------------------|
| `webtext/t12_chocolate_cake`               | Table 12 — cakemerchant.com (web)     | k=40, T=1.0, 128 out tok (384-tok context)       |
| `summarization/t14a_chauvet_cave`          | Table 14 — cnn_dailymail / Daily Mail | k=2,  T=1.0, 100 out tok, ` TL;DR:` trigger      |
| `summarization/t14b_uboat_yacht`           | Table 14 — cnn_dailymail / Daily Mail | k=2,  T=1.0, 100 out tok, ` TL;DR:` trigger      |
| `summarization/t14c_yemen_war`             | Table 14 — cnn_dailymail / CNN        | k=2,  T=1.0, 100 out tok, ` TL;DR:` trigger      |
| `translation/t15a_en_fr_release`           | Table 15                              | greedy (k=1, T=0), few-shot EN=FR pairs          |
| `translation/t15b_fr_en_hernia`            | Table 15                              | greedy (k=1, T=0), few-shot FR=EN pairs          |
| `translation/t15c_en_fr_kerry`             | Table 15                              | greedy (k=1, T=0), few-shot EN=FR pairs          |
| `translation/t15d_fr_en_kerry`             | Table 15                              | greedy (k=1, T=0), few-shot FR=EN pairs          |
| `qa_coqa/t16_olympics_torch`               | Table 16                              | greedy (k=1, T=0), doc + Q/A history + `A:`      |
| `qa_coqa/t17_tom_the_dog`                  | Table 17                              | greedy (k=1, T=0), doc + Q/A history + `A:`      |

## Provenance notes per category

- **`webtext/t12`** — Source page is the Cake Merchant blog post
  `https://cakemerchant.com/2014/12/11/mint-chocolate-cookie-crunch-cake/`.
  Prompt rebuilt by tokenizing the article prefix up to the paper's visible-chunk
  anchor and keeping the last 384 tokens (paper Table 12's context length).
  Result is exactly 384 tokens. See `build_info.json`.
- **`summarization/t14a/b/c`** — Articles are from the public `cnn_dailymail`
  HuggingFace dataset (the exact CNN/Daily Mail dataset the paper used for
  summarization per §3.6). Prompt is the full article body + ` TL;DR:`
  trigger; the paper truncated the article in the PDF with `...` markers but
  the model received the full text.
- **`translation/t15a-d`** — Paper §3.7 describes the few-shot format
  `english sentence = french sentence` but does NOT publish the specific
  example pairs used. Our prompts use a fixed 3-pair few-shot header
  (`hello`/`cat`/`books`) baked into each `prompt.txt`. The source/target
  sentences themselves are reproduced verbatim from Table 15.
- **`qa_coqa/t16-17`** — Paper §3.5 prints the full passage + Q/A history +
  trailing `A:` in the tables; prompts are verbatim.

## What's intentionally missing

- **Tables 7-11 (WebText random contexts)** — The paper caption says
  "Contexts are 768 tokens, with approximately 256 tokens worth of paragraphs
  shown." We could only access the visible ~256 tokens; the source pages were
  either not in any public dataset or no longer indexed by web search. Running
  the model on a 256-token transcription of a 768-token context isn't a
  meaningful paper-reproduction signal, so these cases are not shipped.
- **Table 13 (out-of-distribution unicorns)** — Omitted intentionally.

The companion project `~/projects/openwebtext_search/` contains the searches
and tokenizer-driven prompt-rebuild script used to validate the cases above.
