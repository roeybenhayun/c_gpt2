# Temporarily disabled

This case's prompt is 1682 tokens (full Daily Mail article + " TL;DR:" trigger).
Two bugs in `gpt2.c` make it unsafe to run as-is:

1. `encode_response` buffer in `gpt2.c:1857` (~8 KB) is smaller than the
   tokenizer's JSON reply for ~1024+ token prompts, so the reply truncates
   mid-array.
2. `gpt2.c:2120-2122` `continue`s on parse failure — fine for interactive
   mode, but with `--prompt` (cli_input set) it loops on the same input
   forever spamming "Failed to parse tokens!".

Separately, even with the buffer fix, the 1682-token prompt exceeds GPT-2's
1024-token context window, so the prompt itself also needs truncation to be
runnable on stock GPT-2.

To re-enable: rename `metadata.json.disabled` back to `metadata.json` (after
both gpt2.c bugs are fixed AND the prompt is re-truncated to ~900 tokens).

Also cleared the 1.5 MB `Failed to parse tokens!` spam in
`generated_int8_large.txt` left from the original stuck run.
