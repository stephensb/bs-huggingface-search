# HF Model Explorer

A single-page, dependency-free front end for searching Hugging Face Hub models with
sorting and filtering the official site does not offer. The browser talks straight to
`https://huggingface.co/api/models`; there is no backend and no build step.

## Run

Use the hosted copy at **https://stephensb.github.io/bs-huggingface-search/**, open
`index.html` directly in a browser, or serve the folder:

```sh
python3 -m http.server 8765
# then open http://127.0.0.1:8765/
```

## How it works

1. **Fetch** (top bar) pulls a slice of the Hub using the API's server-side options:
   free-text search, author, task, library, required tags, gguf-only, hosted-only, and
   the server ordering (downloads / likes / trending / newest / recently modified).
   The ordering decides *which* models you get when there are more than the limit, so
   pick it to match what you care about. Up to 10,000 models are fetched in pages of 1,000.
2. **Filters** (sidebar) and **sorting** (column headers) then run entirely in the browser
   and update instantly.

## Filters

- Name include / exclude regex (e.g. exclude `gguf|awq|gptq` to drop quantized re-uploads)
- Parameter count range in billions (from safetensors metadata, GGUF metadata as fallback, or
  parsed from the name such as `27B`, `30B-A3B`, `8x7B` when metadata is missing or clearly
  describes a different file; those values are shown with a `~`)
- Minimums for 30-day downloads, all-time downloads, likes, trending score, GGUF context length, live providers
- Created / modified date ranges, with quick "created in last N days" buttons
- Gated, quantized, has safetensors, has GGUF, custom code, has a live inference provider
- Base-model relation: originals only, or finetunes / quantizations / adapters / merges
- Tags that must all be present / must all be absent
- Facets with counts computed from the fetched set: license, architecture, task, library,
  quantization format, safetensors dtype, language, author, inference provider, base model

## Sorting

Click a column header to sort, click again to flip, shift-click to add secondary keys.
Nulls always sort last. Derived columns worth trying:

| column | meaning |
|---|---|
| momentum | 30-day downloads ÷ all-time downloads. Near 100% means new or surging. |
| dl/day | all-time downloads ÷ days since the repo was created |
| likes/10k dl | likes per 10,000 all-time downloads, a rough quality-vs-popularity signal |

Use the **columns** menu to show hidden ones (author, library, quant, gguf ctx, gguf size,
providers, base model, relation, age).

## Checking the actual model files

The Hub's own GGUF metadata describes one file per repo and often lands on a draft or
projector file, so the app can look at the real files in three increasingly precise ways:

1. **File names** come with every fetch. They identify the main weight files versus
   auxiliary GGUFs (mmproj, draft/MTP heads), list the quant variants present (facet
   "gguf quant available", column "gguf quants") and flag vision projectors.
2. **check files** (toolbar) fetches the file tree with sizes for every filtered model
   that has not been checked yet, up to 300 per click with four requests in flight. It
   yields the real size of each quant, total repo size, and a parameter estimate from the
   highest-precision file's size and its bits-per-weight (shown with `~`). Results are
   cached per model and revision. The same button in an expanded row checks one model.
3. **read GGUF header** (expanded row) range-reads the header of the smallest main GGUF
   file, which lists every tensor's dimensions, and computes the exact parameter count
   plus architecture, context length and expert count. It costs roughly 4 to 12 MB per
   model because the tokenizer vocabulary lives in the header, so it is per-model only.
   Gated repos cannot be read this way without a token.

Safetensors metadata from the Hub is exact and is never overridden.

## Local cache

Every fetch is stored in the browser's IndexedDB, keyed by the exact fetch options.
Pressing **Fetch** with options that were fetched before loads the stored result
instantly (also offline) and the status line shows how old it is; press **refresh** to
bypass the cache and pull current numbers. The 25 most recent queries are kept, and the
status line offers **clear** to drop them all, together with the file checks.

The cache works when the page is opened as a local file in Chrome and Firefox. Chrome
shares one storage area between all local files, so other local HTML pages could read or
clear it. Safari does not allow IndexedDB from local files; there the app just always
fetches from the network.

## Other

- Click a row to expand it: architectures, dtypes, base models, providers, datasets,
  arXiv links, all tags, and links to the model page, file tree, raw API JSON, and the
  Hub's finetune / quantization listings.
- **copy link** copies a URL that restores the fetch options, filters, facets, sort and
  columns. Opening such a URL fetches automatically.
- **CSV** / **JSON** export the currently filtered and sorted rows (CSV uses visible columns).

## API notes

- The Hub does not filter by parameter count, date or download thresholds server-side,
  which is why the app fetches a slice and filters locally.
- `search` matches substrings of the repo name only, not the model card.
- The Hub's GGUF metadata (parameter count, context length, size) describes a single file
  in the repo. In multi-file repos it often picks a draft/MTP head or mmproj file, so a 27B
  repo can report 1.9B parameters. The app cross-checks against the name and overrides
  when they disagree by more than 2x.
- Unauthenticated rate limit is 500 requests per 5 minutes; each fetch uses one request
  per 1,000 models.
- The `arch` checkbox requests `config.json` metadata (architecture, model_type). It adds
  roughly 60% to the payload; untick it for faster large fetches.
