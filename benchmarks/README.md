# Long-context benchmark profiles

The harness covers the OOLONG and LongBench-v2 CodeQA benchmarks evaluated in the [Recursive Language Models paper](https://arxiv.org/abs/2512.24601):

| Command | Default profile | Score |
| --- | --- | --- |
| `npm run eval:oolong` | All 400 held-out OOLONG-synth test cases at 131,072 tokens, with 50 cases from each of the eight test datasets | Upstream OOLONG score |
| `npm run eval:longbench` | All 50 LongBench-v2 CodeQA cases from the `Code Repository Understanding` domain | Multiple-choice accuracy |

Both evals keep the long context in `context.txt`, expose only the question to the root model, and score the last successful structured `rlm.final` result. The harness loads pi-ipython from `../pi-ipython` (or `PI_IPYTHON_ROOT`) together with this package. They are opt-in because they call the configured model and can consume substantial time and tokens.

## Run the full profiles

```bash
npm run eval:oolong -- --model openai-codex/gpt-5.6-sol
npm run eval:longbench -- --model openai-codex/gpt-5.6-sol
```

The OOLONG default uses 400 held-out cases instead of the 25-case validation sample from the upstream training environment. The RLM paper's 131K OOLONG column used all 50 `trec_coarse` validation cases. Run that exact subset with:

```bash
npm run eval:oolong -- \
  --model openai-codex/gpt-5.6-sol \
  --split validation \
  --dataset trec_coarse \
  --count 50
```

Use smaller runs while changing the extension:

```bash
npm run eval:oolong -- --model openai-codex/gpt-5.6-sol --count 3
npm run eval:longbench -- --model openai-codex/gpt-5.6-sol --count 3
```

Check case selection without calling a model:

```bash
npm run eval:oolong -- --dry-run --count 3
npm run eval:longbench -- --dry-run --count 3
```

Dry runs still stream dataset rows from Hugging Face. LongBench and the longest OOLONG rows can transfer large contexts.

## Results and reproducibility

Results go to `benchmark-results/<benchmark>-<timestamp>.jsonl`. Raw Pi events and stderr go into the adjacent `.artifacts` directory. The run record includes the model, thinking level, dataset revision, dependency versions, Pi version, Git commit, and hashes of benchmark and runtime inputs. Each case records context and prompt hashes. The summary reports score, failures, token usage, and cost.

The default profiles preserve dataset order. `--shuffle-buffer N` enables deterministic streaming shuffling with `--seed`, but changes the selected subset when `--count` is below the full profile size.

Use the same model, thinking level, profile, Pi version, inputs, and clean Git state when comparing runs. A seed controls optional case shuffling, not provider sampling. Repeat important comparisons to estimate model variance. The benchmark prompt asks the root model to use `ipython`, but does not prescribe one decomposition algorithm.

Environment variables mirror the integration suite:

- `PI_RLM_BENCH_MODEL`
- `PI_RLM_BENCH_THINKING`, default `low`
- `PI_RLM_BENCH_AGENT_DIR`

The direct dependencies are isolated by `uv` and pinned in `benchmarks/requirements.txt`. Resolved transitive versions are recorded in each run. Nothing is installed into the extension-owned kernel runtime.
