# pi-rlm

A Pi package that adds recursive language-model child calls to the `ipython` tool
of [pi-ipython](https://github.com/Vzlentin/pi-ipython). Inside the kernel, `rlm`
is bound to a [librlm](https://github.com/Vzlentin/librlm) client:

```python
h = await rlm.spawn("Summarize this.", context=text)
[r] = await rlm.gather([h])
r.text if r.status == "ok" else r.error
await rlm.final({"answer": r.text})
```

Children are fresh, tool-free, depth-1 completions on the model and thinking level
of the running `ipython` call. Handles survive across cells; a failed or cancelled
cell cancels the children it spawned. `rlm.final` prints its value at the end of the
cell's output (and in `details.final`) but does not end the turn.

## Install

Install both packages. Requirements are pi-ipython's, plus Git.

```bash
pi install https://github.com/Vzlentin/pi-ipython
pi install https://github.com/Vzlentin/pi-rlm
```

pi-rlm uses the librlm commit pinned in `extensions/librlm.ts`. It clones
`https://github.com/Vzlentin/librlm` into `${XDG_DATA_HOME:-~/.local/share}/pi-rlm/librlm`
on first use and checks out the pin with a detached `HEAD`. At session start,
a clone already at the pin needs no network access. Otherwise, pi-rlm fetches
and checks out the pin. If synchronization fails, it warns and blocks prompt
loading and kernel startup. Set `RLM_LIBRLM_ROOT=/absolute/path` to use a
development checkout instead (never changed by pi-rlm). A librlm whose host
protocol or prompt schema pi-rlm does not support fails the kernel start or
prompt loudly.

## How it works

- On pi-ipython's `ipython:kernel-starting` event, pi-rlm starts its child host (a
  private Unix socket), passes `RLM_HOST_SOCKET` and `RLM_HOST_TOKEN` to the kernel
  only, and loads librlm's `rlm.ipython_extension` in the kernel. librlm owns
  handles, gather, release and final inside the kernel; pi-rlm runs the child
  completions through Pi's model registry.
- Child usage is added to the `ipython` result's `usage`, with `details.nestedUsage`
  and `details.children` (`spawned`, `completed`) for the call during which it was
  observed. Running children are shown in the status line.
- When `ipython` is active, the RLM API and guidance from librlm's
  `rlm/prompts/ipython.json` are added as an `rlm` system prompt section.

## Tests

```bash
npm install
npm test
```

The kernel tests need a sibling pi-ipython checkout (`../pi-ipython`, or
`PI_IPYTHON_ROOT`) and librlm (`RLM_LIBRLM_ROOT`, default `~/Dev/librlm`). See
[tests/README.md](tests/README.md) for the model-backed acceptance tests.

## Paper benchmarks

Run the public OOLONG and LongBench-v2 CodeQA profiles from the Recursive Language Models paper:

```bash
npm run eval:oolong -- --model openai-codex/gpt-5.6-sol
npm run eval:longbench -- --model openai-codex/gpt-5.6-sol
```

Both are opt-in and can consume substantial time and tokens. See
[`benchmarks/README.md`](benchmarks/README.md).

## License

MIT
