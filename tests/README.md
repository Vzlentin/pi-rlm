# Verification

`npm test` is model-free. It runs typechecking, host cancellation, headless child
sessions, librlm location and update, benchmark data-isolation tests, and a real
kernel test through pi-ipython: the prompt section, cross-cell handles, usage
attribution, `rlm.final`, cwd, gather cancellation, and depth rejection. The
durable test runs both packages' pi-durable adapters with a faux model: a child
owned by a background task, gather in the same and in a later cell, cancellation,
a failed cell, a grandchild left running at depth 2, and a restart.

The kernel test needs a pi-ipython checkout (`PI_IPYTHON_ROOT`, default
`../pi-ipython`, with `npm install` run) and librlm (`RLM_LIBRLM_ROOT`, default
`~/Dev/librlm`).

Do not add tests that match source strings, pin method names, or merely restate
implementation constants. Use the child process fixture across runner, host, and
kernel tests.

Model-backed acceptance (explicitly opt-in):

```sh
PI_IPYTHON_ROOT=/path/to/pi-ipython \
RLM_LIBRLM_ROOT=/path/to/librlm \
PI_RLM_TEST_MODEL=openai-codex/gpt-6.1-sol npm run test:integration
```

Use an available model from `pi --offline --list-models`. Pi, Node.js 22.19 or
newer, Python, and `uv` must be on `PATH`. The pi-ipython checkout needs its npm
dependencies and Python runtime. The selected provider needs existing credentials
in the source agent directory or its environment variables.

The suite creates temporary settings that load this worktree, pi-ipython, and the
optional provider extension through normal discovery. Root sessions and children
share these settings and a temporary working directory. No `-e` paths are passed
or forwarded. Existing `auth.json` and `models.json` files are linked, not copied.
Other user packages, project resources, and kernel persistence are disabled.

This runs real Pi CLI/RPC sessions with both packages: child completions, usage
attribution, concurrent gather, retained handles, release, cancellation, recovery
and process/socket cleanup. The parallel case also checks separate child and
grandchild kernels and rejects another spawn at depth 2.
`test:integration:full` adds the costly large-context case.

Optional settings:

- `PI_RLM_TEST_THINKING`: defaults to `low`.
- `PI_RLM_TEST_TIMEOUT`: per-run seconds, defaults to `300`.
- `PI_RLM_TEST_AGENT_DIR`: source agent directory for existing authentication and model configuration. Defaults to `PI_CODING_AGENT_DIR` or `~/.pi/agent`.
- `PI_RLM_TEST_PROVIDER_EXTENSION`: local custom-provider extension loaded through the temporary settings.
