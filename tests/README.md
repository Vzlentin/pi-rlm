# Verification

`npm test` is model-free. It runs typechecking, host cancellation, custom-provider
dispatch, librlm location and update, benchmark data-isolation tests, and a real
kernel test through pi-ipython: the prompt section, cross-cell handles, usage
attribution, `rlm.final`, and failed-cell cancellation.

The kernel test needs a pi-ipython checkout (`PI_IPYTHON_ROOT`, default
`../pi-ipython`, with `npm install` run) and librlm (`RLM_LIBRLM_ROOT`, default
`~/Dev/librlm`).

Do not add tests that match source strings, pin method names, or merely restate
implementation constants. Keep provider mocks at the provider boundary.

Model-backed acceptance (explicitly opt-in):

```sh
PI_RLM_TEST_MODEL=openai-codex/gpt-5.6-sol npm run test:integration
```

This runs real Pi CLI/RPC sessions with both packages: child completions, usage
attribution, concurrent gather, retained handles, release, cancellation, recovery
and process/socket cleanup. `test:integration:full` adds the costly large-context case.

Optional settings:

- `PI_RLM_TEST_THINKING`: defaults to `low`.
- `PI_RLM_TEST_TIMEOUT`: per-run seconds, defaults to `300`.
- `PI_RLM_TEST_AGENT_DIR`: credentials and provider settings directory.
- `PI_RLM_TEST_PROVIDER_EXTENSION`: explicit custom-provider extension; all other extension discovery is disabled.
