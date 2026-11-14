# Free-model aggregator — make.fun `free-model-aggregator-01`

One server, Python 3.13 standard library only (`main.py`, no dependencies).
It reads the trial brief (`MAKEFUN_PROMPT_FILE`), registers with the trial
env, and answers the client's OpenAI Chat Completions and Anthropic
Messages requests — JSON and SSE streaming — by routing each request to
the best free model that meets its requirements.

## How it works

- **Catalog from the brief alone.** The provider directory and the sealed
  quality table are embedded in `prompt.txt`; the catalog is built from
  those, so `python3 main.py --list` prints byte-for-byte what the server
  registers.
- **Lossless wire translation.** Each request is normalised (system text,
  messages, tool names, `max_tokens`, stop sequences, image bytes, JSON
  mode) and rebuilt in the chosen provider's wire — OpenAI, Anthropic or
  Gemini — so the provider echoes exactly what the client sent.
- **Budget-aware greedy routing.** Candidates are ranked by sealed rating.
  One attempt per provider per round, with a conservative attempt cap per
  provider (the floor of its drawn budget) and a skip after two consecutive
  failures (spent budget or outage). A second round retries providers that
  failed exactly once, which recovers scheduled single-call failures.
  JSON-mode requests from an OpenAI client never go to an Anthropic-wire
  provider, whose wire cannot carry `response_format`.
- **Faithful answers.** The provider's text is returned whole, with its
  usage counts as its wire reports them (the known usage formula is the
  fallback) and a family-0 stop reason (`stop` / `end_turn`) in the
  client's wire. Streamed answers are re-emitted with correct event order
  and a final usage event.
- **Deterministic.** Every choice is a function of the brief and the
  provider answers seen so far — no wall clock, no unseeded randomness —
  so a replay makes the same choices and leaves the same state export.

## Run

```sh
python3 main.py --list     # print the catalog, exit 0
python3 main.py            # trial mode (needs MAKEFUN_ENV_URL,
                           # MAKEFUN_PROMPT_FILE, MAKEFUN_TRIAL_SEED)
```

## Grade locally

```sh
node cli/makefun.mjs grade-local --kit <kit> \
  --bot ~/workspace/makefun/free-model-aggregator-01-entry \
  --driver process --instances 3
```

## License

MIT — see `LICENSE`.
