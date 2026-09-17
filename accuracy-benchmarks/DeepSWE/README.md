# DeepSWE Benchmark

Runs the DeepSWE agentic coding benchmark (113 tasks) against a Kimi-K3 (or other model)
endpoint using Datacurve's Pier framework with the `mini-swe-agent` harness — matching the
methodology used for Datacurve's official public leaderboard.

## Setup

### 1. Install Docker (use the official repo, not Ubuntu's `docker.io`)

Ubuntu's bundled `docker.io` ships an outdated `docker compose` that lacks `--project-name`
support, which Pier requires.

```bash
apt update && apt install -y ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt update
apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
systemctl enable --now docker
```

If `docker.service` fails to start with `no sockets found via socket activation`:
```bash
systemctl daemon-reload
systemctl reset-failed docker.service docker.socket
systemctl unmask docker.socket docker.service
systemctl enable docker.socket
systemctl start docker.socket docker.service
```

### 2. Install Pier and clone the task set

```bash
uv tool install git+https://github.com/datacurve-ai/pier
git clone https://github.com/datacurve-ai/deep-swe ~/deep-swe
```

## Running against a custom endpoint

The model string's **prefix** must match a provider Pier's `pier/agents/utils.py`
(`PROVIDER_KEYS`) recognizes — it determines which API-key env var Pier looks for
and which domains get added to the sandbox's network allowlist. Using the wrong
prefix, or a made-up one, causes `AuthenticationError`, `NotFoundError`, or a DNS
resolution failure inside the sandbox.

| Model      | Provider prefix | Env var           |
|------------|------------------|--------------------|
| Kimi-K3    | `moonshot/`      | `MOONSHOT_API_KEY` |
| GLM        | `zai/`           | `ZAI_API_KEY`      |

The actual model name after the prefix must match what the endpoint's `/v1/models`
reports — confirm with:
```bash
curl -sS <endpoint>/v1/models -H "Authorization: Bearer <key>" | python3 -m json.tool
```

Custom `api_base` is passed via `--ak model_kwargs`, but the sandbox's network
allowlist is built from specific env var names (`OPENAI_BASE_URL`, `OPENAI_API_BASE`,
`ANTHROPIC_BASE_URL`, etc.) — **not** from `model_kwargs` directly. Set
`OPENAI_BASE_URL` alongside `model_kwargs` so the endpoint's domain is actually
allowlisted:

```bash
cd ~/deep-swe
export MOONSHOT_API_KEY="<key>"          # or ZAI_API_KEY, etc.
export OPENAI_BASE_URL="<endpoint>/v1"    # required for the network allowlist

pier run -p tasks --agent mini-swe-agent \
  --model "moonshot/kimi-k3" \
  --ak model_kwargs='{"api_base": "<endpoint>/v1"}' \
  --n-concurrent 4 \
  --job-name my_job_name
```

Always run `--n-tasks 1` first to confirm the connection/provider config works
before scaling to the full 113-task run.

## Scoring

Reward is strict all-or-nothing per task (no partial credit); F2P/P2P show
partial progress toward it. **Use the per-task `verifier/reward.json` files as
the authoritative score**, not just the rolled-up `result.json` summary — under
retries, `result.json`'s `n_trials` count can include duplicate attempts at the
same task, which distorts the aggregate math (`n_trials` may not equal 113).

```bash
python3 -c "
import json, os
job_dir = 'jobs/my_job_name'
total, count = 0, 0
for t in os.listdir(job_dir):
    p = os.path.join(job_dir, t, 'verifier', 'reward.json')
    if os.path.exists(p):
        with open(p) as f:
            total += json.load(f).get('reward', 0)
        count += 1
print(f'{count} tasks scored, average reward: {total/count:.4f}')
"
```

## Results (this run)

| Model          | Endpoint                | Reward  | Tasks |
|----------------|--------------------------|---------|-------|
| Kimi-K3        | `<AMD dedicated endpoint>` | 0.6460  | 113   |
| Kimi-K3        | `inference.do-ai.run`    | 0.5310  | 113   |
| GLM-5.3-Flash  | `inference.do-ai.run`    | 0.4955  | 111   |
