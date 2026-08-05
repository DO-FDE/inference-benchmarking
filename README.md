# inference-benchmarking

Tools for benchmarking and characterizing LLM inference-serving performance.

This repo collects the scripts we use to load-test served LLM endpoints
(vLLM, SGLang, dedicated/hosted endpoints), measure latency and throughput
under concurrency, and score results against SLA targets. It's organized by
tool; new tooling gets its own top-level directory.

## Layout

```
aiperf/    concurrency-sweep + SLA-scoring tools built on aiperf
```

## aiperf/

A concurrency sweep across workload shapes and prefix-cache scenarios, scored
against inter-token-latency (ITL) SLA gates.

- **`run_aiperf_workload_shapes.sh`** — orchestrator. Drives `aiperf profile`
  across a concurrency grid for each workload shape (e.g. 8k / 64k / 256k input,
  1k output) and each cache ratio (fraction of the input served from a reused
  cached prefix), against a local port or a remote endpoint. Each run is scored
  once per ITL gate (`claw` < 25 ms, `chat` < 66.7 ms) with no re-run.
- **`parse_sweep.py`** — post-processor. Converts an aiperf sweep export into a
  per-GPU summary CSV: per-GPU throughput, TTFT/ITL percentiles, identity
  validation, degeneracy detection, and per-point SLA pass/fail plus the max
  concurrency under the ITL gate. `run_aiperf_workload_shapes.sh` calls it
  automatically; it also runs standalone.

### Requirements

- [`aiperf`](https://github.com/ai-dynamo/aiperf) on `PATH`
- Python 3
- A served, OpenAI-compatible chat endpoint (prefix caching enabled if you use
  non-zero cache ratios)

### Quick start

```bash
cd aiperf

# Local server, default shapes/cache ratios/gates:
./run_aiperf_workload_shapes.sh --model amd/Kimi-K2.7-Code-MXFP4 --port 8000 --gpus 8

# One ad-hoc shape (simple mode):
./run_aiperf_workload_shapes.sh --model <id> \
    --isl 8000 --osl 1000 --concurrency 1,2,4,8,16 --cache 90

# Remote / dedicated endpoint:
./run_aiperf_workload_shapes.sh --model deepseek-ai/DeepSeek-V4-Flash \
    --url https://host/ --api-key KEY \
    --isl 5000 --osl 1000 --cache 100 \
    --concurrency 16,32,64,128,256 --skip-health-check

# Parse an existing sweep by hand:
python3 parse_sweep.py --sweep-dir /path/to/sweep_aggregate \
    --gpus 8 --osl-target 1000 --itl-p50-target 25 --out-csv summary.csv
```

Run either script with `--help` for the full flag list.