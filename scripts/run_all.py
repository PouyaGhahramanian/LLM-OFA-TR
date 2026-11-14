#!/usr/bin/env python
"""Run the full THD-Temporal experiment grid on one or more GPUs (resumable).

Grid: 6 models x {pretrained, finetuned} x {static, ofa, ofa-adapt} = 36 stream runs + 6 fine-tuning jobs.
Finished jobs (metrics.json / finetune_summary.json present) are skipped, so the script can be re-started.

Example:
  python scripts/run_all.py --gpus 0 1
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
# slowest first, so the long jobs start early
MODEL_ORDER = ["deberta", "electra", "bert", "roberta", "albert", "distilbert"]
CONFIGS = ["ofa-adapt", "ofa", "static"]


def build_jobs(models):
    jobs = []
    for m in models:
        jobs.append({"name": f"finetune_{m}", "deps": [], "prio": 0,
                     "cmd": [PY, "scripts/finetune.py", "--model", m],
                     "done": REPO / "models" / f"{m}_finetuned" / "finetune_summary.json"})
    for setting, prio in (("finetuned", 1), ("pretrained", 2)):
        for m in models:
            for c in CONFIGS:
                jobs.append({"name": f"{setting}_{m}_{c}", "prio": prio,
                             "deps": [f"finetune_{m}"] if setting == "finetuned" else [],
                             "cmd": [PY, "scripts/stream.py", "--model", m, "--setting", setting, "--config", c],
                             "done": REPO / "results" / "stream" / f"{setting}_{m}_{c}" / "metrics.json"})
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", nargs="+", default=["0"])
    ap.add_argument("--models", nargs="+", default=MODEL_ORDER)
    args = ap.parse_args()

    logs = REPO / "logs"
    logs.mkdir(exist_ok=True)
    jobs = build_jobs(args.models)
    state = {j["name"]: ("done" if j["done"].exists() else "pending") for j in jobs}
    lock = threading.Lock()
    status_file = logs / "queue_status.json"

    def dump():
        status_file.write_text(json.dumps({"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "state": state},
                                          indent=1), encoding="utf-8")

    def next_job():
        with lock:
            ready = [j for j in jobs if state[j["name"]] == "pending"
                     and all(state[d] == "done" for d in j["deps"])]
            if not ready:
                return None, any(v in ("pending", "running") for v in state.values())
            j = sorted(ready, key=lambda x: (x["prio"], args.models.index(x["name"].split("_")[1])
                                              if x["name"].split("_")[1] in args.models else 0))[0]
            state[j["name"]] = "running"
            dump()
            return j, True

    def worker(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   TOKENIZERS_PARALLELISM="false")
        env.setdefault("HF_HOME", "/data/Pouya-data/hf_cache")
        while True:
            j, more = next_job()
            if j is None:
                if not more:
                    return
                # waiting on a dependency (or a failed one)
                with lock:
                    blocked = all(any(state[d] == "failed" for d in jj["deps"])
                                  for jj in jobs if state[jj["name"]] == "pending")
                    if blocked and not any(v == "running" for v in state.values()):
                        return
                time.sleep(20)
                continue
            t0 = time.time()
            with open(logs / f"{j['name']}.log", "w", encoding="utf-8") as log:
                log.write(f"# gpu={gpu} cmd={' '.join(j['cmd'])}\n")
                log.flush()
                rc = subprocess.call(j["cmd"], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            with lock:
                state[j["name"]] = "done" if (rc == 0 and j["done"].exists()) else "failed"
                print(f"[gpu {gpu}] {j['name']}: {state[j['name']]} in {(time.time() - t0) / 60:.1f} min",
                      flush=True)
                dump()

    dump()
    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus]
    for t in threads:
        t.start()
        time.sleep(5)
    for t in threads:
        t.join()
    failed = [k for k, v in state.items() if v != "done"]
    print("ALL DONE" if not failed else f"FINISHED WITH {len(failed)} NOT DONE: {failed}", flush=True)


if __name__ == "__main__":
    main()
