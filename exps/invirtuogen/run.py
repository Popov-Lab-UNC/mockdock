"""
InVirtuoGen x mockdock Benchmark Host Runner
============================================
Orchestrates the MockDock benchmark evaluation for InVirtuoGen.
Starts a local TCP server for scoring and runs the InVirtuoGen
optimization loop in a separate virtual environment subprocess.
"""

from __future__ import annotations

import logging
import os
import pathlib
import datetime
import socket
import json
import threading
import subprocess
import time
import click

from mockdock import MDOracle

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
BENCHMARKS = ["DPP4", "CHK1", "ITK", "PEPCK", "TTK", "VEGFR2", "PptT"]


class OracleServer:
    """A lightweight TCP socket server wrapping MDOracle to score SMILES batches."""
    
    def __init__(self, oracle: MDOracle, host: str = "127.0.0.1"):
        self.oracle = oracle
        self.host = host
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Bind to port 0 to let the OS select an available port
        self.server_socket.bind((self.host, 0))
        self.port = self.server_socket.getsockname()[1]
        self.server_socket.listen(1)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        log.info("Oracle server started on %s:%d", self.host, self.port)

    def _run(self):
        while self.running:
            try:
                self.server_socket.settimeout(1.0)
                client_socket, _ = self.server_socket.accept()
            except socket.timeout:
                continue
            except Exception:
                break
            
            client_thread = threading.Thread(
                target=self._handle_client, args=(client_socket,), daemon=True
            )
            client_thread.start()

    def _handle_client(self, client_socket: socket.socket):
        # High timeout because molecular docking can take several minutes
        client_socket.settimeout(600.0)
        buffer = ""
        while self.running:
            try:
                data = client_socket.recv(4096)
                if not data:
                    break
                buffer += data.decode("utf-8")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line.strip():
                        continue
                    req = json.loads(line)
                    action = req.get("action")
                    if action == "score":
                        smiles_list = req.get("smiles", [])
                        scores = self.oracle.score(smiles_list)
                        res = {
                            "status": "ok",
                            "scores": scores,
                            "budget_remaining": self.oracle.budget_remaining,
                        }
                        client_socket.sendall((json.dumps(res) + "\n").encode("utf-8"))
                    elif action == "status":
                        res = {
                            "status": "ok",
                            "budget_remaining": self.oracle.budget_remaining,
                        }
                        client_socket.sendall((json.dumps(res) + "\n").encode("utf-8"))
                    elif action == "get_initial_compounds":
                        n = req.get("n", 25)
                        initial_df = self.oracle.get_initial_compounds()
                        compounds = []
                        if not initial_df.is_empty():
                            subset = initial_df.head(n)
                            smiles_col = "canonical_smiles"
                            for row in subset.iter_rows(named=True):
                                smiles = row[smiles_col]
                                score = row.get("score") if "score" in row else None
                                if score is None:
                                    scores = self.oracle.score([smiles])
                                    score = scores.get(smiles, 0.0)
                                compounds.append({"smiles": smiles, "score": score})
                        res = {
                            "status": "ok",
                            "compounds": compounds
                        }
                        client_socket.sendall((json.dumps(res) + "\n").encode("utf-8"))
                    elif action == "finish":
                        res = {"status": "ok"}
                        client_socket.sendall((json.dumps(res) + "\n").encode("utf-8"))
                        break
            except Exception as e:
                log.error("Error in oracle server client handler: %s", e)
                break
        client_socket.close()

    def stop(self):
        self.running = False
        self.server_socket.close()
        self.thread.join(timeout=2.0)


def run_benchmark(
    benchmark: str,
    budget: int,
    seed: int,
    outputs_root: pathlib.Path,
    ckpt_path: str,
    invgen_python: str,
    invgen_dir: str,
    batch_size: int,
    clip_reward_upper_bound: bool,
    run_parent: pathlib.Path,
    n_warmup: int = 25,
):
    log.info("=" * 60)
    log.info(" Benchmark : %s", benchmark)
    log.info(" Budget    : %d", budget)
    log.info(" Seed      : %d", seed)
    log.info("=" * 60)

    output_dir = outputs_root / benchmark
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_run_dir = run_parent / benchmark
    benchmark_run_dir.mkdir(parents=True, exist_ok=True)

    # Initialize host-side MDOracle
    oracle = MDOracle(
        benchmark,
        budget=budget,
        run_dir=benchmark_run_dir,
        clip_reward_upper_bound=clip_reward_upper_bound,
    )
    log.info("Run directory: %s", oracle.run_dir)

    # Scaffold fragment conditioning
    scaffold = oracle.fragment_smiles_with_dummies
    if not scaffold:
        log.error("Scaffold conditioning is required for benchmark %s", benchmark)
        return
    log.info("Scaffold conditioning: %s", scaffold)

    # Start the TCP socket server
    server = OracleServer(oracle)
    server.start()

    t0 = time.time()

    # Determine GPU device
    cuda_device = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    # Handle multi-GPU allocation or cpu fallback
    device_str = "cuda:0" if torch_cuda_is_available_host() else "cpu"

    # Launch optimizer.py in the isolated InVirtuoGen environment
    optimizer_script = (SCRIPT_DIR / "optimizer.py").as_posix()
    
    cmd = [
        invgen_python,
        optimizer_script,
        "--ckpt", ckpt_path,
        "--device", device_str,
        "--oracle-port", str(server.port),
        "--benchmark", benchmark,
        "--fragment-prompt", scaffold,
        "--seed", str(seed),
        "--budget", str(budget),
        "--batch-size", str(batch_size),
        "--dt", "0.01",
        "--eta", "999",
        "--temperature", "1.0",
        "--first_pop_size", "10",
        "--use_prompter",
        "--use_mutation",
        "--train_mutation",
        "--aggressive_bandit",
        "--n-warmup", str(n_warmup),
    ]

    log.info("Launching InVirtuoGen subprocess...")
    try:
        # Run subprocess with InVirtuoGen directory as cwd so tokenizer relative paths resolve
        result = subprocess.run(
            cmd,
            cwd=invgen_dir,
            capture_output=True,
            text=True,
            timeout=7200, # 2 hours timeout
        )
    except subprocess.TimeoutExpired:
        log.warning("InVirtuoGen subprocess timed out after 2 hours.")
        result = None

    # Shutdown the TCP server
    server.stop()

    if result is not None:
        if result.returncode != 0:
            log.error("Optimizer subprocess failed with return code %d", result.returncode)
            for line in result.stderr.splitlines():
                log.error("[subprocess-err] %s", line)
        else:
            log.info("Optimizer subprocess finished successfully.")
            # Optionally print some stdout lines
            for line in result.stdout.splitlines()[-20:]:
                log.info("[subprocess-out] %s", line)

    total_time = time.time() - t0
    oracle_time = oracle._total_prep_time + oracle._total_dock_time + oracle._total_analysis_time
    total_gen_time_sec = max(0.0, total_time - oracle_time)
    n_gen = max(1, len(oracle.results_df))

    # Export top 10 poses
    try:
        oracle.export_top_poses(n=10)
    except Exception as exc:
        log.warning("Could not export top poses: %s", exc)

    # Save metrics JSON file
    oracle.save_metrics(
        extra={
            "model": "invirtuogen",
            "seed": seed,
            "total_generation_time_sec": total_gen_time_sec,
            "n_generated_ligands": n_gen,
        }
    )

    log.info(
        "Benchmark %s complete. Budget used: %d/%d, rounds: %d",
        benchmark,
        oracle.budget_used,
        oracle.max_budget,
        oracle.generation_round,
    )


def torch_cuda_is_available_host() -> bool:
    """Preflight check if CUDA is available on host (or SLURM job allocation)"""
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except ImportError:
        pass

    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return True

    import shutil
    if shutil.which("nvidia-smi") is not None:
        return True

    return False


@click.command()
@click.option(
    "--benchmark",
    "benchmarks",
    multiple=True,
    type=click.Choice(BENCHMARKS, case_sensitive=False),
    help="Benchmark(s) to run. Defaults to all configured benchmarks.",
)
@click.option("--budget", default=1000, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--out", default=None)
@click.option(
    "--ckpt",
    default="/work/users/s/h/shuhang/invirtuogen/checkpoints/invirtuo_gen_big.ckpt",
    help="Path to the InVirtuoGen checkpoint file.",
)
@click.option(
    "--invgen-python",
    default="/work/users/s/h/shuhang/invirtuogen/.venv/bin/python",
    help="Path to python inside InVirtuoGen virtual environment.",
)
@click.option(
    "--invgen-dir",
    default="/work/users/s/h/shuhang/invirtuogen",
    help="Root directory of InVirtuoGen repo.",
)
@click.option("--batch-size", default=64, show_default=True)
@click.option("--run-dir", default=None)
@click.option(
    "--clip-reward-upper-bound/--no-clip-reward-upper-bound",
    default=False,
    show_default=True,
)
@click.option(
    "--n-warmup",
    default=25,
    show_default=True,
    help="Number of initial compounds to use as warmup.",
)
def main(
    benchmarks,
    budget,
    seed,
    out,
    ckpt,
    invgen_python,
    invgen_dir,
    batch_size,
    run_dir,
    clip_reward_upper_bound,
    n_warmup,
):
    benchmarks = list(benchmarks) if benchmarks else BENCHMARKS
    outputs_root = pathlib.Path(out) if out else SCRIPT_DIR / "outputs"
    outputs_root.mkdir(parents=True, exist_ok=True)

    if run_dir is not None:
        run_parent = pathlib.Path(run_dir)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_parent = SCRIPT_DIR / f"run_{ts}"
    run_parent.mkdir(parents=True, exist_ok=True)

    for bm in benchmarks:
        run_benchmark(
            benchmark=bm,
            budget=budget,
            seed=seed,
            outputs_root=outputs_root,
            ckpt_path=ckpt,
            invgen_python=invgen_python,
            invgen_dir=invgen_dir,
            batch_size=batch_size,
            clip_reward_upper_bound=clip_reward_upper_bound,
            run_parent=run_parent,
            n_warmup=n_warmup,
        )

    log.info("All benchmarks complete.")


if __name__ == "__main__":
    main()
