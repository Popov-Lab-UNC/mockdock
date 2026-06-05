"""
GenMol generation wrapper — runs inside the Apptainer container only.

Reads generation parameters from args and writes generated SMILES
(one per line) to stdout. The host-side run.py calls this as a subprocess,
keeping GenMol's rdkit-pypi completely isolated from mockdock's rdkit.
"""
from __future__ import annotations

import argparse
import random
import sys


def main():
    parser = argparse.ArgumentParser(description="GenMol SMILES generator")
    parser.add_argument("--model-path", required=True, help="Path to model_v2.ckpt")
    parser.add_argument(
        "--mode",
        choices=["scaffold", "evolve", "denovo"],
        default="scaffold",
        help="Generation mode",
    )
    parser.add_argument("--scaffold", default=None, help="Scaffold SMILES with * attachment points")
    parser.add_argument("--base-smiles", default=None, help="Base SMILES for evolve mode")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    from genmol.sampler import Sampler
    sampler = Sampler(args.model_path)

    samples: list[str] = []

    if args.mode == "scaffold":
        if not args.scaffold:
            print("ERROR: --scaffold required for scaffold mode", file=sys.stderr)
            sys.exit(1)
        try:
            samples = sampler.fragment_completion(
                args.scaffold,
                num_samples=args.num_samples,
                apply_filter=False,
            )
        except Exception as e:
            print(f"ERROR: fragment_completion failed: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.mode == "evolve":
        if not args.base_smiles:
            print("ERROR: --base-smiles required for evolve mode", file=sys.stderr)
            sys.exit(1)
        for _ in range(args.num_samples):
            try:
                smi = sampler.mask_modification(args.base_smiles)
                if smi:
                    smi = max(smi.split("."), key=len)
                    samples.append(smi)
            except Exception:
                pass

    else:  # denovo
        try:
            samples = sampler.de_novo_generation(num_samples=args.num_samples)
        except Exception as e:
            print(f"ERROR: de_novo_generation failed: {e}", file=sys.stderr)
            sys.exit(1)

    for smi in samples:
        if smi and smi.strip():
            print(smi.strip())


if __name__ == "__main__":
    main()
