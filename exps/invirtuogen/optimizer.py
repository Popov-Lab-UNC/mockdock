import sys
import os
import re
import copy
import random
import argparse
import socket
import json
import types

# 1. Setup python path to import in_virtuo_gen and in_virtuo_reinforce
sys.path.insert(0, os.getcwd())

# 2. Monkeypatch wandb to prevent login/network/logging errors in the cluster environment
import wandb
wandb.init = lambda *args, **kwargs: wandb.run
wandb.log = lambda *args, **kwargs: None
wandb.Image = lambda *args, **kwargs: None
wandb.finish = lambda *args, **kwargs: None
if not hasattr(wandb, "run") or wandb.run is None:
    wandb.run = types.SimpleNamespace()
wandb.run.log_code = lambda *args, **kwargs: None

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

from in_virtuo_gen.models.invirtuofm import InVirtuoFM
from in_virtuo_gen.train_utils.metrics import evaluate_smiles
from in_virtuo_reinforce.genetic_ppo import InVirtuoFMOptimizer, OptimizerConfig
from in_virtuo_reinforce.utils import rank_based_sampling, decompose_smiles

# Helper for attachment point enumeration
def enumerate_attach_points(smi: str) -> str:
    counter = {"i": 0}
    def repl(_):
        counter["i"] += 1
        return f"[{counter['i']}*]"
    return re.sub(r"\[\*\]|\*", repl, smi)

# Prepend scaffold to prompts helper
def prepend_scaffold_to_prompts(prompts, scaffold_tokens, pad_token_id):
    B, L = prompts.shape
    scaffold_tokens = scaffold_tokens.to(prompts.device)
    Ls = len(scaffold_tokens)
    new_prompts = torch.full((B, L + Ls), pad_token_id, dtype=torch.long, device=prompts.device)
    for b in range(B):
        valid_tokens = prompts[b][prompts[b] != pad_token_id]
        combined = torch.cat([scaffold_tokens, valid_tokens])
        new_prompts[b, :len(combined)] = combined
    return new_prompts

# Simple Socket Client
class OracleClient:
    def __init__(self, host='127.0.0.1', port=None):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))

    def score(self, smiles_list):
        req = {"action": "score", "smiles": smiles_list}
        self.socket.sendall((json.dumps(req) + "\n").encode('utf-8'))
        
        buffer = ""
        while True:
            data = self.socket.recv(4096)
            if not data:
                raise RuntimeError("Socket connection closed by server")
            buffer += data.decode('utf-8')
            if "\n" in buffer:
                line, _ = buffer.split("\n", 1)
                res = json.loads(line)
                return res["scores"]

    def get_budget_remaining(self):
        req = {"action": "status"}
        self.socket.sendall((json.dumps(req) + "\n").encode('utf-8'))
        buffer = ""
        while True:
            data = self.socket.recv(4096)
            if not data:
                raise RuntimeError("Socket connection closed by server")
            buffer += data.decode('utf-8')
            if "\n" in buffer:
                line, _ = buffer.split("\n", 1)
                res = json.loads(line)
                return res["budget_remaining"]

    def get_initial_compounds(self, n=25):
        req = {"action": "get_initial_compounds", "n": n}
        self.socket.sendall((json.dumps(req) + "\n").encode('utf-8'))
        buffer = ""
        while True:
            data = self.socket.recv(4096)
            if not data:
                raise RuntimeError("Socket connection closed by server")
            buffer += data.decode('utf-8')
            if "\n" in buffer:
                line, _ = buffer.split("\n", 1)
                res = json.loads(line)
                return res["compounds"]

    def close(self):
        try:
            req = {"action": "finish"}
            self.socket.sendall((json.dumps(req) + "\n").encode('utf-8'))
        except Exception:
            pass
        self.socket.close()

# Drop-in replacement for the Oracle class
class MockDockOracleAdapter:
    def __init__(self, client, max_oracle_calls=1000):
        self.client = client
        self.max_oracle_calls = max_oracle_calls
        self.mol_buffer = {}
        self.sa_scorer = lambda x: [0.0] * len(x) if isinstance(x, list) else 0.0
        self.diversity_evaluator = lambda x: 0.0
        self.predicted_auc = 0.0
        self.same_auc = 0
        self.name = "mockdock"
        self.task_label = "mockdock"
        self._call_count = 0

    @property
    def budget(self):
        return self.max_oracle_calls

    @property
    def finish(self):
        return self.client.get_budget_remaining() <= 0

    def score_smi(self, smi):
        scores = self([smi])
        return scores[0]

    def __call__(self, smiles_lst):
        is_str = isinstance(smiles_lst, str)
        if is_str:
            smiles_lst = [smiles_lst]
        
        if self.finish:
            return 0.0 if is_str else [0.0] * len(smiles_lst)
        
        # Call host socket server to score the batch
        scores_dict = self.client.score(smiles_lst)
        scores = [scores_dict.get(smi, 0.0) for smi in smiles_lst]
        
        for smi, score in zip(smiles_lst, scores):
            self._call_count += 1
            # Cache the latest/highest score for PMO population selections
            if smi not in self.mol_buffer:
                self.mol_buffer[smi] = [score, self._call_count]
                
        return scores[0] if is_str else scores

    def assign_evaluator(self, evaluator):
        pass

    def sort_buffer(self):
        self.mol_buffer = dict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    def log_intermediate(self, mols=None, scores=None, finish=False):
        pass

    def save_result(self, suffix=None):
        pass

    def save_result_tot(self, suffix=None):
        pass

    def save_qed_sa(self, oracle):
        pass

class MockDockInVirtuoOptimizer(InVirtuoFMOptimizer):
    def __init__(self, client, scaffold_tokens, *args, **kwargs):
        # We must disable some BaseOptimizer constructor setup that requires TDC oracles
        kwargs["oracle"] = "mockdock"
        kwargs["target"] = ""
        
        # Add defaults for required OptimizerConfig fields if not present
        kwargs.setdefault("start_t", 0.0)
        kwargs.setdefault("use_prescreen", False)
        kwargs.setdefault("start_seed", kwargs.get("seed", 0))
        kwargs.setdefault("start_rank", 0)

        # Filter kwargs to only contain valid fields to avoid unexpected keyword argument errors
        # in OptimizerConfig constructor. Valid keys include those in OptimizerConfig plus base optimizer keys
        valid_keys = set(OptimizerConfig.__dataclass_fields__.keys()) | {"device", "output_dir", "max_oracle_calls"}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

        super().__init__(*args, **filtered_kwargs)
        self.oracle = MockDockOracleAdapter(client, max_oracle_calls=kwargs.get("max_oracle_calls", 1000))
        self.scaffold_tokens = scaffold_tokens

    def initialize_population(self):
        B = int(self.config.first_pop_size * 1.1)
        n_oracle = [self.bandit.select_length() for _ in range(int(B))]
        prompt = self.scaffold_tokens.to(self.device) if self.scaffold_tokens is not None else None

        samples, init_ids = self.model.sample(
            num_samples=B,
            temperature=1.0,
            noise=1.0,
            oracle=n_oracle,
            eta=self.config.eta,
            prompt=prompt,
            force_prompt=True,
            return_uni=True,
        )

        valid, smiles, _ = evaluate_smiles(
            generated_ids=samples,
            tokenizer=self.tokenizer,
            return_values=True,
            print_flag=False,
            print_metrics=False,
            exclude_salts=False,
        )

        # Submit ALL B generated SMILES (including invalids) to the oracle to charge budget correctly.
        scores = self.oracle(smiles)

        # Keep only the valid ones for training population
        valid_samples = [torch.tensor(samples[i]) for i in valid][: self.config.first_pop_size]
        valid_init_ids = [torch.tensor(init_ids[i]) for i in valid][: self.config.first_pop_size]
        valid_smiles = [smiles[i] for i in valid][: self.config.first_pop_size]
        valid_scores = [scores[i] for i in valid][: self.config.first_pop_size]

        self.scores.extend(valid_scores)
        self.qed.extend([QED.qed(Chem.MolFromSmiles(smi)) for smi in valid_smiles if Chem.MolFromSmiles(smi) is not None])
        self.sa.extend([sascorer.calculateScore(Chem.MolFromSmiles(smi)) for smi in valid_smiles if Chem.MolFromSmiles(smi) is not None])

        for i in range(len(valid_samples)):
            if valid_scores[i] > 0:
                if self.config.use_prompter:
                    self.prompter.update_with_score(valid_smiles[i], valid_scores[i])
                    if not self.config.no_bandit:
                        self.prompter.bandit.update(len(valid_samples[i][valid_samples[i] != self.model.pad_token_id]), valid_scores[i])
                else:
                    if not self.config.no_bandit:
                        self.bandit.update(len(valid_samples[i][valid_samples[i] != self.model.pad_token_id]), valid_scores[i])

        self.train_ids = [s[s != self.model.pad_token_id] for s in valid_samples]
        self.train_x_0 = [idx[: len(s)] for idx, s in zip(valid_init_ids, self.train_ids)]
        self.train_scores = valid_scores

        # Get and process initial warmup compounds if requested
        n_warmup = getattr(self.config, "n_warmup", 0)
        warmup_smiles = []
        warmup_scores = []
        if n_warmup > 0:
            try:
                compounds = self.oracle.client.get_initial_compounds(n_warmup)
                for comp in compounds:
                    smi = comp["smiles"]
                    score = comp["score"]
                    if smi and score is not None and score > 0:
                        mol = Chem.MolFromSmiles(smi)
                        if mol is not None:
                            warmup_smiles.append(smi)
                            warmup_scores.append(score)
            except Exception as e:
                print(f"Error fetching/parsing warmup compounds: {e}")

        # If we have warmup compounds, tokenize them and add them to our lists!
        if len(warmup_smiles) > 0:
            print(f"Adding {len(warmup_smiles)} warmup/initial compounds to population PPO updates.")
            
            # 1. Add to the oracle client's buffer so the optimizer knows about them
            for smi, score in zip(warmup_smiles, warmup_scores):
                if smi not in self.oracle.mol_buffer:
                    self.oracle.mol_buffer[smi] = [score, len(self.oracle.mol_buffer) + 1]
                    
            # 2. Tokenize them using decompose_smiles and tokenizer
            warmup_samples = []
            warmup_init_ids = []
            for smi in warmup_smiles:
                frags = decompose_smiles(smi, max_frags=self.config.max_frags)
                token_ids = self.tokenizer.encode(" ".join(frags))
                seq = torch.tensor(token_ids)
                warmup_samples.append(seq)
                
                init_id = torch.randint(4, 203, (len(seq),))
                warmup_init_ids.append(init_id)
                
            # 3. Add to prompter/bandit
            for smi, score, seq in zip(warmup_smiles, warmup_scores, warmup_samples):
                if self.config.use_prompter:
                    self.prompter.update_with_score(smi, score)
                    if not self.config.no_bandit:
                        self.prompter.bandit.update(len(seq[seq != self.model.pad_token_id]), score)
                else:
                    if not self.config.no_bandit:
                        self.bandit.update(len(seq[seq != self.model.pad_token_id]), score)
            
            # 4. Extend self.scores, self.qed, self.sa
            self.scores.extend(warmup_scores)
            self.qed.extend([QED.qed(Chem.MolFromSmiles(smi)) for smi in warmup_smiles])
            self.sa.extend([sascorer.calculateScore(Chem.MolFromSmiles(smi)) for smi in warmup_smiles])

            # 5. Extend train lists for the PPO step
            warmup_train_ids = [s[s != self.model.pad_token_id] for s in warmup_samples]
            warmup_train_x_0 = [idx[: len(s)] for idx, s in zip(warmup_init_ids, warmup_train_ids)]
            
            self.train_ids.extend(warmup_train_ids)
            self.train_scores.extend(warmup_scores)
            self.train_x_0.extend(warmup_train_x_0)

        self.prev_novelty = 1.0

        if len(self.train_scores) > 0 and max(self.train_scores) > 0 and self.config.num_timesteps > 0:
            self.reinforce([item for item in self.train_ids], [item for item in self.train_scores], [item for item in self.train_x_0])
            self.add_experience()

        # Avoid deepcopy CUDA hangs by using load_state_dict
        self.prior.model.load_state_dict(self.model.model.state_dict())
        self.train_ids, self.train_scores, self.train_x_0 = [], [], []

    def generate_offspring_batch(self):
        n_oracle = []
        num_samples = int(min(200, int((self.config.offspring_size) // max(0.5, self.prev_novelty))) * 1.1)
        
        # Clamp to remaining budget to avoid over-generation at the very end
        budget_remaining = self.oracle.client.get_budget_remaining()
        if num_samples > budget_remaining:
            num_samples = max(1, budget_remaining)

        if self.config.use_prompter and len(self.scores) > 0 and max(self.scores) > 0:
            self.prompter.offspring_size = num_samples
            prompts, n_oracle = self.prompter.build_prompts_and_masks(dev=self.device)
            assert len(prompts) == num_samples
            
            # Prepend scaffold tokens to the prompter crossover prompts
            if self.scaffold_tokens is not None:
                prompts = prepend_scaffold_to_prompts(prompts, self.scaffold_tokens, self.model.pad_token_id)
        else:
            prompts = self.scaffold_tokens if self.scaffold_tokens is not None else None
            if prompts is not None and prompts.shape[0] != num_samples:
                prompts = prompts.unsqueeze(0).repeat(num_samples, 1)
            for i in range(num_samples):
                n_oracle.append(self.bandit.select_length())

        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            all_seqs, all_x_0 = self.model.sample(
                prompt=prompts,
                num_samples=num_samples,
                temperature=self.config.temperature,
                noise=0,
                oracle=n_oracle,
                start_t=self.config.start_t,
                fade_prompt=False,
                force_prompt=True,
                dt=self.config.dt,
                temperature_scaling=False,
                eta=self.config.eta,
                return_uni=True,
            )

        valid, all_smiles, metrics = evaluate_smiles(
            all_seqs, self.tokenizer, exclude_salts=True, return_values=True, print_flag=False, print_metrics=False, return_unique_indices=True
        )

        # Submit ALL generated SMILES (including invalids/duplicates) to the oracle to charge budget correctly.
        scores = self.oracle(all_smiles)

        # Keep track of novel/valid structures for stats
        valid_new_smiles = []
        valid_new_seqs = []
        valid_new_x_0 = []
        for i, smi in enumerate(all_smiles):
            if i in valid:
                if smi not in self.mol_buffer:
                    valid_new_smiles.append(smi)
                    valid_new_seqs.append(all_seqs[i])
                    valid_new_x_0.append(all_x_0[i])

        novelty = len(valid_new_seqs) / len(all_seqs) if len(all_seqs) > 0 else 0.0
        self.prev_novelty = novelty
        self.prev_validity = metrics["validity"]

        print(f"Batch metrics: validity={metrics['validity']:.2f}%, novelty={novelty:.2f}%, budget_remaining={budget_remaining}")

        # Update mol_buffer, prompter/bandit, and training lists
        for i, smi in enumerate(all_smiles):
            if i in valid:
                score = scores[i]
                
                if smi not in self.mol_buffer:
                    self.mol_buffer[smi] = [score, len(self.mol_buffer) + 1]
                    
                    mol_obj = Chem.MolFromSmiles(smi)
                    if mol_obj is not None:
                        qed_val = QED.qed(mol_obj)
                        sa_val = sascorer.calculateScore(mol_obj)
                        self.qed.append(qed_val)
                        self.sa.append(sa_val)
                    self.scores.append(score)

                    if score > 0:
                        if self.config.use_prompter:
                            self.prompter.update_with_score(smi, score)
                            if not self.config.no_bandit:
                                seq = torch.tensor(all_seqs[i])
                                self.prompter.bandit.update(len(seq[seq != self.model.pad_token_id]), score)
                        else:
                            if not self.config.no_bandit:
                                seq = torch.tensor(all_seqs[i])
                                self.bandit.update(len(seq[seq != self.model.pad_token_id]), score)

                    # Add to training buffer for PPO reinforcement
                    if len(self.train_ids) < self.config.tot_offspring_size - self.config.mutation_size * (not self.used_mutation and self.config.train_mutation):
                        self.train_ids.append(torch.tensor(all_seqs[i]))
                        self.train_scores.append(score)
                        self.train_x_0.append(all_x_0[i])

        if not self.used_mutation and not self.config.classic_ga:
            self.run_ga()
        self.set_c_neg(novelty)

    def evolve_population(self):
        self.generate_offspring_batch()
        if len(self.train_ids) >= self.config.tot_offspring_size and self.config.num_timesteps > 0:
            self.used_mutation = not self.config.use_mutation
            self.reinforce([item for item in self.train_ids], [item for item in self.train_scores], [item for item in self.train_x_0])

            # Avoid deepcopy CUDA hangs by using load_state_dict
            self.prior.model.load_state_dict(self.model.model.state_dict())
            self.add_experience()
            self.train_ids, self.train_scores, self.train_x_0 = [], [], []

        elif len(self.train_ids) >= self.config.tot_offspring_size:
            print("resetting train_ids, train_scores, train_x_0")
            self.train_ids, self.train_scores, self.train_x_0 = [], [], []

        best_score = sorted(self.mol_buffer.values(), key=lambda x: x[0], reverse=True)[0][0]
        self.global_step += 1

def main():
    parser = argparse.ArgumentParser(description="InVirtuoGen Optimization Subprocess")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--oracle-port", type=int, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--fragment-prompt", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--eta", type=float, default=999.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    
    # PMO optimization args
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--beta", type=float, default=0.99)
    parser.add_argument("--rl_lr", type=float, default=1e-4)
    parser.add_argument("--num_timesteps", type=int, default=10)
    parser.add_argument("--num_reinforce_steps", type=int, default=10)
    parser.add_argument("--use_prompter", action="store_true")
    parser.add_argument("--clip_eps", type=float, default=0.5)
    parser.add_argument("--c_neg", type=float, default=0.2)
    parser.add_argument("--use_mutation", action="store_true")
    parser.add_argument("--reverse_kl", action="store_true")
    parser.add_argument("--mutation_size", type=int, default=10)
    parser.add_argument("--experience_replay_size", type=int, default=24)
    parser.add_argument("--no_sample_uni", action="store_true")
    parser.add_argument("--train_mutation", action="store_true")
    parser.add_argument("--vocab_size", type=int, default=64)
    parser.add_argument("--first_pop_size", type=int, default=10)
    parser.add_argument("--max_frags", type=int, default=5)
    parser.add_argument("--aggressive_bandit", action="store_true")
    parser.add_argument("--no_bandit", action="store_true")
    parser.add_argument("--tot_offspring_size", type=int, default=None)
    parser.add_argument("--no_mask", action="store_true")
    parser.add_argument("--classic_ga", action="store_true")
    parser.add_argument("--entropy_bonus", action="store_true")
    parser.add_argument("--n-warmup", type=int, default=25)

    args = parser.parse_args()

    # Set up random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if "cuda" in args.device:
        torch.cuda.manual_seed_all(args.seed)

    # Establish socket connection to host oracle server
    print(f"Connecting to host oracle server on port {args.oracle_port}...")
    client = OracleClient(port=args.oracle_port)

    # Construct the config dictionary for MockDockInVirtuoOptimizer
    config_dict = dict(vars(args))
    config_dict["device"] = args.device
    config_dict["target"] = ""
    config_dict["oracle"] = "mockdock"
    config_dict["max_oracle_calls"] = args.budget
    config_dict["offspring_size"] = args.batch_size
    config_dict["big"] = "big" in args.ckpt
    config_dict["output_dir"] = "results_temp"
    config_dict["identifier"] = f"mockdock_{args.benchmark}_{args.seed}"
    config_dict["task"] = args.benchmark
    config_dict["alpha"] = 0.01 # dummy value
    
    if config_dict["tot_offspring_size"] is None:
        config_dict["tot_offspring_size"] = config_dict["offspring_size"] + config_dict["experience_replay_size"]

    # Model and Tokenizer loading
    model_device = torch.device(args.device)
    print(f"Loading checkpoint {args.ckpt}...")
    # Load model on cpu first, then tokenizer is accessed, and we will send model to device in constructor
    optimizer_model = InVirtuoFM.load_from_checkpoint(args.ckpt, map_location="cpu")
    tokenizer = optimizer_model.tokenizer

    # Prepare Scaffold prompt token IDs
    scaffold_labeled = enumerate_attach_points(args.fragment_prompt) + " "
    print(f"Scaffold SMILES (raw): {args.fragment_prompt}")
    print(f"Scaffold SMILES (labeled): {scaffold_labeled}")
    scaffold_token_list = tokenizer.encode(scaffold_labeled)
    scaffold_tokens = torch.tensor(scaffold_token_list, dtype=torch.long)
    print(f"Scaffold tokens: {scaffold_token_list}")

    # Initialize custom optimizer
    optimizer = MockDockInVirtuoOptimizer(
        client=client,
        scaffold_tokens=scaffold_tokens,
        **config_dict
    )

    # Overwrite the initialized model and prior with our loaded one
    optimizer.model = optimizer_model.to(model_device)
    optimizer.prior = InVirtuoFM.load_from_checkpoint(args.ckpt, map_location="cpu").to(model_device)
    optimizer.prior.model.requires_grad = False

    # Start the PMO optimization loop
    print("Starting optimization loop...")
    optimizer._optimize(oracle=None, config=None)

    # Clean up
    print("Optimization finished. Closing connection.")
    client.close()

if __name__ == "__main__":
    main()
