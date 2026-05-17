import argparse
import re
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from odesteer.utils import get_project_dir


ultrafeedback_df_cols = [
    'Model',
    'Steering Method',
    'RM Mean',
    'RM P90',
    'RM Win-Rate vs NoSteer',
]

steer_methods = [
    "NoSteer", "RepE", "ITI", "CAA", "MiMiC",
    "HPRSteer", "ReControl", "TruthFlowSteer", "LinAcT",
    "BODES"
]
steer_order_map = {val: i for i, val in enumerate(steer_methods)}


def normalize_prompt(prompt: str) -> str:
    """Normalize prompt by removing extra whitespace for matching."""
    normalized = re.sub(r"\s+", " ", prompt.strip())
    return normalized


def get_steer_sort_key(value):
    parts = value.split('-', 1)
    first_part = parts[0]
    remaining = parts[1] if len(parts) > 1 else ''

    first_order = steer_order_map.get(first_part, len(steer_methods))
    return (first_order, remaining)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type = str, default = "Llama3.1-8B-Base")
    parser.add_argument("-l", "--layer_idx", type = int, default = 13)
    parser.add_argument("-r", "--reward_model", type = str, default = "Skywork/Skywork-Reward-V2-Llama-3.1-8B")
    parser.add_argument("-b", "--batch_size", type = int, default = 4)
    parser.add_argument("-s", "--seed", type = int, default = 42)
    parser.add_argument("-d", "--display", action = "store_true")
    return parser.parse_args()


class UltrafeedbackEvaluator:
    def __init__(self, reward_model_name: str = "Skywork/Skywork-Reward-V2-Llama-3.1-8B", device: str = "auto"):
        """Initialize reward model for evaluation."""
        print(f"Loading reward model: {reward_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()

    @torch.no_grad()
    def evaluate_with_reward_model(
        self,
        prompts: list[str],
        outputs: list[str],
        batch_size: int = 4
    ) -> list[float]:
        """Evaluate responses with reward model."""
        rewards = []

        for i in tqdm(range(0, len(prompts), batch_size), desc="Evaluating with RM"):
            batch_prompts = prompts[i:i + batch_size]
            batch_outputs = outputs[i:i + batch_size]

            # Build inputs for each sample (following original implementation)
            batch_encodings = []
            for prompt, output in zip(batch_prompts, batch_outputs):
                # Format with chat template if available
                if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
                    try:
                        text = self.tokenizer.apply_chat_template(
                            [{"role": "user", "content": prompt},
                             {"role": "assistant", "content": output}],
                            tokenize=False,
                            add_generation_prompt=False
                        )
                    except Exception:
                        # Fallback if chat template fails
                        text = f"User: {prompt}\nAssistant: {output}"
                else:
                    # Use simple format for models without chat template
                    text = f"User: {prompt}\nAssistant: {output}"

                enc = self.tokenizer(text, truncation=True, max_length=2048, return_tensors="pt")
                batch_encodings.append(enc)

            # Pad sequences
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [enc["input_ids"].squeeze(0) for enc in batch_encodings],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id
            )
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                [enc["attention_mask"].squeeze(0) for enc in batch_encodings],
                batch_first=True,
                padding_value=0
            )

            # Move to device
            input_ids = input_ids.to(self.model.device)
            attention_mask = attention_mask.to(self.model.device)

            # Get rewards
            model_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            # Handle different reward model output formats
            if hasattr(model_outputs, 'score'):
                # If model returns a score attribute (some reward models)
                batch_rewards = model_outputs.score.float().cpu().tolist()
                if isinstance(batch_rewards, float):
                    batch_rewards = [batch_rewards]
            elif hasattr(model_outputs, 'logits'):
                # Standard logits output
                if model_outputs.logits.shape[-1] == 1:
                    # Single score per sample
                    batch_rewards = model_outputs.logits.squeeze(-1).float().cpu().tolist()
                else:
                    # Multiple classes - use softmax on last class (positive class)
                    batch_rewards = model_outputs.logits.softmax(-1)[:, -1].float().cpu().tolist()

                if isinstance(batch_rewards, float):
                    batch_rewards = [batch_rewards]
            else:
                # Direct output case
                batch_rewards = model_outputs.float().cpu().tolist()
                if isinstance(batch_rewards, float):
                    batch_rewards = [batch_rewards]

            rewards.extend(batch_rewards)

        return rewards

    def reward_model_win_rate(
        self,
        prompts: list[str],
        cand_a: list[str],
        cand_b: list[str],
        batch_size: int = 8
    ) -> float:
        """Compare two candidate lists A vs B on the same prompts; return A win-rate."""
        s_a = self.evaluate_with_reward_model(prompts, cand_a, batch_size=batch_size)
        s_b = self.evaluate_with_reward_model(prompts, cand_b, batch_size=batch_size)

        # Count wins and ties (for self-comparison)
        wins = 0
        ties = 0
        for a, b in zip(s_a, s_b):
            if a > b:
                wins += 1
            elif a == b:
                ties += 1

        # Ties count as 0.5 wins (standard convention)
        total_wins = wins + 0.5 * ties
        return total_wins / len(s_a) if s_a else 0.0

    def parse_file_info(self, file_path: Path) -> tuple[str, str]:
        temp_lst = file_path.stem.split('-')[:-2]
        model = '-'.join(temp_lst[:4])
        steer_method = '-'.join(temp_lst[4:])
        return model, steer_method

    def load_existing_results(self, eval_path: Path, columns: list[str]) -> pd.DataFrame:
        if eval_path.exists():
            return pd.read_csv(eval_path)
        return pd.DataFrame(columns = columns)

    def evaluate_ultrafeedback_dataset(
        self,
        args: argparse.Namespace,
        raw_result_dir: Path,
        eval_path: Path,
    ) -> None:
        eval_df = self.load_existing_results(eval_path, ultrafeedback_df_cols)

        # Load NoSteer outputs for comparison
        nosteer_outputs = {}
        nosteer_file = raw_result_dir / args.model / f"{args.model}-l{args.layer_idx}-NoSteer-UltrafeedbackBinarized-seed{args.seed}.jsonl"
        if nosteer_file.exists():
            print(f"Loading NoSteer baseline from {nosteer_file.name}")
            nosteer_df = pd.read_json(nosteer_file, orient="records", lines=True)
            for _, row in nosteer_df.iterrows():
                nosteer_outputs[normalize_prompt(row['prompt'])] = row['output']
            print(f"Loaded {len(nosteer_outputs)} NoSteer outputs for comparison")
        else:
            print(f"Warning: NoSteer baseline not found at {nosteer_file}")

        for file_path in raw_result_dir.glob(f"**/{args.model}-l{args.layer_idx}-*-UltrafeedbackBinarized-seed{args.seed}.jsonl"):
            model, steer_method = self.parse_file_info(file_path)

            if steer_method in eval_df['Steering Method'].values:
                print(f"Skipping {steer_method} - already evaluated")
                continue

            print(f"Evaluating {model} with {steer_method} on UltrafeedbackBinarized ...")

            # Load data
            df = pd.read_json(file_path, orient = "records", lines=True)
            prompts, outputs = df.prompt.tolist(), df.output.tolist()

            # Reward evaluation
            rewards = self.evaluate_with_reward_model(prompts, outputs, args.batch_size)
            rm_mean = np.nanmean(rewards)
            rm_p90 = float(np.quantile(rewards, 0.90))

            # Win-rate vs NoSteer
            rm_win_vs_nosteer = float('nan')
            if steer_method != "NoSteer" and len(nosteer_outputs) > 0:
                # Match NoSteer outputs using normalized prompts
                nosteer_matched = []
                current_matched = []
                matched_prompts = []
                for p, out in zip(prompts, outputs):
                    key = normalize_prompt(p)
                    if key in nosteer_outputs:
                        nosteer_matched.append(nosteer_outputs[key])
                        current_matched.append(out)
                        matched_prompts.append(p)

                if len(nosteer_matched) > 0:
                    rm_win_vs_nosteer = self.reward_model_win_rate(
                        matched_prompts, current_matched, nosteer_matched, batch_size=args.batch_size
                    )
                    print(f"→ NoSteer comparison: {len(nosteer_matched)} matched samples, win-rate: {rm_win_vs_nosteer:.4f}")
                else:
                    print("→ Warning: No matching NoSteer outputs found")
            elif steer_method == "NoSteer":
                rm_win_vs_nosteer = 0.5  # Self-comparison
                print("→ NoSteer vs NoSteer: 0.5 (baseline)")

            # Save detailed results
            detailed_eval_results_df = df.copy()
            detailed_eval_results_df['reward'] = rewards
            detailed_eval_results_dir = eval_path.parent.parent / "detailed_eval_results" / model
            detailed_eval_results_dir.mkdir(parents = True, exist_ok = True)
            detailed_eval_results_df.to_json(
                detailed_eval_results_dir / f"{model}-{steer_method}-UltrafeedbackBinarized-seed{args.seed}.jsonl",
                orient = "records",
                lines = True
            )

            # Add summary to evaluation DataFrame
            eval_df.loc[len(eval_df)] = [
                model, steer_method, rm_mean, rm_p90, rm_win_vs_nosteer
            ]

            eval_df = eval_df.sort_values(by = "Steering Method", key = lambda x: x.map(get_steer_sort_key))

            # Print results
            print(f"Average RM Mean: {rm_mean:.4f}")
            print(f"RM P90: {rm_p90:.4f}")
            print(f"RM Win-Rate vs NoSteer: {rm_win_vs_nosteer:.4f}")

            eval_df.to_csv(eval_path, index=False)

        eval_df = eval_df.sort_values(by = "Steering Method", key = lambda x: x.map(get_steer_sort_key))
        eval_df.to_csv(eval_path, index = False)


def main() -> None:
    """Main evaluation function."""
    args = parse_args()

    output_dir = get_project_dir() / "results" / "ultrafeedback"
    raw_result_dir = output_dir / "raw_outputs"
    eval_path = output_dir / "eval_results" / "stat_results" / f"{args.model}-l{args.layer_idx}-UltrafeedbackBinarized-seed{args.seed}.csv"
    eval_path.parent.mkdir(parents = True, exist_ok = True)

    evaluator = UltrafeedbackEvaluator(args.reward_model, device="auto")
    evaluator.evaluate_ultrafeedback_dataset(args, raw_result_dir, eval_path)


if __name__ == "__main__":
    main()
