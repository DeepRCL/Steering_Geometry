import argparse
import json
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

from odesteer.utils import get_project_dir
from odesteer.utils.metric import QualityEvaluator, ToxicityEvaluator


toxicity_df_cols = [
    'Model',
    'Steering Method',
    'Toxicity',
    'Perplexity',
    'Dist-1',
    'Dist-2',
    'Dist-3',
]

steer_methods = [
    "NoSteer", "RepE", "ITI",  "CAA", "MiMiC", "LinAcT",
    "ODESteer"
]
steer_order_map = {val: i for i, val in enumerate(steer_methods)}


def get_steer_sort_key(value: str) -> tuple[int, str]:
    parts = value.split('-', 1)
    first_part = parts[0]
    remaining = parts[1] if len(parts) > 1 else ''
    
    first_order = steer_order_map.get(first_part, len(steer_methods))
    return (first_order, remaining)
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type = str, default = "Llama3.1-8B-Base")
    parser.add_argument("-l", "--layer_idx", type = int, default = 13)
    parser.add_argument("-b", "--batch_size", type = int, default = 10)
    parser.add_argument("-s", "--seed", type = int, default = 42)
    parser.add_argument("-d", "--display", action = "store_true")
    return parser.parse_args()


class ModelEvaluator:    
    def __init__(self, display: bool = False):
        self.toxicity_evaluator = ToxicityEvaluator(display = display)
        self.quality_evaluator = QualityEvaluator(device = "auto")
    
    def parse_file_info(self, file_path: Path) -> Tuple[str, str]:
        temp_lst = file_path.stem.split('-')[:-2]
        model = '-'.join(temp_lst[:4])
        steer_method = '-'.join(temp_lst[4:])
        return model, steer_method
    
    def load_existing_results(self, eval_path: Path, columns: List[str]) -> pd.DataFrame:
        if eval_path.exists():
            return pd.read_csv(eval_path)
        return pd.DataFrame(columns = columns)
    
    def evaluate_quality_metrics(self, outputs: List[str], batch_size: int = 10) -> Tuple[float, float, float, float]:
        ppls, dist_1, dist_2, dist_3 = self.quality_evaluator.batch_evaluate(outputs, batch_size)
        return ppls, dist_1, dist_2, dist_3 
    
    def evaluate_toxicity_metrics(self, outputs: List[str], batch_size: int = 10) -> Tuple[float, float, float, float]:
        toxicity_scores = self.toxicity_evaluator.batch_evaluate(outputs, batch_size)
        return toxicity_scores
    
    def save_detailed_results(
        self, 
        detailed_results: List[dict], 
        output_path: Path
    ) -> None:
        output_path.parent.mkdir(parents = True, exist_ok = True)
        with open(output_path, 'w') as f:
            for result in detailed_results:
                f.write(json.dumps(result) + '\n')
    
    def evaluate_toxicity_dataset(
        self, args: argparse.Namespace, 
        raw_result_dir: Path, 
        eval_path: Path,
    ) -> None:
        eval_df = self.load_existing_results(eval_path, toxicity_df_cols)
        
        for file_path in raw_result_dir.glob(f"**/{args.model}-l{args.layer_idx}-*-RealToxicityPrompts-seed{args.seed}.jsonl"):
            model, steer_method = self.parse_file_info(file_path)
            
            if steer_method in eval_df['Steering Method'].values:
                # print(f"Skipping {model} with {steer_method} on RealToxicityPrompts ...")
                continue
            
            print(f"Evaluating {model} with {steer_method} on RealToxicityPrompts ...")
            
            # Load and evaluate data
            df = pd.read_json(file_path, orient = "records", lines=True)
            outputs: List[str] = df.output.tolist()

            toxicity_scores: List[float] = self.toxicity_evaluator.batch_evaluate(outputs, batch_size = args.batch_size)
            mean_toxicity = np.nanmean(toxicity_scores)
            
            ppls, dist_1, dist_2, dist_3 = self.evaluate_quality_metrics(outputs, batch_size = args.batch_size)
            mean_ppls = np.nanmean(ppls)
            mean_dist_1 = np.nanmean(dist_1)
            mean_dist_2 = np.nanmean(dist_2)
            mean_dist_3 = np.nanmean(dist_3)
            
            # Save detailed results for each record
            detailed_eval_results_df = df.copy()
            detailed_eval_results_df['toxicity'] = toxicity_scores
            detailed_eval_results_df['ppl'] = ppls
            detailed_eval_results_df['dist_1'] = dist_1
            detailed_eval_results_df['dist_2'] = dist_2
            detailed_eval_results_df['dist_3'] = dist_3
            detailed_eval_results_dir = eval_path.parent.parent / "detailed_eval_results" / model
            detailed_eval_results_dir.mkdir(parents = True, exist_ok = True)
            detailed_eval_results_df.to_json(detailed_eval_results_dir / f"{model}-{steer_method}-RealToxicityPrompts-seed{args.seed}.jsonl", orient = "records", lines = True)
            
            # Add summary to evaluation DataFrame
            eval_df.loc[len(eval_df)] = [
                model, steer_method,
                mean_toxicity,
                mean_ppls, mean_dist_1, mean_dist_2, mean_dist_3
            ]
            
            eval_df = eval_df.sort_values(by = "Steering Method", key = lambda x: x.map(get_steer_sort_key))
            
            # Print results
            print(f"Average Toxicity Score: {mean_toxicity:.2f}")
            print(f"Average PPL: {mean_ppls:.2f}")
            print(f"Average Dist 1: {mean_dist_1:.2f}")
            print(f"Average Dist 2: {mean_dist_2:.2f}")
            print(f"Average Dist 3: {mean_dist_3:.2f}")
            
            eval_df.to_csv(eval_path, index=False)
        
        eval_df = eval_df.sort_values(by = "Steering Method", key = lambda x: x.map(get_steer_sort_key))
        eval_df.to_csv(eval_path, index = False)


def main() -> None:
    args = parse_args()
    output_dir = get_project_dir() / "results" / "toxicity"
    raw_result_dir = output_dir / "raw_outputs" / args.model
    eval_path = output_dir / "eval_results" / "stat_results" / f"{args.model}-l{args.layer_idx}-RealToxicityPrompts-seed{args.seed}.csv"
    eval_path.parent.mkdir(parents = True, exist_ok = True)
        
    evaluator = ModelEvaluator(args.display)
    evaluator.evaluate_toxicity_dataset(args, raw_result_dir, eval_path)


if __name__ == "__main__":
    main()