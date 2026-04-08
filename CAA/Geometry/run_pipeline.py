import os
import argparse
import json
import torch
import h5py

from .config import PipelineConfig, SCHWARTZ_CIRCUMPLEX_ORDER, safe_name
from .model_loader import load_model
from .data_loader import DataLoader
from .steering.caa import CAASteeringMethod
from .layer_selection import select_layer
from .evaluate import evaluate_steering
from .geometry import analyze_geometry

def get_steering_method(name: str):
    if name == "caa":
        return CAASteeringMethod()
    raise ValueError(f"Unknown steering method: {name}")

def run_pipeline(config: PipelineConfig, modules_to_run: list):
    torch.set_grad_enabled(False)
    
    data_loader = DataLoader(config.dataset_path, eval_split=config.eval_split, seed=config.seed)
    
    model_info = None
    if any(m in modules_to_run for m in ['extract', 'evaluate']):
        model_info = load_model(config.model_name, device=config.device)
        
    steering_method = get_steering_method(config.steering_method)
    
    layers_to_extract = list(range(
        int(model_info.n_layers * config.layer_start_frac) if model_info else 10,
        model_info.n_layers if model_info else 30
    ))
    
    vec_dir = config.subdir("vectors")
    act_dir = config.subdir("activations")
    
    vectors_all = {} # {value: {layer: tensor}}
    
    # MODULE 1 & 2: Extraction and Vector Computation
    if 'extract' in modules_to_run:
        print("Running Module 1 & 2: Extraction and Vector Computation...")
        
        for val in SCHWARTZ_CIRCUMPLEX_ORDER:
            pairs = data_loader.get_train_pairs(val)
            steering_method.clear_activations()
            
            vecs = steering_method.compute_vectors(pairs, model_info, layers_to_extract, val)
            vectors_all[val] = vecs
            
            val_safe = safe_name(val)
            val_vec_dir = os.path.join(vec_dir, val_safe)
            os.makedirs(val_vec_dir, exist_ok=True)
            
            for l_idx, vec_tensor in vecs.items():
                torch.save(vec_tensor, os.path.join(val_vec_dir, f"layer_{l_idx}.pt"))
                
            if config.save_activations:
                steering_method.save_activations(act_dir, val)
    else:
        # Load vectors from disk
        print("Loading precomputed steering vectors...")
        for val in SCHWARTZ_CIRCUMPLEX_ORDER:
            val_safe = safe_name(val)
            val_vec_dir = os.path.join(vec_dir, val_safe)
            vectors_all[val] = {}
            if os.path.exists(val_vec_dir):
                for f in os.listdir(val_vec_dir):
                    if f.startswith("layer_") and f.endswith(".pt"):
                        l_idx = int(f.split("_")[1].split(".")[0])
                        vectors_all[val][l_idx] = torch.load(os.path.join(val_vec_dir, f))

    requested = set(modules_to_run)
    if not requested.intersection({"layer_select", "evaluate", "geometry"}):
        print("Requested modules completed.")
        return
    
    # MODULE 3: Layer Selection
    selected_layer = config.layer_override
    if 'layer_select' in modules_to_run:
        print("Running Module 3: Layer Selection...")
        if selected_layer is None:
            # Need activations for cosine consistency
            activations_all = {}
            for val in SCHWARTZ_CIRCUMPLEX_ORDER:
                val_safe = safe_name(val)
                p = os.path.join(act_dir, f"{val_safe}.h5")
                activations_all[val] = {'pos': {}, 'neg': {}}
                if os.path.exists(p):
                    with h5py.File(p, "r") as f:
                        for pol in ['pos', 'neg']:
                            for l_str in f[pol].keys():
                                l_idx = int(l_str.split("_")[1])
                                activations_all[val][pol][l_idx] = {}
                                for sid in f[pol][l_str].keys():
                                    activations_all[val][pol][l_idx][sid] = torch.tensor(f[pol][l_str][sid][()])
            
            selected_layer = select_layer(config, vectors_all, activations_all)
        else:
            print(f"Skipping auto-selection. Using override layer: {selected_layer}")

    if not requested.intersection({"evaluate", "geometry"}):
        print("Requested modules completed.")
        return
    
    if selected_layer is None:
        # Load from previous run if skip layer_select
        layer_file = os.path.join(config.subdir("layer_selection"), "selected_layer.json")
        if os.path.exists(layer_file):
            with open(layer_file) as f:
                d = json.load(f)
                selected_layer = d["selected_layer"]
        else:
            raise ValueError("No selected layer found. Run layer_select module or provide --layer_override.")
            
    print(f"Target layer for steering: {selected_layer}")
    
    # Get vectors for selected layer
    target_vectors = {val: vectors_all[val][selected_layer] for val in SCHWARTZ_CIRCUMPLEX_ORDER}
            
    # MODULE 4: Evaluate
    if 'evaluate' in modules_to_run:
        print("Running Module 4: Evaluate Steering...")
        evaluate_steering(config, data_loader, model_info, steering_method, target_vectors, selected_layer)
        
    # MODULE 5: Geometry
    if 'geometry' in modules_to_run:
        print("Running Module 5: Geometry Analysis...")
        analyze_geometry(config, target_vectors)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--relations_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="CAA/Geometry/outputs")
    parser.add_argument("--modules", type=str, default="all", help="Comma-separated list: extract,layer_select,evaluate,geometry or 'all'")
    parser.add_argument("--layer_override", type=int, default=None)
    parser.add_argument("--alpha", type=str, default="0.5,1.0,2.0,4.0", help="Comma-separated alphas")
    
    args = parser.parse_args()
    
    config = PipelineConfig(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        relations_path=args.relations_path,
        output_dir=args.output_dir,
        layer_override=args.layer_override,
        alpha_values=[float(a) for a in args.alpha.split(",")]
    )
    
    config.save()
    
    if args.modules == "all":
        mods = ["extract", "layer_select", "evaluate", "geometry"]
    else:
        mods = args.modules.split(",")
        
    run_pipeline(config, mods)
