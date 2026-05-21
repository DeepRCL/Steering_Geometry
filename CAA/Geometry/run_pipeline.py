import os
import argparse
import json
import torch
import h5py

from .config import PipelineConfig, SCHWARTZ_CIRCUMPLEX_ORDER, safe_name
from .model_loader import load_model
from .data_loader import DataLoader
from .steering.caa import CAASteeringMethod
from .steering.spherical import SphericalSteeringMethod
from .layer_selection import select_layer
from .evaluate import evaluate_steering
from .geometry import analyze_geometry

def get_steering_method(name: str, config: PipelineConfig):
    normalized_name = name.lower()
    if normalized_name == "caa":
        return CAASteeringMethod()
    if normalized_name in {"spherical", "sphericalsteer", "spherical_steer"}:
        return SphericalSteeringMethod(
            kappa=config.spherical_kappa,
            beta=config.spherical_beta,
            steer_position=config.spherical_steer_position,
        )
    raise ValueError(f"Unknown steering method: {name}")

def load_saved_activations(act_dir: str):
    activations_all = {}
    for val in SCHWARTZ_CIRCUMPLEX_ORDER:
        val_safe = safe_name(val)
        path = os.path.join(act_dir, f"{val_safe}.h5")
        activations_all[val] = {"pos": {}, "neg": {}}
        if not os.path.exists(path):
            continue
        with h5py.File(path, "r") as f:
            for polarity in ["pos", "neg"]:
                for layer_name in f[polarity].keys():
                    layer_idx = int(layer_name.split("_")[1])
                    activations_all[val][polarity][layer_idx] = {}
                    for sample_id in f[polarity][layer_name].keys():
                        activations_all[val][polarity][layer_idx][sample_id] = torch.tensor(
                            f[polarity][layer_name][sample_id][()]
                        )
    return activations_all

def apply_geometry_transform(vectors: dict, transform: str) -> dict:
    if transform == "none":
        return vectors

    if transform not in {"centered", "centered_renorm"}:
        raise ValueError(
            f"Unknown geometry transform: {transform}. "
            "Expected one of: none, centered, centered_renorm."
        )

    mean_vec = torch.stack(
        [vectors[val].detach().cpu().float() for val in SCHWARTZ_CIRCUMPLEX_ORDER]
    ).mean(dim=0)
    transformed = {
        val: vectors[val].detach().cpu().float() - mean_vec
        for val in SCHWARTZ_CIRCUMPLEX_ORDER
    }

    if transform == "centered_renorm":
        transformed = {
            val: vec / vec.norm().clamp_min(1e-12)
            for val, vec in transformed.items()
        }

    return transformed

def resolve_spherical_geometry_alpha(config: PipelineConfig) -> float:
    if config.spherical_geometry_alpha is not None:
        return float(config.spherical_geometry_alpha)

    summary_path = os.path.join(config.model_output_dir, "evaluation", "evaluation_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        best = summary.get("best_overall_alpha_by_mean_accuracy_gain", {}).get("alpha")
        if best is not None:
            return float(best)

    return float(max(config.alpha_values))

def save_geometry_vectors(config: PipelineConfig, vectors: dict, layer_idx: int, metadata: dict):
    out_dir = config.subdir("geometry_vectors")
    manifest = {
        "layer_idx": layer_idx,
        **metadata,
        "vectors": {},
    }
    for val, vec in vectors.items():
        val_safe = safe_name(val)
        filename = f"{val_safe}.pt"
        torch.save(vec.detach().cpu(), os.path.join(out_dir, filename))
        manifest["vectors"][val] = filename

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

def run_pipeline(config: PipelineConfig, modules_to_run: list):
    torch.set_grad_enabled(False)
    
    data_loader = DataLoader(config.dataset_path, eval_split=config.eval_split, seed=config.seed)
    
    model_info = None
    needs_model = any(m in modules_to_run for m in ['extract', 'evaluate'])
    needs_model = needs_model or ('layer_select' in modules_to_run and config.layer_selection_method == "eval_accuracy")
    if needs_model:
        model_info = load_model(config.model_name, device=config.device)
        
    steering_method = get_steering_method(config.steering_method, config)
    
    if model_info:
        if config.layer_override is not None:
            layers_to_extract = [config.layer_override]
        else:
            layer_start_idx = int(model_info.n_layers * config.layer_start_frac)
            layer_end_idx = model_info.n_layers if config.layer_end_frac >= 1.0 else int(model_info.n_layers * config.layer_end_frac)
            layer_end_idx = max(layer_start_idx + 1, min(model_info.n_layers, layer_end_idx))
            layers_to_extract = list(range(layer_start_idx, layer_end_idx))
    else:
        layers_to_extract = list(range(10, 30))
    
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

    available_layers = sorted({
        layer_idx
        for value_vectors in vectors_all.values()
        for layer_idx in value_vectors.keys()
    })
    if requested := set(modules_to_run):
        needs_vectors = bool(requested.intersection({"layer_select", "evaluate", "geometry"}))
        if needs_vectors and not available_layers:
            raise ValueError(
                "No steering vectors were found on disk for this run. "
                "Run the extract module successfully before layer selection, evaluation, or geometry."
            )

    if not requested.intersection({"layer_select", "evaluate", "geometry"}):
        print("Requested modules completed.")
        return
    
    # MODULE 3: Layer Selection
    selected_layer = config.layer_override
    if 'layer_select' in modules_to_run:
        print("Running Module 3: Layer Selection...")
        if selected_layer is None:
            # Need activations for cosine consistency.
            activations_all = load_saved_activations(act_dir)
            
            selected_layer = select_layer(
                config,
                vectors_all,
                activations_all,
                data_loader=data_loader,
                model_info=model_info,
                steering_method=steering_method,
            )
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
        geometry_vectors = target_vectors
        geometry_metadata = {
            "base_vector_type": "learned_steering_vector",
            "geometry_transform": config.geometry_transform,
        }

        if isinstance(steering_method, SphericalSteeringMethod):
            if config.spherical_geometry_vector == "displacement":
                geometry_alpha = resolve_spherical_geometry_alpha(config)
                activations_all = load_saved_activations(act_dir)
                geometry_vectors = steering_method.compute_displacement_vectors(
                    target_vectors,
                    activations_all,
                    selected_layer,
                    geometry_alpha,
                    source=config.spherical_geometry_source,
                )
                geometry_metadata.update(
                    {
                        "base_vector_type": "mean_spherical_displacement",
                        "displacement_definition": "mean(SphericalSteer(x) - x)",
                        "spherical_geometry_alpha": geometry_alpha,
                        "spherical_geometry_source": config.spherical_geometry_source,
                        "spherical_kappa": config.spherical_kappa,
                        "spherical_beta": config.spherical_beta,
                    }
                )
            elif config.spherical_geometry_vector != "prototype":
                raise ValueError(
                    f"Unknown spherical geometry vector mode: {config.spherical_geometry_vector}. "
                    "Expected displacement or prototype."
                )

        geometry_vectors = apply_geometry_transform(geometry_vectors, config.geometry_transform)
        save_geometry_vectors(config, geometry_vectors, selected_layer, geometry_metadata)
        analyze_geometry(config, geometry_vectors)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--relations_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="CAA/Geometry/outputs")
    parser.add_argument("--modules", type=str, default="all", help="Comma-separated list: extract,layer_select,evaluate,geometry or 'all'")
    parser.add_argument("--layer_override", type=int, default=None)
    parser.add_argument(
        "--layer_selection_method",
        type=str,
        default="normalized_l2",
        help="normalized_l2, projection_snr, linear_probe, or eval_accuracy",
    )
    parser.add_argument("--layer_start_frac", type=float, default=0.4, help="Start layer search/extraction at this fraction of depth")
    parser.add_argument("--layer_end_frac", type=float, default=1.0, help="Stop layer search/extraction before this fraction of depth")
    parser.add_argument("--alpha", type=str, default="0.5,1.0,2.0,4.0", help="Comma-separated alphas")
    parser.add_argument("--steering_method", type=str, default="caa", help="caa or spherical")
    parser.add_argument("--spherical_kappa", type=float, default=20.0, help="vMF concentration for spherical steering")
    parser.add_argument("--spherical_beta", type=float, default=-0.15, help="Trigger threshold p_H - p_T > beta")
    parser.add_argument("--spherical_steer_position", type=str, default="last", help="last or all")
    parser.add_argument("--spherical_geometry_alpha", type=float, default=None, help="Alpha used for displacement geometry; default uses best evaluated alpha")
    parser.add_argument("--spherical_geometry_source", type=str, default="neg", help="neg, pos, or all activations for displacement geometry")
    parser.add_argument("--spherical_geometry_vector", type=str, default="displacement", help="displacement or prototype")
    parser.add_argument("--geometry_transform", type=str, default="none", help="none, centered, or centered_renorm")
    
    args = parser.parse_args()
    
    config = PipelineConfig(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        relations_path=args.relations_path,
        output_dir=args.output_dir,
        layer_override=args.layer_override,
        layer_selection_method=args.layer_selection_method,
        layer_start_frac=args.layer_start_frac,
        layer_end_frac=args.layer_end_frac,
        alpha_values=[float(a) for a in args.alpha.split(",")],
        steering_method=args.steering_method,
        spherical_kappa=args.spherical_kappa,
        spherical_beta=args.spherical_beta,
        spherical_steer_position=args.spherical_steer_position,
        spherical_geometry_alpha=args.spherical_geometry_alpha,
        spherical_geometry_source=args.spherical_geometry_source,
        spherical_geometry_vector=args.spherical_geometry_vector,
        geometry_transform=args.geometry_transform,
    )
    
    config.save()
    
    if args.modules == "all":
        mods = ["extract", "layer_select", "evaluate", "geometry"]
    else:
        mods = args.modules.split(",")
        
    run_pipeline(config, mods)
