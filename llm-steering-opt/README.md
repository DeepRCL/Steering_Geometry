# `llm-steering-opt`: A library for optimizing steering vectors for LLMs.

This library contains functions for optimizing steering vectors for LLMs that induce or suppress specific behaviors.

In this repo, the optimization pipeline also supports Schwartz value geometry analysis and writes `geometry/geometry_metrics.json`, including:

- `spearman_rho`
- `pearson_r`
- `circular_distance_spearman`
- `hierarchical_distance_spearman`
- `lower_minus_opposite_cosine`

The last metric is an interpretable separation score:

- `lower_minus_opposite_cosine = same_lower_order_mean_cosine - opposite_higher_order_mean_cosine`

Higher positive values indicate that values from the same lower-order Schwartz family are much closer in steering space than theoretically opposite values.

For a quick tutorial on how to use this library, take a look at the notebook `quickstart.ipynb`.

For per-function documentation, refer to the docstrings in `steering_opt.py`. (Nicer documentation pages are currently under construction.)

For an in-depth look at what you can do with steering vector optimization, please refer to our paper [Investigating Generalization of One-shot LLM Steering Vectors](https://arxiv.org/pdf/2502.18862), where we apply steering vector optimization to tasks such as harmful behavior suppression in alignment-faking models, refusal suppression, and fictitious information generation modulation.

## Installation and usage

To install this module, run

    pip install .

Then, import the module in your own code with

    import steering_opt

## Credits and contact info

For any questions/comments/concerns, feel free to reach out to `jacob [dot] dunefsky [at] yale [dot] edu`.

If this library has been useful to you in your academic research, we'd be grateful if you cited us as

    @misc{dunefsky2025oneshot,
      title={Investigating Generalization of One-shot LLM Steering Vectors}, 
      author={Jacob Dunefsky and Arman Cohan},
      year={2025},
      eprint={2502.18862},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.18862}, 
    }
