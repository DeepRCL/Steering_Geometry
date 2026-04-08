import torch
from typing import List, Tuple, Callable, Optional, Union
import dataclasses
from contextlib import contextmanager
import mdmm
import numpy as np

# utility function
def _nested_list_max(l):
	if isinstance(l, list):
		return max((_nested_list_max(l_) for l_ in l)) if len(l) > 0 else float('-inf')
	return l

# context manager for running a HuggingFace Llama model with hooks
@contextmanager
def hf_hooks_contextmanager(model, hook_infos : List[Tuple[int, Callable]]):
	"""
	A context manager for running a HuggingFace Llama-like model with hooks (particularly steering hooks).

	Args:
		model (HuggingFace model): the model to hook into
		hook_infos: a list of pairs. The first element of each pair is the layer to hook into, and the second element is the hook function to attach.
	
	Example:
		# make and apply a steering hook to a HuggingFace model
		layer = 10
		hook_fn = steering_opt.make_steering_hook_hf(vector)
		# generate tokens while hooked
		with steering_opt.hf_hooks_contextmanager(model, [(layer, hook_fn)]):
			input_tokens = tokenizer("Hello, world.", return_tensors="pt")
			generated_tokens = model.generate(**input_tokens, max_new_tokens=10)
		# print generated tokens
		print(tokenizer.batch_decode(generated_tokens)[0])
	"""

	# set up hooks
	hooks = [ model.model.layers[cur_layer].register_forward_pre_hook(hook_fn) for cur_layer, hook_fn in hook_infos]
	# yield execution
	try:
		yield
	finally:
		# make sure to remove all hooks
		for hook in hooks: hook.remove()

# functions for making steering hooks
def make_abl_mat(v):
	"""
	Makes a matrix M from a vector v such that applying M to a vector x ablates the component of x in the direction of v (i.e. projects x onto the orthogonal complement of v).

	This is useful for ablation steering and clamp steering (see Sec. 2 of https://arxiv.org/pdf/2411.09003), where we want to remove all information in the direction of v from model activations x (ablation steering), or where we want to set the component of x in the direction of v to have a certain value (clamp steering).

	For example, the following steering hook clamps the component of x in the direction of v to be 5:
	`steering_opt.make_steering_hook_hf(5*v, make_abl_mat(v))`
	"""
	return (-torch.outer(v, v)/(v.norm().item()**2))

def make_steering_hook_hf(vector_, matrix=None, token=None):
	"""
	Makes a hook for steering the activations of a HuggingFace model.

	Args:
		vector_: a vector which will be added to the activations
		matrix (optional): a matrix, such that the product of that matrix with the activations will be added to the activations
		token (optional): an int or a slice denoting which tokens to apply steering to.
	"""
	if token is None:
		token = slice(None)
	def hook_fn(module, args):
		x = args[0]
		vector = vector_.to(x) if isinstance(vector_, torch.Tensor) else vector_
		x_sliced = x[:, token].detach().clone()
		x[:, token] = x_sliced + vector

		if matrix is not None:
			affine_term = torch.zeros_like(x)
			affine_term[:, token] = torch.einsum('...n, mn -> ...m', x_sliced, matrix.to(x))
			x = x + affine_term

		return x
	return hook_fn
 
# code duplication because tflens requires certain named args in hook_fn
def make_steering_hook_tflens(vector, matrix=None, token=None):
	"""
	Makes a hook for steering the activations of a TransformerLens model.

	Args:
		vector_: a vector which will be added to the activations
		matrix (optional): a matrix, such that the product of that matrix with the activations will be added to the activations
		token (optional): an int or a slice denoting which tokens to apply steering to.
	"""
	if token is None:
		token = slice(None)
	def hook_fn(x, hook):
		x_sliced = x[:, token]
		x[:, token] = x_sliced + vector

		if matrix is not None:
			affine_term = torch.zeros_like(x)
			affine_term[:, token] = torch.einsum('...n, mn -> ...m', x_sliced, matrix.to(x))
			x = x + affine_term

		return x
	return hook_fn

# hooks for getting activations
def make_activs_hook_hf(outlist):
	"""
	Makes a hook for storing the activations of a HuggingFace model.

	Args:
		outlist (list): a list to which the activations of the model will be appended
	"""
	def hook_fn(module, args):
		x = args[0]
		outlist.append(x)
		return x
	return hook_fn

## sampling-related functions

def get_completion_logprob(model, prompt, completion, tokenizer=None, coldness=1, return_all_probs=False, do_one_minus=False, do_log=True, eps=0, use_transformer_lens=True, **kwargs):
	"""
	Gets the model's log probabilities of a completion for a prompt.

	Args:
		model: the model to be used
		prompt (str): the input prompt to the model
		completion (str): the completion whose probability is to be obtained
		tokenizer (required for HuggingFace models): The tokenizer associated with the model
		coldness (float, 1 by default): The coldness/inverse temperature parameter used in computing probabilities
		return_all_probs (False by default): If True, then return the probabilities for each token. Otherwise, only return the joint probability of the whole sequence.
		do_one_minus (False by default): If True, then take the probability of the complement of the completion.
		do_log (True by default): If True, then use log probabilities (base 10).
		eps (float, 0 by default): Used to avoid underflow errors.
		use_transformer_lens (True by default): If True, then the model is a TransformerLens model. Otherwise, the model is a HuggingFace model. Note: for HuggingFace models, one can use the wrapper get_completion_logprob_hf().
		**kwargs: additional keyword arguments passed to the model.

	Returns:
		If return_all_probs is False, then returns the joint (log) probability of the sequence. Otherwise, returns a tuple containing the joint (log) probability of the sequence and the (log) probability of each token
	"""
	if use_transformer_lens:
		get_tokens = lambda prompt: model.to_tokens(prompt).tolist()[0]
		get_logits = lambda prompt: model(prompt, **kwargs)[0]
	else:
		if tokenizer is None:
			raise Exception("Not using TransformerLens -- but tokenizer is None!")
		get_tokens = lambda prompt: tokenizer(prompt).input_ids
		get_logits = lambda prompt: model(tokenizer(prompt, return_tensors='pt').input_ids, **kwargs).logits[0]

	prompt_tokens = get_tokens(prompt)
	prompt_len = len(prompt_tokens)
	all_tokens = get_tokens(prompt + completion)
	completion_tokens = all_tokens[prompt_len:]
	completion_len = len(completion_tokens)

	logits = get_logits(prompt + completion).float()

	probs = torch.nn.functional.softmax(logits*coldness, dim=-1)
	if do_one_minus: probs = 1 - probs

	cur_loss = 0 if do_log else 1
	if return_all_probs:
		all_probs = []
	for completion_token_idx in range(0, completion_len):
		completion_token = completion_tokens[completion_token_idx]
		prompt_token_idx = prompt_len+completion_token_idx-1
		target_prob = probs[prompt_token_idx, completion_token]
		if do_log: target_prob = torch.log(target_prob+eps)
		if do_log:
			cur_loss += target_prob
		else:
			cur_loss *= target_prob
		if return_all_probs: all_probs.append(target_prob.item())
	return cur_loss if not return_all_probs else (cur_loss, all_probs)

def get_completion_logprob_hf(model, prompt, completion, tokenizer, **kwargs):
	"""
	Gets a HuggingFace model's log probabilities of a completion for a prompt.

	Args:
		model: the model to be used
		prompt (str): the input prompt to the model
		completion (str): the completion whose probability is to be obtained
		tokenizer: the tokenizer associated with the model
		coldness (float, 1 by default): The coldness/inverse temperature parameter used in computing probabilities
		return_all_probs (False by default): If True, then return the probabilities for each token. Otherwise, only return the joint probability of the whole sequence.
		do_one_minus (False by default): If True, then take the probability of the complement of the completion.
		do_log (True by default): If True, then use log probabilities.
		eps (float, 0 by default): Used to avoid underflow errors.
		**kwargs: additional keyword arguments passed to the model.

	Returns:
		If return_all_probs is False, then returns the joint (log) probability of the sequence. Otherwise, returns a tuple containing the joint (log) probability of the sequence and the (log) probability of each token
	"""
	return get_completion_logprob(model, prompt, completion, tokenizer=tokenizer, use_transformer_lens=False, **kwargs)

@torch.no_grad()
def sample_most_likely_completions_hf(model, tokenizer, dst_prompt, src_prompt=None, k=5, iters=5, coldness=1, do_one_minus=False, gc_interval=3, use_total_probs=False, reverse=False, return_log_probs=False, return_token_probs=True, **kwargs):
	"""
	Performs greedy beam search sampling for a HuggingFace model.

	Args:
		model: the model to be used
		tokenizer: the tokenizer for the model
		dst_prompt (str): the prompt given as input to the model, whose completions are to be sampled
		src_prompt (optional, str): if this prompt is given, then the returned completions will be those that maximize the *difference* in probability between when dst_prompt is used as the input versus when src_prompt is used.
		k (int, 5 by default): the number of beams (and the number of completions to return).
		iters (int, 5 by default): the number of tokens to sample.
		coldness (float, 1 by default): the coldness/inverse temperature parameter that affects the entropy of the model's distribution.
		do_one_minus (False by default): if True, then when computing completion probabilities, take the probability of the complement of the completion. (This means that the function will return the least-likely completions. This is useful in conjunction with src_prompt.)
		gc_interval (int, 3 by default): every gc_interval iterations, the garbage collector will be run to prevent OOMs.
		use_total_probs (False by default): if True, then return the joint probability of each completion rather than the probability of each individual token in each completion.
		reverse (False by default): if True, then return the least likely completions (similar to do_one_minus).
		return_log_probs (False by default): if True, then return log (base 10) probabilities of completions.
		**kwargs: additional keyword arguments that will be passed to the model.
	
	Returns:
		completions (list of strs): a list of the k sampled completions.
		completion_probs (list): if use_total_probs is True, then a list of the joint (log) probabilities of each completion. Otherwise, a list of lists, one for each completion, where each inner list contains the (log) probability of each token in that completion.
	"""

	src_logits = model(tokenizer(src_prompt, return_tensors='pt').input_ids).logits[:,-1].float() if src_prompt is not None else None
	dst_logits = model(tokenizer(dst_prompt, return_tensors='pt').input_ids).logits[:,-1].float()
	src_probs = torch.nn.functional.softmax(src_logits*coldness, dim=-1) if src_prompt is not None else 0
	dst_probs = torch.nn.functional.softmax(dst_logits*coldness, dim=-1)
	prob_diffs = dst_probs - src_probs
	prob_diffs = prob_diffs * (-1 if reverse else 1)
	top_prob_diffs, token_idxs = torch.topk(prob_diffs, k=k)
	cur_completions = tokenizer.batch_decode(token_idxs.T)
	cur_completion_probs = top_prob_diffs.T.tolist()

	i = 0
	for i in range(iters):
		if src_prompt is not None:
			src_logits = model(tokenizer([src_prompt + x for x in cur_completions], return_tensors='pt').input_ids).logits[:,-1].float()
			src_probs = torch.nn.functional.softmax(src_logits*coldness, dim=-1)
		else:
			src_probs = 0
		dst_logits = model(tokenizer([dst_prompt + x for x in cur_completions], return_tensors='pt').input_ids).logits[:,-1].float()
		dst_probs = torch.nn.functional.softmax(dst_logits*coldness, dim=-1)
		prob_diffs = dst_probs - src_probs
		prob_diffs = prob_diffs * (-1 if reverse else 1)

		if not use_total_probs:
			v, idxs = torch.topk(prob_diffs.flatten(), k=k)
		else:
			prod_val = torch.tensor(cur_completion_probs).prod(dim=-1).to(prob_diffs.device)
			total_prob_diffs = torch.einsum('nd, n -> nd', prob_diffs, prod_val)
			_, idxs = torch.topk(total_prob_diffs.flatten(), k=k)
			v = prob_diffs.flatten()[idxs]
			
		completion_idxs, token_idxs = torch.unravel_index(idxs, prob_diffs.shape)
		
		new_completions = []
		new_probs = []
		for completion_idx, token_idx, token_prob in zip(completion_idxs, token_idxs, v):
			new_completions.append(tokenizer.batch_decode([tokenizer(cur_completions[completion_idx], add_special_tokens=False).input_ids + [token_idx]])[0])
			new_probs.append(cur_completion_probs[completion_idx] + [token_prob.item()])
		cur_completions = new_completions
		cur_completion_probs = new_probs

	if gc_interval is not None and i+1 % gc_interval == 0:
		gc.collect()
		torch.cuda.empty_cache()
	cur_completion_probs = np.array(cur_completion_probs)
	if return_log_probs:
		cur_completion_probs = np.log(cur_completion_probs)
		if not return_token_probs: cur_completion_probs = np.sum(cur_completion_probs, axis=-1)
	else:
		if not return_token_probs: cur_completion_probs = np.prod(cur_completion_probs, axis=-1)
	return cur_completions, cur_completion_probs

## functions and classes for performing steering optimization ##

@dataclasses.dataclass
class TrainingDatapoint:
	"""
	A datapoint used for optimizing steering vectors.

	Members:
		prompt (str): the prompt used in this datapoint
		src_completions (optional, list of strs): a list of completions whose probabilities on this prompt should be minimized by the steering vector (i.e. suppression steering targets)
		dst_completions (optional, list of strs): a list of completions whose probabilities on this prompt should be maximized by the steering vector (i.e. promotion steering targets)
		src_completions_target_losses (optional, list of floats): a list of target losses for each suppression steering target, such that if all targets' losses fall below their respective given target losses, the optimization process will stop early
		dst_completions_target_losses (optional, list of floats): a list of target losses for each promotion steering target, such that if all targets' losses fall below their respective given target losses, the optimization process will stop early
		token (optional, slice or int): the tokens in this prompt that steering should be applied to when optimizing on this datapoint. If not given, then steering will be applied to all tokens.
		is_negative (False by default): if True, then the vector being optimized will be negated on this datapoint
	"""

	prompt: str
	src_completions: List[str] = dataclasses.field(default_factory=list)
	dst_completions: List[str] = dataclasses.field(default_factory=list)
	src_completions_target_losses: Optional[List[float]] = None
	dst_completions_target_losses: Optional[List[float]] = None
	token: Optional[Union[slice, int]] = None
	is_negative: bool = False

# a utility function used in output-constrained steering
def _mdmm_grad_accumulate_backward(mdmm_module):
	for c in mdmm_module:
		c_return = c()
		c_return.value.backward()

def optimize_vector(model, datapoints, layer,
	eps=1e-6, lr=0.01, max_iters=None, coldness=0.7,
	normalize_token_length=False, only_hook_prompt=False, use_transformer_lens=False, tokenizer=None,
	target_loss=None, return_info=True, do_target_loss_sum=True, return_loss_history=False, return_vec_history=False,
	target_loss_target_iters=1, satisfice=False, do_one_minus=True,
	max_norm=None, starting_norm=1, starting_vec=None,
	vector_clamp=None, affine_rank=None, max_affine_norm=2, starting_affine_norm=1,
	noise_scale=None, do_tangent_space_noise=True, do_noise_abl_relu=False, noise_iters=1, do_antipgd=False,
	do_output_constr=False, custom_output_constr_loss_func=None, custom_output_constr_pre_loss_func=None,
	output_constr_norm_initial_scale=1, output_constr_lr=None, max_output_constr_iters=None,
	debug=False,
):
	"""
	Optimize a steering vector on a set of datapoints.

	Args:
		Required args:
			model: the model to optimize the steering vector for
			datapoints (list of TrainingDatapoints): the list of TrainingDatapoints to optimize over
			layer (int or list of ints): the layer(s) to apply the steering vector to. If an int, then only optimize the steering vector at that layer. Otherwise, optimize the steering vector at all layers in the list.

		HuggingFace-related args:
			use_transformer_lens (False by default): set to True if the model being used is a TransformerLens model. If the model is a HuggingFace model, then set to False (and pass a value for tokenizer).
			tokenizer (required for HuggingFace models): the tokenizer associated with the HuggingFace model being used

		General hyperparams:
			eps (float, 1e-6 by default): a small constant used to prevent underflow errors
			lr (float, 0.01 by default): the learning rate for the optimizer
			coldness (float, 0.7 by default): the coldness/inverse temperature parameter used for computing probabilities

		Early stopping and loss-related args:
			max_iters (int, optional): if set, then terminate optimization after this many steps.
			target_loss (float, optional): if set, then stop the optimization when the loss stays below target_loss for target_loss_target_iters steps.
			do_target_loss_sum (True by default): used with target_loss. If True, then stop optimization when the sum of losses on all completions is below target_loss. If False, then stop optimization when each completion's loss is individually below target_loss.
			target_loss_target_iters (int, 1 by default): used for early stopping. If the loss stays below target_loss for this many optimization steps, or the absolute difference in loss from the previous step to the current step stays below eps for this many steps, then exit the optimization loop early.
			satisfice (False by default): if True, then penalize the vector based on the squared difference between the actual loss and target_loss.
			normalize_completion_length (False by default): if True, then divide the loss for each completion by the number of tokens in the completion.
			do_one_minus (True by default): if True, then for src completions, compute loss using the log of one minus the probability of each completion. If False, then for src completions, compute loss based on the negative log probability of each completion.

		Return value options:
			return_info (True by default): if True, then in addition to the steering vector itself, return a dictionary containing info about the optimization process.
			return_loss_history (False by default): if True, then return a list of losses after each optimization step.
			return_vec_history (False by default): if True, then return a list containing the steering vector after each optimization step.

		Misc args:
			only_hook_prompt (False by default): if True, then only apply the steering vector to tokens in the prompt (rather than all tokens, including those in the completion).
			debug (False by default): if True, then print out loss information at each optimization step.

		Norm-constrained steering args:
			max_norm (float, optional): the maximum norm of the steering vector. If set, then after each optimization step, if the vector's norm exceeds max_norm, it will be rescaled to max_norm.
			starting_norm (float, 1 by default): the starting norm of the steering vector. Before optimization, the steering vector is initialized to a randomly spherically-distributed vector with this norm.
			starting_vec (optional): if given, then this vector is used to initialize the vector being optimized, instead of the default random optimization.

		Clamp/affine steering args:
			vector_clamp (float, optional): if set, then optimize the vector to perform clamp steering. For a vector v, clamp steering with v on activations x entails ablating the component of x in the v direction, and then adding vector_clamp*v to the resulting activations.
			affine_rank (int, optional): if set, then perform affine steering, which optimizes a low-rank *steering matrix* in addition to a steering vector. For vector v and matrix M, affine steering with v and M on activations x entails mapping x to x + Mx + v. affine_rank is the rank of the matrix M.
			max_affine_norm (int, 2 by default): the low-rank steering matrix is internally factorized into two matrices M_l and M_r such that M = M_l^T M_r. After each optimization step, if the norm of any of the columns of M_l and M_r is greater than max_affine_norm, then that column is rescaled to have norm max_affine_norm. (This approach, instead of using e.g. spectral norm, was inspired by MELBO.)
			starting_affine_norm (int, 1 by default): the norm of the columns of M_l and M_r upon initialization.

		Noisy steering args:
			noise_scale (float, optional): if set, then add Gaussian noise multiplied by noise_scale to the activations at each optimization step (as a form of regularization).
			do_tangent_space_noise (True by default): if True, then project the noise vector onto the tangent space of the loss w.r.t. the steering vector. (Ideally, an approximation to prevent noise from inducing instability in the loss.)
			do_noise_abl_relu (False by default): only takes effect when do_tangent_space_noise is set. If True, then only ablates the component of the noise vector that points in the direction of decreasing loss (i.e. don't do ablation if the noise vector points in the direction of increasing loss). Ideally, this approximates choosing noise that only ever increases the loss or keeps it the same.
			noise_iters (1 by default): how many times noise should be sampled at each optimization step before updating the steering vector.
			do_antipgd (False by default): if True, then uses anti-correlated noise (see https://proceedings.mlr.press/v162/orvieto22a/orvieto22a.pdf)

		Output-constrained steering args:
			do_output_constr (False by default): if True, then perform output-constrained steering. This entails the following: after finding a steering vector that satisfies the target loss, then perform constrained minimization to optimize a vector with the smallest norm that does not increase the loss.
			output_constr_lr (float, optional): if set, then use this learning rate when performing output-constrained optimization instead of the learning rate used for the base steering optimization phase.
			max_output_constr_iters (int, optional): if set, then terminate output-constrained optimization after this many steps

			custom_output_constr_loss_func (function, optional, only supports TransformerLens): if set, then during the output-constrained phase, instead of minimizing the vector norm, minimize this loss function (while also ensuring that the vector's norm doesn't increase beyond its initial value). The function should have the following signature: custom_output_constr_loss_func(model, datapoints, layer, vector, matrix_left, matrix_right, only_hook_prompt, **kwargs). In this function, model, datapoints, layer, vector, and only_hook_prompt are the same as those passed to optimize_vector(). matrix_left and matrix_right are the two factor matrices used in affine steering if affine_rank is not None; otherwise, they are None. custom_output_constr_loss_func() can take optional kwargs, which will be set to the result of running custom_output_constr_pre_loss_func() before the first output-constrained optimization step (if custom_output_constr_pre_loss_func is not None). This function should return a scalar PyTorch tensor which can be backpropagated.
			custom_output_constr_pre_loss_func (function, optional, only supports TransformerLens): if set, then this function will be run before the first output-constrained optimization step. It should return a dictionary, which will be passed as kwargs to custom_output_constr_loss_func(). This function should have the signature custom_output_constr_pre_loss_func(model, datapoints, layer, vector, matrix_left, matrix_right, only_hook_prompt).
			output_constr_norm_initial_scale (float, 1 by default, only supports TransformerLens): only used with custom output-constrained loss functions. This is the amount by which the norm constraint (which prevents the vector's norm from increasing beyond its initial value) will be scaled. Constrained optimization works best when the constraints have similar scale, so this parameter allows the norm constraint to be placed at a similar scale to the custom loss function.
	
	Returns a tuple containing the following elements in order:
		vector: the steering vector which has been optimized
		matrix (optional): if affine_norm is not None, then the affine steering matrix which has been optimized
		info (optional): if return_info is True, then a dictionary containing the following items:
			iters: the number of optimization steps taken
			norm: the norm of the returned vector
			loss: if do_target_loss_sum is True, then the sum of losses. Otherwise, individual completion losses, and if return_loss_history is True, individual completion losses for all optimization steps are returned.
			vec_history (optional): if return_vec_history is True, then a list containing the steering vector after each optimization step
			output_constr_iters (optional): if output-constrained optimization was performed, then the number of optimization steps taken during the output-constrained optimization phase
			
	"""
	if use_transformer_lens:
		if output_constr_lr is None: output_constr_lr = lr
	if use_transformer_lens:
		d_model = model.cfg.d_model
		get_tokens = lambda prompt: model.to_tokens(prompt).tolist()[0]
		def get_hooked_logits(prompt, hook_infos):
			fwd_hooks = [(f'blocks.{cur_layer}.hook_resid_pre', hook_fn) for cur_layer, hook_fn in hook_infos]
			with model.hooks(fwd_hooks=fwd_hooks):
				return model(prompt)[0]
		make_steering_hook = make_steering_hook_tflens
	else:
		if tokenizer is None:
			raise Exception("Not using TransformerLens -- but tokenizer is None!")
		d_model = model.config.hidden_size
		get_tokens = lambda prompt: tokenizer(prompt).input_ids
		def get_hooked_logits(prompt, hook_infos):
			cur_tokens = tokenizer(prompt, return_tensors='pt').input_ids
			with hf_hooks_contextmanager(model, hook_infos):
				logits = model(cur_tokens, use_cache=False).logits[0]
			return logits 
		make_steering_hook = make_steering_hook_hf
	if starting_vec is None:
		with torch.no_grad():
			vector = torch.randn(d_model)
			vector = starting_norm * vector / vector.norm()
			vector = vector.cuda()
	else:
		vector = starting_vec.detach().clone()
	vector.requires_grad_(True)

	if affine_rank is not None:
		with torch.no_grad():
			matrix_left = torch.randn(affine_rank, d_model)
			matrix_right = torch.randn(affine_rank, d_model)

			matrix_left = torch.einsum('rm, r -> rm', matrix_left, starting_affine_norm/matrix_left.norm(dim=1))
			matrix_right = torch.einsum('rm, r -> rm', matrix_right, starting_affine_norm/matrix_right.norm(dim=1))
		matrix_left.requires_grad_(True)
		matrix_right.requires_grad_(True)
	else:
		matrix_left = None
		matrix_right = None

	all_src_completions_tokens = []
	all_dst_completions_tokens = []
	all_prompt_lens = []
	all_hook_fns = []

	# this array stores the individual loss for each completion for each datapoint
	# this is necessary for use with output-constrained optimization: in order to avoid
	#	using up too much memory, we introduce a separate constraint for each completion
	#	for each datapoint, rather than constraining the average loss over all completions.
	# doing so allows us to use gradient accumulation over our constraints.

	all_completion_losses = []
	loss_history = []
	vec_history = []
	def check_if_target_loss_hit(all_completion_losses, target_loss):
		target_loss_hit = True
		for datapoint, datapoint_losses in zip(datapoints, all_completion_losses):
			for i, src_completion_loss in enumerate(datapoint_losses[0]):
				cur_target_loss = target_loss if datapoint.src_completions_target_losses is None else datapoint.src_completions_target_losses[i]
				if src_completion_loss > cur_target_loss:
					target_loss_hit = False
					break
			if not target_loss_hit: break # god I wish that Python just let us use GOTOs
			for i, dst_completion_loss in enumerate(datapoint_losses[1]):
				cur_target_loss = target_loss if datapoint.dst_completions_target_losses is None else datapoint.dst_completions_target_losses[i]
				if dst_completion_loss > cur_target_loss:
					target_loss_hit = False
					break
			if not target_loss_hit: break
		return target_loss_hit

	for datapoint in datapoints:
		prompt = datapoint.prompt
		prompt_tokens = get_tokens(prompt)
		prompt_len = len(prompt_tokens)
		
		src_completions = datapoint.src_completions
		dst_completions = datapoint.dst_completions

		src_completions_tokens = []
		for src_completion in src_completions:
			src_completions_tokens.append(get_tokens(prompt + src_completion)[prompt_len:])
		dst_completions_tokens = []
		for dst_completion in dst_completions:
			dst_completions_tokens.append(get_tokens(prompt + dst_completion)[prompt_len:])

		all_completion_losses.append([
			[None for _ in range(len(src_completions))],
			[None for _ in range(len(dst_completions))],
		])

		all_src_completions_tokens.append(src_completions_tokens)
		all_dst_completions_tokens.append(dst_completions_tokens)
		all_prompt_lens.append(prompt_len)

	params = [vector]
	if affine_rank is not None:
		params = params + [matrix_left, matrix_right]

	def get_completion_loss(datapoint_idx, completion_idx, vector, matrix, is_src_completion=True, do_one_minus=True):
		datapoint = datapoints[datapoint_idx]
		prompt = datapoint.prompt
		prompt_len = all_prompt_lens[datapoint_idx]

		completion = datapoint.src_completions[completion_idx] if is_src_completion else datapoint.dst_completions[completion_idx]
		completion_tokens = all_src_completions_tokens[datapoint_idx][completion_idx] if is_src_completion else all_dst_completions_tokens[datapoint_idx][completion_idx]
		completion_len = len(completion_tokens)
		if datapoint.is_negative: vector = -vector

		if only_hook_prompt:
			if vector_clamp is None: hook_fn = make_steering_hook(vector, matrix=matrix, token=slice(0,prompt_len))
			else: hook_fn = make_steering_hook(vector_clamp*vector, matrix=make_abl_mat(vector), token=slice(0,prompt_len))
		else:
			if vector_clamp is None: hook_fn = make_steering_hook(vector, matrix=matrix, token=datapoint.token)
			else: hook_fn = make_steering_hook(vector_clamp*vector, matrix=make_abl_mat(vector), token=datapoint.token)
		if isinstance(layer, list):
			hook_infos = [ (cur_layer, hook_fn) for cur_layer in layer]
		else:
			hook_infos = [ (layer, hook_fn) ]
		
		cur_loss = 0

		logits = get_hooked_logits(prompt + completion, hook_infos)
		probs = torch.nn.functional.softmax(logits*coldness, dim=-1)

		for completion_token_idx in range(0, completion_len):
			completion_token = completion_tokens[completion_token_idx]
			prompt_token_idx = prompt_len+completion_token_idx-1
			target_prob = torch.log(1-probs[prompt_token_idx, completion_token] + eps) if is_src_completion and do_one_minus else torch.log(probs[prompt_token_idx, completion_token] + eps)
			if is_src_completion and not do_one_minus: target_prob = -target_prob
			if debug: print(datapoint_idx, completion_idx, completion_token_idx, is_src_completion, target_prob.item(), completion_token)

			cur_loss -= target_prob
		if normalize_token_length:
			cur_loss = cur_loss / completion_len

		return cur_loss

	prev_noise = 0
	def get_completion_loss_with_noise(datapoint_idx, completion_idx, vector, matrix, is_src_completion=True, do_one_minus=True):
		nonlocal prev_noise
		if noise_scale is None: return get_completion_loss(datapoint_idx, completion_idx, vector, matrix, is_src_completion=is_src_completion)

		noise = 0
		if noise_scale is not None:
			noise = torch.randn(vector.shape) * noise_scale
			noise = noise.detach()

		#if debug:
		#	with torch.no_grad():
		#		get_completion_loss(datapoint_idx, completion_idx, noise, matrix, is_src_completion=is_src_completion)

		if not do_tangent_space_noise:
			new_noise = noise
			if do_antipgd: new_noise = noise - prev_noise
			prev_noise = noise
			return get_completion_loss(datapoint_idx, completion_idx, vector + new_noise.to(device=vector.device), matrix, is_src_completion=is_src_completion)

		# time to do tangent space noise
		# here's the procedure:
		#	1. get gradient of loss at point
		#	2. remove gradient component from noise
		#	3. get loss at point+noise when adding steering vector
		zero_vec = torch.zeros_like(vector).requires_grad_(True)
		unsteered_loss = get_completion_loss(datapoint_idx, completion_idx, zero_vec, None, is_src_completion=is_src_completion)
		grad = torch.autograd.grad(outputs=unsteered_loss, inputs=zero_vec)[0]
		with torch.no_grad():
			abl_component = torch.dot(noise.to(grad), grad)/(grad.norm()**2)
			if do_noise_abl_relu:
				abl_component = -torch.nn.functional.relu(-abl_component)
			ablated_noise = noise.to(grad) + abl_component
		return get_completion_loss(datapoint_idx, completion_idx, vector + ablated_noise, matrix, is_src_completion=is_src_completion, do_one_minus=do_one_minus)

	optimizer = torch.optim.Adam(params, lr=lr)

	loss = None
	prev_loss = None
	iters = 0
	target_loss_cur_iters = 0
	prev_loss_cur_iters = 0

	while True:
		if max_iters is not None and iters > max_iters:
			if debug: print("Max iters reached.")	
			break
		if target_loss is not None and loss is not None:
			if do_target_loss_sum:
				if loss <= (target_loss if not satisfice else target_loss + eps):
					target_loss_cur_iters += 1
					if debug: print(f"Loss stopping threshold {target_loss} hit. Cur num iters: {target_loss_cur_iters}")
				else:
					target_loss_cur_iters = 0

			if not do_target_loss_sum:
				target_loss_hit = check_if_target_loss_hit(all_completion_losses, target_loss if not satisfice else target_loss + eps) 
				if target_loss_hit:
					target_loss_cur_iters += 1
					if debug: print(f"Loss stopping threshold {target_loss} hit. All completion losses: {all_completion_losses}. Cur num iters: {target_loss_cur_iters}")
				else:
					target_loss_cur_iters = 0

			if target_loss_cur_iters >= target_loss_target_iters:
				if debug: print(f"Loss stopping threshold {target_loss} hit. Breaking.")
				break

		optimizer.zero_grad()
		prev_loss = loss
		loss = 0

		for datapoint_idx, datapoint in enumerate(datapoints):
			for src_completion_idx in range(len(datapoint.src_completions)):
				for noise_iter in range(noise_iters):
					# I think that we have to do this every time to prevent "backwarding through graph a second time" errors
					if affine_rank is not None:
						matrix = matrix_left.T @ matrix_right
					else:
						matrix = None
					cur_loss = get_completion_loss_with_noise(datapoint_idx, src_completion_idx, vector, matrix, is_src_completion=True, do_one_minus=do_one_minus)
					loss += cur_loss.item()
					all_completion_losses[datapoint_idx][0][src_completion_idx] = cur_loss.item()
					if satisfice:
						cur_target_loss = target_loss if datapoint.src_completions_target_losses is None else datapoint.src_completions_target_losses[src_completion_idx]
						cur_loss = (cur_loss - cur_target_loss)**2
					cur_loss.backward()

			for dst_completion_idx in range(len(datapoint.dst_completions)):
				for noise_iter in range(noise_iters):
					# I think that we have to do this every time to prevent "backwarding through graph a second time" errors
					if affine_rank is not None:
						matrix = matrix_left.T @ matrix_right
					else:
						matrix = None
					cur_loss = get_completion_loss_with_noise(datapoint_idx, dst_completion_idx, vector, matrix, is_src_completion=False)
					loss += cur_loss.item()
					all_completion_losses[datapoint_idx][1][dst_completion_idx] = cur_loss.item()
					if satisfice:
						cur_target_loss = target_loss if datapoint.dst_completions_target_losses is None else datapoint.dst_completions_target_losses[dst_completion_idx]
						cur_loss = (cur_loss - cur_target_loss)**2
					cur_loss.backward()

		#loss /= len(datapoints)
		if prev_loss is not None and abs(prev_loss - loss) < eps:
			prev_loss_cur_iters += 1
		if prev_loss_cur_iters >= target_loss_target_iters:
			if debug:
				print("prev_loss reached")
				print("prev_loss, loss:", prev_loss, loss)
			break

		optimizer.step()

		# if we've reached our max norm, then normalize our parameters
		with torch.no_grad():
			if max_norm is not None and (cur_norm := torch.linalg.norm(vector)) > max_norm:
				vector[:] = max_norm * vector / torch.linalg.norm(vector)

			# normalize rows of left and right low rank matrices
			# according to the original MELBO post this works better than spectral norm
			if affine_rank is not None and max_affine_norm is not None:
				cur_affine_norms_left = matrix_left.norm(dim=1)
				affine_coeffs_left = torch.where(cur_affine_norms_left > max_affine_norm, max_affine_norm/cur_affine_norms_left, 1) 

				cur_affine_norms_right = matrix_right.norm(dim=1)
				affine_coeffs_right = torch.where(cur_affine_norms_right > max_affine_norm, max_affine_norm/cur_affine_norms_right, 1) 

				matrix_left[:] = torch.einsum('rm, r -> rm', matrix_left, affine_coeffs_left)
				matrix_right[:] = torch.einsum('rm, r -> rm', matrix_right, affine_coeffs_right)
		if return_loss_history: loss_history.append(loss)
		if return_vec_history: vec_history.append([x.detach().clone().cpu().float().numpy() for x in params])
		iters += 1

	if debug:
		print("Final loss:", loss)
		print("Number of iters:", iters)
		if prev_loss is not None: print("Difference between current loss and previous iter's loss:", abs(prev_loss - loss))

	retdict = {}
	retdict['iters'] = iters
	retdict['loss'] = loss if do_target_loss_sum else (all_completion_losses if not return_loss_history else loss_history)
	if return_vec_history: retdict['vec_history'] = vec_history
	retdict['norm'] = vector.norm().item()

	if not do_output_constr:
		retvals = (vector,)
		if affine_rank is not None:
			retvals = retvals + (matrix_left.T @ matrix_right,)
		if return_info:
			retvals = retvals + (retdict,)
		return retvals
	
	### Output-Constrained Optimization ###
	# okay, now it's time to do output-constrained optimization
	old_loss = loss
	if target_loss is None: target_loss = _nested_list_max(all_completion_losses)

	# first, compute scaling factor
	with torch.no_grad():
		starting_norm = vector.norm().item()
		if matrix_left is not None and matrix_right is not None:
			# use frobenius norm for matrix
			# TODO: maybe change?
			starting_norm += ((matrix_left.T @ matrix_right)**2).sum().sqrt().item()
		scale_factor = starting_norm/(eps+target_loss)
	
	# now, make our constraints
	output_constraints = []
	def make_output_constraint_func(datapoint_idx, completion_idx, vector, matrix_left=None, matrix_right=None, is_src_completion=True, do_one_minus=True):
		def constraint():
			matrix = None
			if matrix_left is not None and matrix_right is not None:
				matrix = matrix_left.T @ matrix_right
			return get_completion_loss_with_noise(datapoint_idx, completion_idx, vector, matrix, is_src_completion=is_src_completion, do_one_minus=do_one_minus)
		return constraint 

	for datapoint_idx, datapoint in enumerate(datapoints):
		for src_completion_idx, src_completion in enumerate(datapoint.src_completions):
			output_constraint_func = make_output_constraint_func(datapoint_idx, src_completion_idx, vector, matrix_left, matrix_right, is_src_completion=True, do_one_minus=do_one_minus)
			output_constraints.append(
				mdmm.MaxConstraint(output_constraint_func, scale=scale_factor, max=min(target_loss, all_completion_losses[datapoint_idx][0][src_completion_idx]+eps))
			)
		for dst_completion_idx, dst_completion in enumerate(datapoint.dst_completions):
			output_constraint_func = make_output_constraint_func(datapoint_idx, dst_completion_idx, vector, matrix_left, matrix_right, is_src_completion=False)
			output_constraints.append(
				mdmm.MaxConstraint(output_constraint_func, scale=scale_factor, max=min(target_loss, all_completion_losses[datapoint_idx][1][dst_completion_idx]+eps))
			)
	
	# if we're using a custom loss function (i.e. not just optimizing the vector norm), then constrain our vector norm too
	# TODO: figure out how to do scale factors with custom loss functions
	if custom_output_constr_loss_func is not None:
		def norm_constraint_func():
			loss = torch.linalg.norm(vector)
			if matrix_left is not None and matrix_right is not None:
				loss += ((matrix_left.T @ matrix_right)**2).sum().sqrt()
			return loss
		output_constraints.append(mdmm.MaxConstraint(norm_constraint_func, scale=1, max=output_constr_norm_initial_scale*norm_constraint_func().item()))
	
	# if we're using a custom loss function, then here is where preliminary information can be computed to be used in the optimization loop
	custom_output_constr_dict = None
	if custom_output_constr_pre_loss_func is not None:
		custom_output_constr_dict = custom_output_constr_pre_loss_func(model, datapoints, layer, vector, matrix_left, matrix_right, only_hook_prompt=only_hook_prompt)

	# now, do the actual optimization
	mdmm_module = mdmm.MDMM(output_constraints)
	if output_constr_lr is None: output_constr_lr = lr
	optimizer = mdmm_module.make_optimizer(params, lr=output_constr_lr)

	loss = None
	prev_loss = None
	iters = 0
	while prev_loss is None or loss <= prev_loss:
		if max_output_constr_iters is not None and iters > max_output_constr_iters:
			if debug: print("Max output-constr iters reached.")	
			break
		prev_loss = loss#.item() if loss is not None else None
		prev_vec = vector.detach().clone()
		
		optimizer.zero_grad()

		if custom_output_constr_loss_func is not None and use_transformer_lens:
			# NOTE: currently, custom loss funcs are only supported with transformer_lens
			if custom_output_constr_dict is not None:
				loss = custom_output_constr_loss_func(model, datapoints, layer, vector, matrix_left, matrix_right, only_hook_prompt=only_hook_prompt, **custom_output_constr_dict)
			else:
				loss = custom_output_constr_loss_func(model, datapoints, layer, vector, matrix_left, matrix_right, only_hook_prompt=only_hook_prompt)
		else:
			# use default loss

			# NOTE: loss is currently vector norm + frobenius norm of matrix
			# maybe this should be changed?
			my_loss = torch.linalg.norm(vector)
			if matrix_left is not None and matrix_right is not None:
				my_loss += ((matrix_left.T @ matrix_right)**2).sum().sqrt()
			my_loss.backward()
			loss = my_loss.item()

		# backprop constraint gradients
		_mdmm_grad_accumulate_backward(mdmm_module)

		optimizer.step()
		
		if debug: print(loss, prev_loss, iters)
		iters += 1
	
	# finally, prepare our return value
	retvals = (prev_vec,)
	retdict['norm'] = prev_vec.norm().item()
	retdict['output_constr_iters'] = iters
	if affine_rank is not None:
		retvals = retvals + (matrix_left.T @ matrix_right,)
	if return_info:
		retvals = retvals + (retdict,)
	return retvals

def make_melbo_loss_funcs(target_layer):
	"""
	Make custom loss functions for performing MELBO (www.lesswrong.com/posts/ioPnHKFyy4Cw2Gr2x) in conjunction with output-constrained optimization. Only supports TransformerLens.

	Args:
		target_layer: the layer at which the distance between steered and unsteered activations will be used to compute the MELBO loss.

	Returns:
		melbo_pre_loss_func, melbo_loss_func: two functions which should be passed to optimize_vector() as custom_output_constr_pre_loss_func and custom_output_constr_loss_func respectively
	"""
	make_steering_hook = make_steering_hook_tflens
	def melbo_pre_loss_func(model, datapoints, layer, vector, matrix_left, matrix_right, only_hook_prompt=None):
		hook_point = f'blocks.{target_layer}.hook_resid_pre'
		retdict = {'target_layer_activs': []}
		for datapoint in datapoints:
			prompt = datapoint.prompt
			prompt_len = len(model.to_tokens(prompt).tolist()[0])

			src_completion_activs = []
			for src_completion in datapoint.src_completions:
				with torch.no_grad():
					_, cache = model.run_with_cache(prompt + src_completion, stop_at_layer=target_layer+1, names_filter=[hook_point])
					activs = cache[hook_point][0, prompt_len-1:]
				src_completion_activs.append(activs)

			dst_completion_activs = []
			for dst_completion in datapoint.dst_completions:
				with torch.no_grad():
					_, cache = model.run_with_cache(prompt + dst_completion, stop_at_layer=target_layer+1, names_filter=[hook_point])
					activs = cache[hook_point][0, prompt_len-1:]
				dst_completion_activs.append(activs)

			datapoint_activs = [src_completion_activs, dst_completion_activs]
			retdict['target_layer_activs'].append(datapoint_activs)
		return retdict

	hook_dict = {}
	def capture_hook(x, hook):
		hook_dict['activs'] = x
		return x

	def melbo_loss_func(model, datapoints, layer, vector, matrix_left, matrix_right, target_layer_activs=None, only_hook_prompt=None, only_calculate_loss=False):
		loss = 0
		hook_point = f'blocks.{target_layer}.hook_resid_pre'
		for datapoint_idx, datapoint in enumerate(datapoints):
			prompt = datapoint.prompt
			prompt_len = len(model.to_tokens(prompt).tolist()[0])

			matrix = matrix_left.T @ matrix_right if matrix_left is not None and matrix_right is not None else None 
			if only_hook_prompt:
				if vector_clamp is None: hook_fn = make_steering_hook(vector, matrix=matrix, token=slice(0,prompt_len))
				else: hook_fn = make_steering_hook(vector_clamp*vector, matrix=make_abl_mat(vector), token=slice(0,prompt_len))
			else:
				if vector_clamp is None: hook_fn = make_steering_hook(vector, matrix=matrix, token=datapoint.token)
				else: hook_fn = make_steering_hook(vector_clamp*vector, matrix=make_abl_mat(vector), token=datapoint.token)
			if isinstance(layer, list):
				hook_infos = [ (f'blocks.{cur_layer}.hook_resid_pre', hook_fn) for cur_layer in layer]
			else:
				hook_infos = [ (f'blocks.{layer}.hook_resid_pre', hook_fn) ]

			for completion_idx, src_completion in enumerate(datapoint.src_completions):
				with model.hooks(fwd_hooks=hook_infos + [(hook_point, capture_hook)]):
					model(prompt + src_completion, stop_at_layer=target_layer+1)
				activs = hook_dict['activs'][0, prompt_len-1:]
				original_activs = target_layer_activs[datapoint_idx][0][completion_idx]
				mean_distance = -((activs-original_activs).norm(dim=-1).mean())
				loss += mean_distance.item()
				if not only_calculate_loss:
					mean_distance.backward()
				
			dst_completion_activs = []
			for completion_idx, dst_completion in enumerate(datapoint.dst_completions):
				with model.hooks(fwd_hooks=hook_infos + [(hook_point, capture_hook)]):
					model(prompt + dst_completion, stop_at_layer=target_layer+1)
				activs = hook_dict['activs'][0, prompt_len-1:]
				original_activs = target_layer_activs[datapoint_idx][1][completion_idx]
				mean_distance = -((activs-original_activs).norm(dim=-1).mean())
				loss += mean_distance.item()
				if not only_calculate_loss:
					mean_distance.backward()

		return loss
	return melbo_pre_loss_func, melbo_loss_func

def optimize_vector_minibatch_hf(model, tokenizer, prompts, layer,
	src_completions=None, dst_completions=None,
	minibatch_size=5,
	eps=1e-6, lr=0.01, max_iters=None, coldness=0.7,
	target_loss=None, target_loss_target_iters=1, satisfice=False,
	starting_norm=1, max_norm=None,
	affine_rank=None, max_affine_norm=None,
	debug=False, return_info=True,
	vector_clamp=None
):
	"""
	An alternative version to optimize_vector() that uses minibatching to speed up optimization. More limited than optimize_vector(), but faster. Only supports HuggingFace.

	Args:
		model: the HuggingFace model to optimize for
		tokenizer: the associated tokenizer
		prompts: a list of prompts to optimize over
		src_completions: a list of completions to suppress. All src_completions will be optimized over for all prompts.
		dst_completions: a list of completions to promote. All dst_completions will be optimized over for all prompts.
		
		All other arguments are the same as in optimize_vector() (although note that there are many arguments to optimize_vector() that are not supported by optimize_vector_minibatch_hf()).

	Returns a tuple containing the following elements in order:
		vector: the steering vector which has been optimized
		matrix: if affine_norm is not None, then the affine steering matrix which has been optimized
		info: if return_info is True, then a dictionary containing the following items:
			iters: the number of optimization steps taken
			norm: the norm of the returned vector
			loss: the total loss at the end of optimization
	"""

	if src_completions is None: src_completions = []
	if dst_completions is None: dst_completions = []
	d_model = model.config.hidden_size
	get_tokens = lambda prompt: tokenizer(prompt).input_ids
	def get_hooked_logits(prompt, hook_infos):
		cur_tokens = tokenizer(prompt, return_tensors='pt', padding=True, padding_side='left').input_ids
		with hf_hooks_contextmanager(model, hook_infos):
			logits = model(cur_tokens, use_cache=False).logits
		return logits 
	make_steering_hook = make_steering_hook_hf

	with torch.no_grad():
		vector = torch.randn(d_model)
		vector = starting_norm * vector / vector.norm()
		vector = vector.cuda()
	vector.requires_grad_(True)

	def get_completion_minibatch_loss(prompts, completion, vector, matrix=None, is_src_completion=True):
		prompt_lens = []
		for prompt in prompts:
			prompt_lens.append(len(get_tokens(prompt)))

		if vector_clamp is None:
			hook_fn = make_steering_hook(vector, matrix=matrix)
		else:
			hook_fn = make_steering_hook(vector_clamp*vector, make_abl_mat(vector))

		if isinstance(layer, list):
			hook_infos = [ (cur_layer, hook_fn) for cur_layer in layer]
		else:
			hook_infos = [ (layer, hook_fn) ]
		
		cur_loss = 0

		all_tokens = tokenizer([prompt + completion for prompt in prompts], padding=True, padding_side='left', return_tensors='pt')
		with hf_hooks_contextmanager(model, hook_infos):
			logits = model(**all_tokens, use_cache=False).logits
		probs = torch.nn.functional.softmax(logits*coldness, dim=-1)

		max_loss = 0
		for prompt_idx in range(len(prompts)):
			prompt_len = prompt_lens[prompt_idx]
			cur_tokens = all_tokens.input_ids[prompt_idx]
			cur_prompt_probs = probs[prompt_idx]
			token_idx = prompt_len-1
			while token_idx < len(cur_tokens)-1 and (next_token := cur_tokens[token_idx+1]) != tokenizer.pad_token:
				target_prob = (1-cur_prompt_probs[token_idx, next_token]) if is_src_completion else cur_prompt_probs[token_idx, next_token]
				target_logprob = torch.log(target_prob + eps)
				#if debug: print(target_logprob)
				cur_loss -= target_logprob
				token_idx += 1
		return cur_loss

	
	optimizer = torch.optim.Adam([vector], lr=lr)

	loss = None
	prev_loss = None
	iters = 0
	target_loss_cur_iters = 0
	prev_loss_cur_iters = 0

	minibatch_start_idx = 0
	minibatch_end_idx = None
	minibatch_rollover_end_idx = None

	while True:
		if max_iters is not None and iters > max_iters:
			if debug: print("Max iters reached.")	
			break
		if target_loss is not None and loss is not None:
			if loss < target_loss:
				target_loss_cur_iters += 1
				if debug: print(f"Loss stopping threshold {target_loss} hit. Loss: {loss}. Cur num iters: {target_loss_cur_iters}")
			else:
				target_loss_cur_iters = 0

			if target_loss_cur_iters >= target_loss_target_iters:
				if debug: print(f"Loss stopping threshold {target_loss} hit. Breaking.")
				break

		optimizer.zero_grad()
		prev_loss = loss
		loss = 0

		# get minibatch indices, accounting for "rollover" (which happens when minibatch size does not divide dataset len)
		minibatch_start_idx = minibatch_rollover_end_idx if minibatch_rollover_end_idx is not None else minibatch_end_idx if minibatch_end_idx is not None else 0
		minibatch_end_idx = minibatch_start_idx + minibatch_size
		if minibatch_end_idx > len(prompts):
			minibatch_rollover_end_idx = minibatch_end_idx % len(prompts)
			minibatch_end_idx = len(prompts)
		else:
			minibatch_rollover_end_idx = None
		minibatch = prompts[minibatch_start_idx:minibatch_end_idx]
		if minibatch_rollover_end_idx is not None:
			minibatch += prompts[:minibatch_rollover_end_idx]

		for src_completion in src_completions:
			# I think that we have to do this every time to prevent "backwarding through graph a second time" errors
			if affine_rank is not None:
				matrix = matrix_left.T @ matrix_right
			else:
				matrix = None
			cur_loss = get_completion_minibatch_loss(minibatch, src_completion, vector, matrix, is_src_completion=True)
			loss += cur_loss.item()
			if satisfice: cur_loss = (cur_loss - target_loss)**2
			cur_loss.backward()

		for dst_completion in dst_completions:
			# I think that we have to do this every time to prevent "backwarding through graph a second time" errors
			if affine_rank is not None:
				matrix = matrix_left.T @ matrix_right
			else:
				matrix = None
			cur_loss = get_completion_minibatch_loss(minibatch, dst_completion, vector, matrix, is_src_completion=False)
			loss += cur_loss.item()
			if satisfice: cur_loss = (cur_loss - target_loss)**2
			cur_loss.backward()

		loss /= minibatch_size*(len(src_completions)+len(dst_completions))
		if debug: print(loss)
		if prev_loss is not None and abs(prev_loss - loss) < eps:
			prev_loss_cur_iters += 1
		if prev_loss_cur_iters >= target_loss_target_iters:
			if debug:
				print("prev_loss reached")
				print("prev_loss, loss:", prev_loss, loss)
			break

		optimizer.step()

		# if we've reached our max norm, then normalize our parameters
		with torch.no_grad():
			if max_norm is not None and (cur_norm := torch.linalg.norm(vector)) > max_norm:
				vector[:] = max_norm * vector / torch.linalg.norm(vector)

			# normalize rows of left and right low rank matrices
			# according to the original MELBO post this works better than spectral norm
			if affine_rank is not None and max_affine_norm is not None:
				cur_affine_norms_left = matrix_left.norm(dim=1)
				affine_coeffs_left = torch.where(cur_affine_norms_left > max_affine_norm, max_affine_norm/cur_affine_norms_left, 1) 

				cur_affine_norms_right = matrix_right.norm(dim=1)
				affine_coeffs_right = torch.where(cur_affine_norms_right > max_affine_norm, max_affine_norm/cur_affine_norms_right, 1) 

				matrix_left[:] = torch.einsum('rm, r -> rm', matrix_left, affine_coeffs_left)
				matrix_right[:] = torch.einsum('rm, r -> rm', matrix_right, affine_coeffs_right)
		
		iters += 1

	if debug:
		print("Final loss:", loss)
		print("Number of iters:", iters)
		if prev_loss is not None: print("Difference between current loss and previous iter's loss:", abs(prev_loss - loss))

	retdict = {}
	retdict['iters'] = iters
	retdict['loss'] = loss
	retdict['norm'] = vector.norm().item()

	retvals = (vector,)
	if affine_rank is not None:
		retvals = retvals + (matrix_left.T @ matrix_right,)
	if return_info:
		retvals = retvals + (retdict,)
	return retvals
