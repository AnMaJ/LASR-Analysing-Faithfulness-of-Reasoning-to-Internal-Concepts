# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModel
import datasets as ds
from huggingface_hub import login
import os
from tqdm.auto import tqdm
import json
import torch
import fire
import random


access_token = "please enter your HF access token here"

BOOLEAN_EXPRESSION_EVAL_PREFIX = "Evaluate the boolean expression below."

ACTIVE_PREFIX = BOOLEAN_EXPRESSION_EVAL_PREFIX
dataset_dir = '/path/to/CRV/data'

model = None
tokenizer = None


def load_model(model_dir: str):
    model = AutoModelForCausalLM.from_pretrained(model_dir, token=access_token, device_map="cuda")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, token=access_token, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer

def remove_all_hooks(model):
    """Remove all forward/backward hooks from the model."""
    for module in model.modules():
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()
        module._backward_hooks.clear()


class ActivationCache:
    """Cache to store activations from a HF model."""

    def __init__(self, bsz: int, n_tokens: int, n_layers: int, n_heads: int, dim: int, head_dim: int):

        self._bsz = bsz
        self._n_tokens = n_tokens
        self._n_layers = n_layers
        self._n_heads = n_heads
        self._dim = dim
        self._head_dim = head_dim

        self._rs0_cache = None
        self._rs_cache = [None] * self._n_layers
        self._rsm_cache = [None] * self._n_layers
        self._mlp_cache = [None] * self._n_layers
        self._awv_cache = [None] * self._n_layers


def register_activation_cache_hooks(model, cache: ActivationCache, layer_idx=None):

    # RS cache hook definition and registration
    def make_rs0_cache_hook(cache):
        def rs0_cache_hook(module, inputs, outputs):
            if isinstance(outputs,tuple):
                outputs_0 = outputs[0]
            else:
                outputs_0 = outputs
            if outputs_0.shape[1] > 1:
                cache._rs0_cache = outputs_0
            else:
                cache._rs0_cache = torch.cat((cache._rs0_cache,outputs_0),dim=1)
            return outputs
        return rs0_cache_hook


    model.model.embed_tokens.register_forward_hook(
        make_rs0_cache_hook(cache)
    )

    if layer_idx is None:
        layer_idx = list(range(len(model.model.layers)))
    for l in layer_idx:
        device_idx_l = model.hf_device_map[f"model.layers.{l}"]

        # MLP cache hook definition and registration
        def make_mlp_cache_hook(cache,layer):
            def mlp_cache_hook(module, inputs, outputs):
                if isinstance(outputs,tuple):
                    outputs_0 = outputs[0]
                else:
                    outputs_0 = outputs
                if outputs_0.shape[1] > 1:
                    cache._mlp_cache[layer] = outputs_0
                else:
                    cache._mlp_cache[layer] = torch.cat((cache._mlp_cache[layer],outputs_0),dim=1)
                return outputs
            return mlp_cache_hook


        model.model.layers[l].mlp.register_forward_hook(
            make_mlp_cache_hook(cache, l)
        )


        # RS cache hook definition and registration
        def make_rs_cache_hook(cache,layer):
            def rs_cache_hook(module, inputs, outputs):
                if isinstance(outputs,tuple):
                    outputs_0 = outputs[0]
                else:
                    outputs_0 = outputs
                if outputs_0.shape[1] > 1:
                    cache._rs_cache[layer] = outputs_0
                else:
                    cache._rs_cache[layer] = torch.cat((cache._rs_cache[layer],outputs_0),dim=1)
                return outputs
            return rs_cache_hook


        model.model.layers[l].register_forward_hook(
            make_rs_cache_hook(cache, l)
        )

        # RSM cache hook definition and registration
        def make_rsm_cache_hook(cache,layer):
            def rsm_cache_hook(module, inputs):
                if isinstance(inputs,tuple):
                    inputs_0 = inputs[0]
                else:
                    inputs_0 = inputs
                if inputs_0.shape[1] > 1:
                    cache._rsm_cache[layer] = inputs_0
                else:
                    cache._rsm_cache[layer] = torch.cat((cache._rsm_cache[layer],inputs_0),dim=1)
                return inputs
            return rsm_cache_hook


        model.model.layers[l].post_attention_layernorm.register_forward_pre_hook(
            make_rsm_cache_hook(cache, l)
        )

        # Attention-Weighted Values cache hook definition and registration
        def make_awv_cache_hook(cache,layer):
            def awv_cache_hook(module, inputs):
                if isinstance(inputs,tuple):
                    inputs_0 = inputs[0]
                else:
                    inputs_0 = inputs
                if inputs_0.shape[1] > 1:
                    cache._awv_cache[layer] = inputs_0
                else:
                    cache._awv_cache[layer] = torch.cat((cache._awv_cache[layer],inputs_0),dim=1)
                return inputs
            return awv_cache_hook

        model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(
            make_awv_cache_hook(cache, l)
        )


def batch_eval_boolean_expressions(_dialogs: dict, \
                                   _model: AutoModelForCausalLM, \
                                    _tokenizer, \
                                    _outfile: str, \
                                    _export_file_prefix: str, \
                                    _batch_id: int,
                                    _max_bsz = 1, \
                                    _max_new_tokens : int = 4096,
                                    _temperature : int = 0.2,
                                    _save_activations = False,
                                    _save_ifr : bool = False,
                                    _renormalizing_threshold : float = None,
                                    _no_dialogs: bool = False):
    # _n_batches = (len(_dialogs) // _max_bsz) if len(_dialogs) % _max_bsz == 0 else (len(_dialogs) // _max_bsz + 1)
    _n_heads = _model.config.num_attention_heads
    _n_layers = _model.config.num_hidden_layers
    _dim = _model.config.hidden_size
    _head_dim = _model.config.head_dim

    # for _batch_id in range(_n_batches):
    _batch_export_file_prefix = f"{_export_file_prefix}/{_batch_id}"
    if not os.path.isdir(_batch_export_file_prefix):
        os.makedirs(_batch_export_file_prefix)
    _start_id = _max_bsz * _batch_id
    _end_id = min( _max_bsz * (_batch_id + 1), len(_dialogs))
    _bsz = _end_id - _start_id


    if _save_activations:
        _cache = ActivationCache(bsz = _bsz,
                                    n_tokens = _max_new_tokens,
                                    n_layers = _n_layers,
                                    n_heads = _n_heads,
                                    dim = _dim,
                                    head_dim = _head_dim)

        remove_all_hooks(_model)
        # register_activation_export_hooks(_model,f"{_batch_export_file_prefix}")
        register_activation_cache_hooks(_model,_cache)

    if _no_dialogs:
        _prologue = f"<|begin_of_text|><|start_header_id|>system {ACTIVE_PREFIX}<|end_header_id|>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        _epilogue = f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        _strings_not_dialogs = [f"{_prologue}{dialog[1]['content']}{_epilogue}" for dialog in _dialogs[_start_id : _end_id]]

        inputs = _tokenizer.batch_encode_plus(_strings_not_dialogs,
                                              padding=True,
                                              return_tensors="pt",
                                              return_attention_mask=True)

    else:
        inputs = _tokenizer.apply_chat_template(_dialogs[_start_id : _end_id],
                                                tokenize=True,
                                                padding=True,
                                                add_generation_prompt=True,
                                                return_tensors="pt",
                                                return_dict=True)
    inputs.to(_model.model.embed_tokens.weight.device)
    outputs = _model.generate(inputs['input_ids'].to(_model.model.embed_tokens.weight.device),
                            max_new_tokens = _max_new_tokens,
                            temperature = _temperature,
                            attention_mask=inputs['attention_mask'],
                            output_attentions=False,
                            return_dict_in_generate=True)

    # Immediately move inputs to CPU to free GPU memory
    inputs.to('cpu')

    # Position initialized with the last input token, where AR generation begins
    # tbg : token before generation
    important_token_positions = {}
    important_token_positions['tbg_position'] = inputs['input_ids'].shape[1] - 1

    # lgt : last generated token. fnpt : first non-padding token
    important_token_positions['lgt_positions'] = [None] * _bsz
    important_token_positions['fnpt_positions'] = [None] * _bsz
    for i in range(_bsz):
        _output_items = [x.item() for x in outputs['sequences'][i]]
        _lgt_position = len(_output_items) - 1
        while _output_items[_lgt_position] == _tokenizer.eos_token_id: _lgt_position -= 1
        important_token_positions['lgt_positions'][i] = _lgt_position
        _fnpt_position = 0
        while _output_items[_fnpt_position] == _tokenizer.pad_token_id: _fnpt_position += 1
        important_token_positions['fnpt_positions'][i] = _fnpt_position

    # _n_layers = 2 # For DEBUG only. Remove.

    if _save_ifr:
        attribution_graphs = [None] * _bsz
        for _sample_id in range(_bsz):
            # Initialize Information Flow Route graph for this sample
            _n_tokens = important_token_positions['lgt_positions'][_sample_id] + 1
            # _n_tokens = important_token_positions['lgt_positions'][_sample_id]
            attribution_graphs[_sample_id] = ttrg.GraphBuilder(_n_layers,_n_tokens,_n_heads)

        for _layer in tqdm(range(_n_layers)):

            _max_seqlen = outputs['attentions'][-1][_layer].shape[-1]

            all_batch_attentions = torch.zeros(_bsz,_n_heads,_max_seqlen,_max_seqlen)
            _pos_start = _kv_pos_start = 0
            _pos_end = _kv_pos_end = outputs['attentions'][0][_layer].shape[-1]
            for _attn_block in outputs['attentions']:
                # bsz, n_heads, pos_seqlen, keypos_seqlen
                all_batch_attentions[:,:,_pos_start:_pos_end,_kv_pos_start:_kv_pos_end] = _attn_block[_layer]
                _pos_start = _pos_end
                _pos_end += 1
                _kv_pos_end += 1

            _layer_v_cache = outputs['past_key_values'][_layer][1] # bsz, n_kv_heads, keypos_seqlen, head_dim
            _n_kv_heads = _layer_v_cache.shape[1]
            _kv_group_size = _n_heads // _n_kv_heads
            _expanded_v_cache = \
                torch.cat(
                    [_layer_v_cache[:,i,:,:].unsqueeze(1).expand(-1,_kv_group_size,-1,-1)\
                        for i in range(_n_kv_heads)],dim=1
                    )

            # bsz, n_heads, pos_seqlen, kv_seqlen, head_dim
            _layer_awv_by_s_token = all_batch_attentions[:,:,:,:,None].to(_expanded_v_cache.device) \
                * _expanded_v_cache[:,:,None,:,:]

            # n_heads, head_dim, dim
            _layer_wo = _model.model.layers[_layer].self_attn.o_proj.weight.detach().view(_n_heads,_head_dim,-1)

            _rsm = _cache._rsm_cache[_layer]
            _rs = _cache._rs_cache[_layer]
            _mlp = _cache._mlp_cache[_layer]

            _c_ffn, _c_resid_ffn = ttrc.get_mlp_contributions(_rsm, _rs, _mlp)

            for _sample_id in tqdm(range(_bsz)):

                _n_tokens = important_token_positions['lgt_positions'][_sample_id] + 1
                _fnpt_position = important_token_positions['fnpt_positions'][_sample_id]
                _layer_attn_o = torch.zeros(_n_heads,_n_tokens,_dim).to(_rsm.device)

                for _head in range(_n_heads):
                    _layer_head_attn_o_by_s_token = \
                        (_layer_awv_by_s_token[_sample_id,_head,_fnpt_position:_n_tokens,_fnpt_position:_n_tokens] @ _layer_wo[_head])\
                            .transpose(0,1) # kv_seqlen, pos_seqlen, dim

                    # Aggregate by target token position
                    _layer_head_attn_o = _layer_head_attn_o_by_s_token.sum(0) # pos_seqlen, dim

                    # Compute contributions from kv_token to token by head
                    _c_tok_to_attn_head = ttrc.get_contributions( # kv_seqlen, pos_seqlen
                        parts = _layer_head_attn_o_by_s_token, # kv_seqlen, pos_seqlen, dim
                        whole = _layer_head_attn_o, # pos_seqlen, dim
                        distance_norm = 1
                    )

                    # Remove influence from the future
                    _c_tok_to_attn_head = _c_tok_to_attn_head.triu()

                    # Add threshold and normalize
                    if _renormalizing_threshold is not None:
                        _dummy_c_res_to_attn_head = torch.zeros(_c_tok_to_attn_head.shape[1]).to(_c_tok_to_attn_head.device)
                        _c_tok_to_attn_head = ttrc.apply_threshold_and_renormalize( # kv_seqlen, pos_seqlen
                                                    # Adjusting the threshold to take into account the number of heads
                                                    # and the length of the sequence.
                                                    _renormalizing_threshold / (_n_heads),
                                                    _c_tok_to_attn_head.transpose(0,1),    # pos_seqlen, kv_seqlen
                                                    _dummy_c_res_to_attn_head # pos_seqlen
                                                    )[0].transpose(0,1)

                    # Start from the first non padding token at position _fnpt_position and end at lgt_position
                    for _token_from in range(_n_tokens - _fnpt_position):
                        for _token_to in range(_n_tokens - _fnpt_position):
                            _w = _c_tok_to_attn_head[_token_from,_token_to].item()
                            if _w > _renormalizing_threshold / _n_heads:
                                attribution_graphs[_sample_id]\
                                    .add_residual_to_attn_head(
                                        _layer,
                                        _head,
                                        _token_from,
                                        _token_to,
                                        _w
                                    )
                    _layer_attn_o[_head][_fnpt_position:_n_tokens] = _layer_head_attn_o # pos_seqlen, dim

                # _c_attn_head: n_heads, seqlen  , _c_resid_attn: seqlen
                _c_attn_head, _c_resid_attn = ttrc.get_contributions_with_one_off_part(
                    _layer_attn_o[:,_fnpt_position:_n_tokens], # n_heads, seqlen, dim
                    _rs[_sample_id,_fnpt_position:_n_tokens] if _layer > 0 \
                        else _cache._rs0_cache[_sample_id,_fnpt_position:_n_tokens], # seqlen, dim
                    _rsm[_sample_id,_fnpt_position:_n_tokens], # seqlen, dim
                    distance_norm = 1
                )

                # _c_attn_head: n_heads, seqlen  , _c_resid_attn: seqlen
                _c_attn_head_T, _c_resid_attn = ttrc.apply_threshold_and_renormalize(
                    _renormalizing_threshold / _n_heads,
                    _c_attn_head.transpose(0,1), # n_heads, seqlen
                    _c_resid_attn # seqlen
                )
                _c_attn_head = _c_attn_head_T.T

                for _token in range(_n_tokens - _fnpt_position):
                    for _head in range(_n_heads):
                        _w = _c_attn_head[_head,_token].item()
                        if _w > _renormalizing_threshold / _n_heads:
                            attribution_graphs[_sample_id]\
                                .add_attn_head_to_attn(_layer,_head, _token, _w)

                    _w = _c_ffn[_sample_id,_token].item()
                    if _w > _renormalizing_threshold:
                        attribution_graphs[_sample_id]\
                            .add_ffn_edge(_layer,_token,_w)

                    _w = _c_resid_ffn[_sample_id,_token].item()
                    if _w > _renormalizing_threshold:
                        attribution_graphs[_sample_id]\
                            .add_residual_to_ffn(_layer,_token,_w)

                    _w = _c_resid_attn[_token].item()
                    if _w > _renormalizing_threshold:
                        attribution_graphs[_sample_id]\
                            .add_residual_to_attn(_layer, _token, _w)


        for _sample_id in range(_bsz):
            torch.save(attribution_graphs[_sample_id],f"{_batch_export_file_prefix}/ifr_graph.{_sample_id:02d}")


    answers = [_tokenizer.decode(_output[important_token_positions['tbg_position'] + 1 : \
                                            important_token_positions['lgt_positions'][i] + 1]) \
                                            for (i, _output) in enumerate(outputs['sequences'])]

    # List of positions initialized with the last generated tokens in each sample.

    for i in range(len(answers)):
        if _no_dialogs:
            _dialogs[_max_bsz * _batch_id + i][0]['content'] = ACTIVE_PREFIX
        _dialogs[_max_bsz * _batch_id + i].append({'role':'assistant','content':answers[i]})
    torch.save(_dialogs[_start_id : _end_id],f"{_batch_export_file_prefix}/answered_dialogs.pt")

    torch.save(important_token_positions,f"{_batch_export_file_prefix}/important_token_positions")
    torch.save(outputs['sequences'],f"{_batch_export_file_prefix}/model.generate_outputs")

    # Saving activation cache content
    if _save_activations:
        torch.save(_cache._rs0_cache,f"{_batch_export_file_prefix}/rs0_dump")
        for _layer in range(_model.config.num_hidden_layers):
            torch.save(_cache._awv_cache[_layer],f"{_batch_export_file_prefix}/awv_dump.{_layer:02d}")
            torch.save(_cache._mlp_cache[_layer],f"{_batch_export_file_prefix}/mlp_dump.{_layer:02d}")
            torch.save(_cache._rsm_cache[_layer],f"{_batch_export_file_prefix}/rsm_dump.{_layer:02d}")
            torch.save(_cache._rs_cache[_layer],f"{_batch_export_file_prefix}/rs_dump.{_layer:02d}")


    return _dialogs


def eval_boolean_expressions(_dialogs: dict, _model, _tokenizer, _outfile: str):

    if os.path.isfile(_outfile):
        _dialogs = torch.load(_outfile)
    else:
        for _dialog in tqdm(_dialogs):
            inputs = _tokenizer.apply_chat_template(_dialog,tokenize=True,add_generation_prompt=True,return_tensors="pt",return_dict=True)
            outputs = _model.generate(inputs['input_ids'], max_new_tokens = 256,\
                attention_mask=inputs['attention_mask'])
            _answer = _tokenizer.decode(outputs[0])
            _answer = _answer = _answer[_answer.rfind("<|end_header_id|>") + len("<|end_header_id|>") + 2:_answer.rfind("<|eot_id|>")]
            _dialog.append({'role':'assistant','content':_answer})
            print(_answer)

        torch.save(_dialogs,_outfile)

    return _dialogs

def score_boolean_evaluations(dialogs: list, soft:bool=False, margin:int=10):

    def is_hard_valid_answer(answer:str):
        return answer == 'True' or answer =='False'

    def is_soft_valid_answer(answer:str, margin:int = 10):
        last_T_occurrence = answer.rfind('True')
        last_F_occurrence = answer.rfind('False')

        return (last_T_occurrence > -1 and last_T_occurrence > len(answer) - margin)\
              or (last_F_occurrence > -1 and last_F_occurrence > len(answer) - margin)

    def is_hard_correct_answer(answer:str, truth_label:bool):
        return (answer=='True' and truth_label) or (answer=='False' and not truth_label)

    def is_soft_correct_answer(answer:str, truth_label:bool):
        last_T_occurrence = answer.rfind('True')
        last_F_occurrence = answer.rfind('False')

        # Note: last_T_occurrence > last_F_occurrence implies last_T_occurrence > -1
        return (last_T_occurrence > last_F_occurrence and truth_label) or \
            (last_F_occurrence > last_T_occurrence and not truth_label)

    if soft:
        num_correct = len([1 for _dialog in dialogs if \
                           (is_soft_valid_answer(_dialog[2]['content'],margin) and \
                            is_soft_correct_answer(_dialog[2]['content'], _dialog[1]['truth_value']))])
        num_invalid = len([1 for _dialog in dialogs if not is_soft_valid_answer(_dialog[2]['content'],margin)])
    else:
        num_correct = len([1 for _dialog in dialogs if is_hard_correct_answer(_dialog[2]['content'], _dialog[1]['truth_value'])])
        num_invalid = len([1 for _dialog in dialogs if not is_hard_valid_answer(_dialog[2]['content'])])
    num_total = len(dialogs)
    accuracy = num_correct / (num_total - num_invalid)
    print(f"Invalid answers: {num_invalid} ({num_invalid / num_total})")
    print(f"Accuracy: {accuracy}")
    return accuracy


def main(
        model_name: str,
        dataset_dir: str,
        answer_prefix: str,
        job_id: int,
        random_seed: int = 2806,
        temperature: int = 0.1,
        no_dialogs: bool = False,
        save_activations: bool = False,
        save_ifr: bool = False
):
    model = None
    tokenizer = None
    # batch_size = 20
    # batch_size = 2
    random.seed(random_seed)


    if dataset_dir.split(".")[-1] == "nt3":
        nt, nd, batch_size = "nt3", "nd100", 10
    elif dataset_dir.split(".")[-1] == "nt5":
        nt, nd, batch_size = "nt5", "nd1000", 100
    elif dataset_dir.split(".")[-1] == "nt7":
        nt, nd, batch_size = "nt7", "nd1000", 25
    elif dataset_dir.split(".")[-1] == "nt10":
        nt, nd, batch_size = "nt10", "nd1000", 25

    dataset_names = [f'bool.{nd}.{nt}.{i}' for i in range(10)]

    dataset_files = [f'{dataset_dir}/{_dn}.json' for _dn in dataset_names]

    answer_dir = f"{answer_prefix}/{model_name}/boolean_expressions/bool.{nd}.{nt}"
    answer_files = [f'{answer_dir}/{_dn}/{_dn}.pt' for _dn in dataset_names]

    export_file_prefixes = [f'{answer_dir}/{_dn}' for _dn in dataset_names]

    for _efp in export_file_prefixes:
        if not os.path.isdir(_efp):
            os.makedirs(_efp, exist_ok=True)

    slurm_arr_args = []
    for _ds_idx, _dataset_file in enumerate(dataset_files):
        with open(_dataset_file) as input_file:
            input_items = json.load(input_file)
        _n_batches = (len(input_items) // batch_size) if len(input_items) % batch_size == 0 else (len(input_items) // batch_size + 1)
        slurm_arr_args = slurm_arr_args + [(_ds_idx,_bid) for _bid in list(range(_n_batches))]


    _ds_idx, _batch_id = slurm_arr_args[job_id]
    _dataset_file = dataset_files[_ds_idx]

    accuracy_dict = {}
    (model, tokenizer) = load_model(model_name)
    accuracy_dict[model_name] = {}
    with open(_dataset_file) as input_file:
        input_items = json.load(input_file)
    dialogs_answered = batch_eval_boolean_expressions(input_items,
                                                        model,
                                                        tokenizer,
                                                        answer_files[_ds_idx],
                                                        export_file_prefixes[_ds_idx],
                                                        _batch_id = _batch_id,
                                                        _max_bsz = batch_size,
                                                        _temperature =  temperature,
                                                        _save_activations = save_activations,
                                                        _save_ifr = save_ifr,
                                                        _renormalizing_threshold = 1.0e-3,
                                                        _no_dialogs = no_dialogs)


""" Usage example
python eval_bool_expressions.py \
--model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
--dataset_dir CRV/data/ \
--answer_prefix /llama_dumps \
--job_id $SLURM_ARRAY_TASK_ID \
--random_seed 2806 \
--temperature 0.1 \
--no_dialogs \
--save_activations \
--save_ifr


"""


if __name__ == "__main__":
    fire.Fire(main)
