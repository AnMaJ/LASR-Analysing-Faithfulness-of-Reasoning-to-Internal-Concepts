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

# model_name = 'meta-llama/Llama-3.1-8B-Instruct'


ARITHMETIC_EXPRESSION_EVAL_PREFIX = "Evaluate this arithmetic expression."
ACTIVE_PREFIX = ARITHMETIC_EXPRESSION_EVAL_PREFIX

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
        # device_idx_l = model.hf_device_map[f"model.layers.{l}"]

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


def batch_eval_arithmetic_expressions(_dialogs: dict, \
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
    answers = [_tokenizer.decode(_output[important_token_positions['tbg_position'] + 1 : \
        important_token_positions['lgt_positions'][i] + 1]) \
            for (i, _output) in enumerate(outputs['sequences'])]

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


def score_arithmetic_evaluations(dialogs: list, soft:bool=False, margin:int=10):

    def is_hard_valid_answer(answer:str):
        return answer.isnumeric()

    def is_soft_valid_answer(answer:str, margin:int = 10):

        all_numbers = re.findall(r'-?\d+',answer)
        if len(all_numbers) == 0: return False
        last_number = all_numbers[-1]
        last_number_start_index = answer.rfind(last_number)
        return last_number_start_index > len(answer) - margin


    def is_hard_correct_answer(answer:str, value_label:int):
        return (answer.isnumeric() and int(answer) == value_label)

    def is_soft_correct_answer(answer:str, value_label:int):

        all_numbers = re.findall(r'-?\d+',answer)
        if len(all_numbers) == 0: return False
        last_number = all_numbers[-1]
        return int(last_number) == value_label

    if soft:
        correct_answers = [_dialog[1]['expression_id'] for _dialog in dialogs if \
                           (is_soft_valid_answer(_dialog[2]['content'],margin) and \
                            is_soft_correct_answer(_dialog[2]['content'], _dialog[1]['value']))]
        invalid_answers = [_dialog[1]['expression_id'] for _dialog in dialogs if not is_soft_valid_answer(_dialog[2]['content'],margin)]
    else:
        correct_answers = [_dialog[1]['expression_id'] for _dialog in dialogs if is_hard_correct_answer(_dialog[2]['content'], _dialog[1]['value'])]
        invalid_answers = [_dialog[1]['expression_id'] for _dialog in dialogs if not is_hard_valid_answer(_dialog[2]['content'])]

    num_correct = len(correct_answers)
    num_invalid = len(invalid_answers)
    num_total = len(dialogs)
    accuracy = num_correct / (num_total - num_invalid)
    print(f"Invalid answers: {num_invalid} ({num_invalid / num_total})")
    print(f"Accuracy: {accuracy}")
    return accuracy, correct_answers, invalid_answers

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

    nt = dataset_dir.split(".")[-1]
    nd = dataset_dir.split(".")[-2]
    batch_size = 25
    # num_dialogs = int(nd[2:])
    # num_batches = num_dialogs // batch_size if num_batches % batch_size == 0 else num_dialogs // batch_size + 1


    dataset_names = [f'arith.{nd}.{nt}']
    dataset_files = [f'{dataset_dir}/{_dn}.rnd.json' for _dn in dataset_names]

    answer_dir = f"{answer_prefix}/{model_name}/arithmetic_expressions"
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

    dialogs_answered = batch_eval_arithmetic_expressions(input_items,
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
python eval_arith_expressions.py \
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
