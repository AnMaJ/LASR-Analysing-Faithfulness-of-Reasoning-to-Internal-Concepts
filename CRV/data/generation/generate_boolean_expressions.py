# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import random
import os

random.seed(280669)

nonterminal_nodes = 10
probs = {'and':.4, 'or':.4, 'not':.2}
num_dialogs = 1000

num_splits = 10

dialog_dir = f"./bool.nd{num_dialogs}.nt{nonterminal_nodes}"
rnd_dialog_fn = f"{dialog_dir}/bool.nd{num_dialogs}.nt{nonterminal_nodes}.rnd.json"
split_dialog_prefix = f"{dialog_dir}/bool.nd{num_dialogs}.nt{nonterminal_nodes}."

EMPTY_PREFIX = ""
BOOLEAN_EXPRESSION_EVAL_PREFIX_NO_COT = "Evaluate the boolean expression below. Please answer with either False or True, nothing else."
BOOLEAN_EXPRESSION_EVAL_PREFIX = "Evaluate the boolean expression below."

ACTIVE_PREFIX = EMPTY_PREFIX

def generate_all_boolean_expressions(num_ops: int):
    bool_exp = [None] * num_ops
    if num_ops == 0:
        result = ['True', 'False']
    else:
        for j in range(num_ops):
            bool_exp[j] = generate_all_boolean_expressions(j)
        not_bool_exps = [f"( not {a_bool_exp} )" for a_bool_exp in bool_exp[num_ops-1]]
        and_bool_exps = []
        or_bool_exps = []
        for j in range(num_ops):
            and_bool_exps += [f"( {bool_exp_1} and {bool_exp_2} )" for bool_exp_1 in bool_exp[j] for bool_exp_2 in bool_exp[num_ops - j - 1]]
            or_bool_exps += [f"( {bool_exp_1} or {bool_exp_2} )" for bool_exp_1 in bool_exp[j] for bool_exp_2 in bool_exp[num_ops - j - 1]]
        result = not_bool_exps + and_bool_exps + or_bool_exps
    return result

def sample_operator(probs:dict):
    _acc = 0.0
    _rnd = random.random()
    for _op in probs.keys():
        if (_rnd >= _acc) and (_rnd < _acc + probs[_op]):
            _result = _op
        _acc = _acc + probs[_op]
    return _result

def generate_random_boolean_expression(num_ops: int):

    _result = "N"
    _nt_count = 1
    for i in range(num_ops):
        _nt_to_expand = random.randrange(_nt_count)
        _idx_to_expand = _result.find('N')
        for j in range(_nt_to_expand):
            _idx_to_expand += _result[_idx_to_expand + 1:].find('N') + 1
        _new_op = sample_operator(probs)
        if (_new_op == 'or') or (_new_op == 'and'):
            _expansion = f"( N {_new_op} N )"
            _nt_count += 1
        else: # _new_op = 'not'
            _expansion = f"( {_new_op} N )"
        _result = _result[:_idx_to_expand] + _expansion + _result[_idx_to_expand + 1:]
    _idx_to_expand = _result.find('N')
    while _idx_to_expand > -1:
        _expansion = 'False' if random.getrandbits(1) == 0 else 'True'
        _result = _result[:_idx_to_expand] + _expansion + _result[_idx_to_expand + 1:]
        _idx_to_expand = _result.find('N')
    return _result


def generate_random_bool_expr_dialog(num_ops: int, id: int, sys_prefix: str):
    _new_boolean_expression = generate_random_boolean_expression(num_ops)
    _new_system_turn = \
        {'role': 'system',
         'content': sys_prefix}
    _new_user_turn = \
        {'role': 'user',
         'content': _new_boolean_expression,
         'expression_id': id,
          'truth_value': eval(_new_boolean_expression) }

    _new_dialog = [_new_system_turn, _new_user_turn]

    return _new_dialog

def bool_exp_list2dialogs(bool_exps: list[str], sys_prefix: str):
    dialogs = []
    for id, a_bool_exp in enumerate(bool_exps):
        new_system_turn = \
            {'role': 'system',
            'content': sys_prefix}
        new_user_turn = \
            {'role': 'user',
            'content': a_bool_exp,
            'expression_id': id,
            'truth_value': eval(a_bool_exp) }
        dialogs += [[new_system_turn,new_user_turn]]
    return dialogs

def save_bool_exp_dialogs(dialogs: list,dialog_fn: str):
    with open(dialog_fn,"w") as outfile:
        json.dump(dialogs,outfile,indent=4)

    return dialogs

def generate_random_bool_expr_dialog_file(num_dialogs: int, num_ops: int, sys_prefix: str, dialog_fn: str):

    _all_dialogs = [generate_random_bool_expr_dialog(num_ops, i, sys_prefix) \
                    for i in range(num_dialogs)]
    with open(dialog_fn,"w") as outfile:
        json.dump(_all_dialogs,outfile,indent=4)

    return _all_dialogs

def generate_random_bool_expr_dialog_file1(num_dialogs: int, num_ops: int, sys_prefix: str):

    _all_dialogs = [generate_random_bool_expr_dialog(num_ops, i, sys_prefix) \
                    for i in range(num_dialogs)]

    return _all_dialogs

if not os.path.isdir(dialog_dir):
    os.makedirs(dialog_dir)

rnd_dialogs = generate_random_bool_expr_dialog_file(num_dialogs, nonterminal_nodes,ACTIVE_PREFIX, rnd_dialog_fn)


# when the nonterminal nodes are small, we can generate all possible boolean expressions
all_bool_exps = generate_all_boolean_expressions(nonterminal_nodes)

# # when the nonterminal nodes are large, which we can't possibly generall all possible expressions, e.g., for nt10, then we use the random generation
# rnd_dialogs1 = generate_random_bool_expr_dialog_file1(num_dialogs*num_splits, nonterminal_nodes,ACTIVE_PREFIX)
# all_bool_exps = [dialog[1]['content'] for dialog in rnd_dialogs1]

num_bool_exps = len(all_bool_exps)
split_size = num_dialogs if num_dialogs * num_splits <= num_bool_exps else num_bool_exps // num_splits
random.shuffle(all_bool_exps)

for split_id in range(num_splits):
    dialog_split = bool_exp_list2dialogs(all_bool_exps[split_id * split_size : (split_id + 1) * split_size],ACTIVE_PREFIX)
    save_bool_exp_dialogs(dialog_split, f"{split_dialog_prefix}{split_id}.json")
