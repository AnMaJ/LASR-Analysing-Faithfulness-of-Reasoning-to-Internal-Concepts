# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import random
import os

random.seed(280669)

nonterminal_nodes = 10
probs = {'*':.4, '+':.4, '-':.2}
num_dialogs = 10000

num_splits = 10

dialog_dir = f"./arith.nd{num_dialogs}.nt{nonterminal_nodes}"
rnd_dialog_fn = f"{dialog_dir}/arith.nd{num_dialogs}.nt{nonterminal_nodes}.rnd.json"
split_dialog_prefix = f"{dialog_dir}/arith.nd{num_dialogs}.nt{nonterminal_nodes}."

max_operand = 9

EMPTY_PREFIX = ""
ARITHMETIC_EXPRESSION_EVAL_PREFIX_NO_COT = "Evaluate the arithmetic expression below. Please answer with the final result, nothing else."
ARITHMETIC_EXPRESSION_EVAL_PREFIX = "Evaluate the arithmetic expression below."

ACTIVE_PREFIX = EMPTY_PREFIX

def sample_operator(probs:dict):
    _acc = 0.0
    _rnd = random.random()
    for _op in probs.keys():
        if (_rnd >= _acc) and (_rnd < _acc + probs[_op]):
            _result = _op
        _acc = _acc + probs[_op]
    return _result

def generate_random_arithmetic_expression(num_ops: int):

    _result = "N"
    _nt_count = 1
    for i in range(num_ops):
        _nt_to_expand = random.randrange(_nt_count)
        _idx_to_expand = _result.find('N')
        for j in range(_nt_to_expand):
            _idx_to_expand += _result[_idx_to_expand + 1:].find('N') + 1
        _new_op = sample_operator(probs)
        if (_new_op == '+') or (_new_op == '*'):
            _expansion = f"( N {_new_op} N )"
            _nt_count += 1
        else: # _new_op = 'not'
            _expansion = f"( {_new_op} N )"
        _result = _result[:_idx_to_expand] + _expansion + _result[_idx_to_expand + 1:]
    _idx_to_expand = _result.find('N')
    while _idx_to_expand > -1:
        _expansion = str(random.randint(0,max_operand))
        _result = _result[:_idx_to_expand] + _expansion + _result[_idx_to_expand + 1:]
        _idx_to_expand = _result.find('N')
    return _result


def generate_random_arith_expr_dialog(num_ops: int, id: int, sys_prefix: str):
    _new_arithmetic_expression = generate_random_arithmetic_expression(num_ops)
    _new_system_turn = \
        {'role': 'system',
         'content': sys_prefix}
    _new_user_turn = \
        {'role': 'user',
         'content': _new_arithmetic_expression,
         'expression_id': id,
          'value': eval(_new_arithmetic_expression) }

    _new_dialog = [_new_system_turn, _new_user_turn]

    return _new_dialog

def arith_exp_list2dialogs(arith_exps: list[str], sys_prefix: str):
    dialogs = []
    for id, a_arith_exp in enumerate(arith_exps):
        new_system_turn = \
            {'role': 'system',
            'content': sys_prefix}
        new_user_turn = \
            {'role': 'user',
            'content': a_arith_exp,
            'expression_id': id,
            'value': eval(a_arith_exp) }
        dialogs += [[new_system_turn,new_user_turn]]
    return dialogs

def save_arith_exp_dialogs(dialogs: list,dialog_fn: str):
    with open(dialog_fn,"w") as outfile:
        json.dump(dialogs,outfile,indent=4)

    return dialogs

def generate_random_arith_expr_dialog_file(num_dialogs: int, num_ops: int, sys_prefix: str, dialog_fn: str):

    _all_dialogs = [generate_random_arith_expr_dialog(num_ops, i, sys_prefix) \
                    for i in range(num_dialogs)]
    with open(dialog_fn,"w") as outfile:
        json.dump(_all_dialogs,outfile,indent=4)

    return _all_dialogs

if not os.path.isdir(dialog_dir):
    os.makedirs(dialog_dir)

rnd_dialogs = generate_random_arith_expr_dialog_file(num_dialogs, nonterminal_nodes,ACTIVE_PREFIX, rnd_dialog_fn)

