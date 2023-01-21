#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
import torch
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset, load_metric
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    M2M100Tokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from external.regexDFAEquals import dfa_eual_test
from torch import nn
from transformers import T5Tokenizer

from easy_eval.eval import check_equiv, check_io_consistency

from easy_eval.streg_utils import parse_spec_to_ast

ax = 0
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.19.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

logger = logging.getLogger(__name__)

# A list of all multilingual tokenizer which require src_lang and tgt_lang attributes.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer]

setMulti = {
    'and',
    'or',
    'concat',
}

argument_Dict = {
    'argument0': '<m0>',
    'argument1': '<m1>',
    'argument2': '<m2>',
    'argument3': '<m3>',
    'argument4': '<cap>',
    'argument5': '<num>',
    'argument6': '<low>',
    'argument7': '<any>',
    'argument8': '<let>',
    'argument9': '<vow>',
}

opDict = {
    'optional': 1,
    'endwith': 1,
    'startwith': 1,
    'not': 1,
    'repeatatleast': 2,
    'contain': 1,
    'repeat': 2,
    'star': 1,
    'repeatrange': 3,
}
bina_opDict = {
    'optional': 1,
    'endwith': 1,
    'startwith': 1,
    'not': 1,
    'repeatatleast': 2,
    'contain': 1,
    'repeat': 2,
    'star': 1,
    'repeatrange': 3,
    'and': 2,
    'or': 2,
    'concat': 2,
}


def getArg(desc):
    if desc == 'capital':
        return 'cap>'
    if desc == 'number':
        return 'num>'
    if desc == 'lower-case':
        return 'low>'
    if desc == 'any':
        return 'any>'
    if desc == 'letter':
        return 'let>'
    if desc == 'vowel':
        return 'vow>'
    return desc


def parse_Scratachpad7(item):
    if len(item.split(': ')) != 2:
        return item
    left = item.split(': ')[0]
    item = item.split(': ')[1]
    item = item.replace(',', '')
    if 'optionally' in item:
        item = item.split()
        if len(item) < 2:
            print(item)
            return ""
        return left + '=optional(' + getArg(item[1]) + ')'
    elif 'end with' in item:
        item = item.split()
        if len(item) < 3:
            print(item)
            return ""
        return left + '=endwith(' + getArg(item[2]) + ')'
    elif 'start with' in item:
        item = item.split()
        if len(item) < 3:
            print(item)
            return ""
        return left + '=startwith(' + getArg(item[2]) + ')'
    elif 'exclude' in item:
        item = item.split()
        if len(item) < 2:
            print(item)
            return ""
        return left + '=not(' + getArg(item[1]) + ')'
    elif 'include' in item:
        item = item.split()
        if len(item) < 2:
            print(item)
            return ""
        return left + '=contain(' + getArg(item[1]) + ')'
    elif 'zero or more times' in item:
        item = item.split()
        if len(item) < 2:
            print(item)
            return ""
        return left + '=star(' + getArg(item[1]) + ')'
    elif 'or more times' in item:
        item = item.split()
        idx = -1
        for k in range(len(item)):
            if item[k] == 'or':
                idx = k
        if idx == -1:
            print(item)
            return ' '
        return left + '=repeatatleast(' + getArg(item[1]) + ',' + item[idx - 1] + ')'
    elif 'exactly' in item:
        item = item.split()
        idx = -1
        for k in range(len(item)):
            if item[k] == 'exactly':
                idx = k
        if idx == -1:
            print(item)
            return ' '
        if len(item) <= idx+1:
            print(item)
            return ' '
        return left + '=repeat(' + getArg(item[1]) + ',' + item[idx + 1] + ')'
    elif 'to' in item and 'repeat' in item:
        item = item.split()
        idx = -1
        for k in range(len(item)):
            if item[k] == 'to':
                idx = k
        if idx == -1:
            return ' '
        if len(item) <= idx+1:
            print(item)
            return ' '
        return left + '=repeatrange(' + getArg(item[1]) + ',' + item[idx - 1] + ',' + item[idx + 1] + ')'
    elif 'satisfy' in item and 'and' in item:
        item = item.split()
        tmp = left + '=and(' + getArg(item[1])
        for i in range(len(item)):
            if item[i] == 'and' and i < len(item) - 1:
                tmp = tmp + ',' + getArg(item[i + 1])
        tmp = tmp + ')'
        return tmp
    elif 'satisfy' in item:
        item = item.split()
        tmp = left + '=or(' + getArg(item[1])
        for i in range(len(item)):
            if item[i] == 'or' and i < len(item) - 1:
                tmp = tmp + ',' + getArg(item[i + 1])
        tmp = tmp + ')'
        return tmp
    elif 'connect' in item:
        item = item.split()
        tmp = left + '=concat(' + getArg(item[1])
        for i in range(len(item)):
            if item[i] == 'and' and i < len(item) - 1:
                tmp = tmp + ',' + getArg(item[i + 1])
        tmp = tmp + ')'
        return tmp
    else:
        print(item)
        return " "


def parse_Scratachpad(item):
    if len(item.split(': ')) != 2:
        return item
    left = item.split(': ')[0]
    item = item.split(': ')[1]
    item = item.replace(',', '')
    if 'optional' in item:
        item = item.split()
        if len(item) < 2:
            print(item)
            return ""
        return left + '=optional(' + getArg(item[1]) + ')'
    elif 'end with' in item:
        item = item.split()
        if len(item) < 5:
            print(item)
            return ""
        return left + '=endwith(' + getArg(item[4]) + ')'
    elif 'start with' in item:
        item = item.split()
        if len(item) < 5:
            print(item)
            return ""
        return left + '=startwith(' + getArg(item[4]) + ')'
    elif 'not appear' in item:
        item = item.split()
        if len(item) < 2:
            print(item)
            return ""
        return left + '=not(' + getArg(item[1]) + ')'
    elif 'appear zero or more times' in item:
        item = item.split()
        if len(item) < 2:
            print(item)
            return ""
        return left + '=star(' + getArg(item[1]) + ')'
    elif 'or more times' in item:
        item = item.split()
        if len(item) < 9:
            print(item)
            return ""
        return left + '=repeatatleast(' + getArg(item[8]) + ',' + item[3] + ')'
    elif 'should contain' in item:
        item = item.split()
        if len(item) < 5:
            print(item)
            return ""
        return left + '=contain(' + getArg(item[4]) + ')'
    elif 'contains exactly' in item:
        item = item.split()
        if len(item) < 8:
            print(item)
            return ""
        return left + '=repeat(' + getArg(item[7]) + ',' + item[4] + ')'
    elif 'need to be met' in item:
        item = item.split()
        tmp = left + '=and(' + getArg(item[0])
        for i in range(len(item)):
            if item[i] == 'and':
                tmp = tmp + ',' + getArg(item[i + 1])
        tmp = tmp + ')'
        return tmp
    elif 'to' in item:
        item = item.split()
        idx = -1
        for k in range(len(item)):
            if item[k] == 'appear':
                idx = k
        if idx == -1:
            return ' '
        if len(item) < 6:
            print(item)
            return " "
        return left + '=repeatrange(' + getArg(item[1]) + ',' + item[idx + 1] + ',' + item[idx + 3] + ')'
    elif 'are allowed' in item:
        item = item.split()
        tmp = left + '=or(' + getArg(item[0])
        for i in range(len(item)):
            if item[i] == 'or':
                tmp = tmp + ',' + getArg(item[i + 1])
        tmp = tmp + ')'
        return tmp
    elif 'next is' in item:
        item = item.split()
        tmp = left + '=concat(' + getArg(item[3])
        for i in range(len(item)):
            if item[i] == 'next':
                tmp = tmp + ',' + getArg(item[i + 2])
        tmp = tmp + ')'
        return tmp
    else:
        print(item)
        return " "


class RegexTree:

    def __init__(self, key):
        self.key = key
        self.child = []

    def insertChild(self, nbran):
        self.child.append(nbran)

    def preorderVisit(self):
        pre = []
        pre.append(self.key)
        # print(self.key)
        for i in range(len(self.child)):
            pre.extend(self.child[i].preorderVisit())
        if self.key == 'concat' or self.key == 'and' or self.key == 'or':
            pre.append('end')

        # if len(self.child) == 0:
        #    pre.append("<end>")
        return pre

    def bina_preorderVisit(self):
        pre = []
        pre.append(self.key)
        # print(self.key)

        for i in range(len(self.child)):
            pre.extend(self.child[i].bina_preorderVisit())
        # if len(self.child) == 0:
        #    pre.append("<end>")
        return pre

    def postorderVisit(self):
        post = []
        # print(self.key)
        if self.key == 'concat' or self.key == 'and' or self.key == 'or':
            post.append('begin')
        for i in range(len(self.child)):
            post.extend(self.child[i].postorderVisit())
        # if len(self.child) == 0:
        #    pre.append("<end>")
        post.append(self.key)
        return post

    def bina_postorderVisit(self):
        post = []
        # print(self.key)
        for i in range(len(self.child)):
            post.extend(self.child[i].postorderVisit())
        # if len(self.child) == 0:
        #    pre.append("<end>")
        post.append(self.key)
        return post

    def simplify(self):
        tmp = []
        for i in range(len(self.child)):
            self.child[i].simplify()
            if self.key != self.child[i].key:
                tmp.append(self.child[i])
            else:
                for j in range(len(self.child[i].child)):
                    tmp.append(self.child[i].child[j])
        self.child = tmp

    def binarify(self):
        tmp = self.child
        while setMulti.__contains__(self.key) and len(tmp) > 2:
            right = tmp.pop()
            left = tmp.pop()
            newNode = RegexTree(self.key)
            newNode.child.append(left)
            newNode.child.append(right)
            tmp.append(newNode)
        self.child = tmp
        for i in range(len(self.child)):
            self.child[i].binarify()

    '''
    def insertleft(self, nbran):
        if self.left == None:
            self.left = binarytree2(nbran)
        else:
            t = binarytree2(nbran)
            t.left = self.left
            self.left = t

    def insertright(self, nbran):  # 插入右节点
        if self.right == None:
            self.right = binarytree2(nbran)
        else:
            t = binarytree2(nbran)
            t.right = self.right
            self.right = t
    '''


def build_tree(regex):
    root = RegexTree(regex[0])
    if len(regex) == 1:
        return root
    if regex[1] != '(':
        print(regex)
        return root
    pos = 2
    open_num = 0  # 左括号+1,右括号-1
    for i in range(2, len(regex) - 1):
        if open_num < 0:
            print('error regex open or close')
            return root
        if regex[i] == '(':
            open_num = open_num + 1
            continue
        if regex[i] == ')':
            open_num = open_num - 1
            continue
        if regex[i] == ',' and open_num == 0:
            root.insertChild(build_tree(regex[pos:i]))
            pos = i + 1
    root.insertChild(build_tree(regex[pos:len(regex) - 1]))

    return root


def norm2(regex):
    res = ""
    flag = False
    for index in range(len(regex) - 1, -1, -1):
        if flag and (regex[index] == " " or regex[index] == "("):
            flag = False
            res = '<' + res
        res = regex[index] + res
        if regex[index] == '>':
            flag = True
    if flag == True:
        res = '<' + res
    res = ''.join(res.split())
    return res


def norm3(regex):
    regex = regex.replace('(', ' ( ')
    regex = regex.replace(')', ' ) ')
    regex = regex.replace(',', ' , ')
    if '   ' in regex:
        print('error blank')
        return " "
    else:
        regex = ' '.join(regex.split())
    res = ""
    flag = False
    for index in range(len(regex) - 1, -1, -1):
        if flag and (regex[index] == " " or regex[index] == "("):
            flag = False
            res = '<' + res
        res = regex[index] + res
        if regex[index] == '>':
            flag = True
    if flag:
        res = '<' + res
    res = ''.join(res.split())
    return res


def norm3Str(regex):
    regex = regex.replace('(', ' ( ')
    regex = regex.replace(')', ' ) ')
    regex = regex.replace(',', ' , ')
    if '   ' in regex:
        print('error blank')
        return
    else:
        regex = ' '.join(regex.split())
    res = ""
    flag = False
    t_f = False
    for index in range(len(regex) - 1, -1, -1):
        if flag and t_f and regex[index] == " ":
            flag = False
            t_f = False
            res = '<' + res
        res = regex[index] + res
        if flag and regex[index] != " ":
            t_f = True
        if regex[index] == '>':
            flag = True

    if flag:
        res = '<' + res
    res = ''.join(res.split())
    return res


def norm(regex):
    res = ""
    flag = False
    for index in range(len(regex) - 1, -1, -1):
        if flag and not regex[index].isalpha() and not regex[index].isdigit() and regex[index] != " ":
            flag = False
            res = '<' + res
        res = regex[index] + res
        if regex[index] == '>':
            flag = True
        if regex[index] == 'b':
            res = '\\' + res
    to_repair2 = ['1,', '2,', '3,', '4,', '5,', '6,', '7,', '8,', '9,',
                  '1 ,', '2 ,', '3 ,', '4 ,', '5 ,', '6 ,', '7 ,', '8 ,', '9 ,']
    res = res.replace(" ", '')
    for k in range(len(to_repair2)):
        # lineA = lineA.replace(to_repair[k], '<' + to_repair[k])
        res = res.replace(to_repair2[k], '{' + to_repair2[k] + '}')
        pass
    res = res.replace(" ", '')
    return res


def pre2Original(pre):
    stack = []
    res = ""
    if len(pre) == 1:
        return pre[0]
    for i in range(0, len(pre)):
        op = pre[i]
        if opDict.__contains__(op):
            stack.append(opDict[op])
            res = res + op + ' ( '
        elif setMulti.__contains__(op):
            stack.append(-3)  # mutli
            res = res + op + ' ( '
        elif op != 'end':
            res = res + op
            if stack[-1] != -3:
                stack[-1] = stack[-1] - 1
            while len(stack) > 0 and stack[-1] == 0:
                stack.pop()
                res = res + ' ) '
                if len(stack) > 0:
                    if stack[-1] != -3:
                        stack[-1] = stack[-1] - 1
            if i != len(pre) - 1:
                res = res + ' , '
        else:
            res = res + ' ) '
            stack.pop()
            if len(stack) > 0:

                if stack[-1] == -3:
                    res = res + ' , '
                if stack[-1] != -3:
                    stack[-1] = stack[-1] - 1
                    if stack[-1] != 0:
                        res = res + ' , '
                    while len(stack) > 0 and stack[-1] == 0:
                        stack.pop()
                        res = res + ' ) '
                        if len(stack) > 0:
                            if stack[-1] != -3:
                                stack[-1] = stack[-1] - 1
                                res = res + ' , '
                            if stack[-1] == -3:
                                res = res + ' , '

    res = res.replace(",  )", ")")
    res = res.replace("  ", " ")
    if len(res) >= 1 and res[len(res) - 1] == ' ':
        res = res[0:len(res) - 1]
    # print('res',res)
    return res
    # for i in range


def bina_pre2Original(pre):
    stack = []
    res = ""
    if len(pre) == 1:
        return pre[0]
    for i in range(0, len(pre)):
        op = pre[i]
        if bina_opDict.__contains__(op):
            stack.append(bina_opDict[op])
            res = res + op + ' ( '
        else:
            res = res + op
            if len(stack) == 0:
                return " "
            stack[-1] = stack[-1] - 1
            while len(stack) > 0 and stack[-1] == 0:
                stack.pop()
                res = res + ' ) '
                if len(stack) > 0:
                    stack[-1] = stack[-1] - 1
            if i != len(pre) - 1:
                res = res + ' , '
    res = res.replace(",  )", ")")
    res = res.replace("  ", " ")
    if len(res) >= 1 and res[len(res) - 1] == ' ':
        res = res[0:len(res) - 1]
    # print('res',res)
    return res
    # for i in range


def eval_StrScratachpad(regex):
    tmp = regex.split(' ')

    # tmp = tmp[:-1]

    for i in range(len(tmp)):
        tmp[i] = tmp[i][tmp[i].find('=') - 7:]
    # print(tmp)
    if len(tmp) == 0:
        return regex[8:]
    for j in range(len(tmp)):
        tmp[j] = norm3Str(tmp[j][8:])
        # print(tmp[j])
        if ('result' + str(j + 1)) in tmp[j]:
            return ' '

        # multi to binary
        if tmp[j].count(',') >= 2 and ('or' in tmp[j] or 'and' in tmp[j] or 'concat' in tmp[j]):
            op = tmp[j][:tmp[j].find('(')]
            tokens = tmp[j][tmp[j].find('(') + 1:-1].split(',')
            while len(tokens) != 1:
                a = tokens.pop(0)
                b = tokens.pop(0)
                tokens.insert(0, op + '(' + a + ',' + b + ')')
            tmp[j] = tokens[0]

    res = tmp[-1]

    while res.find('result') != -1:

        idx = res.find('result')
        if len(res) <= idx + 6 or not res[idx + 6].isdigit():
            return ' '
        p = int(res[idx + 6])
        if len(res) <= idx + 7:
            return " "
        res = res[:idx] + tmp[p - 1] + res[idx + 7:]
    # print(res)
    return res


def eval_Scratachpad(regex):
    regex = regex.replace(',>','tmpToken>')
    tmp = regex.split(' ')
    for j in range(len(tmp)):
        tmp[j] = norm3(tmp[j][8:])
        # print(tmp[j])
        if ('result' + str(j + 1)) in tmp[j]:
            return ' '

        # multi to binary
        if tmp[j].count(',') >= 2 and ('or' in tmp[j] or 'and' in tmp[j] or 'concat' in tmp[j]):
            op = tmp[j][:tmp[j].find('(')]
            tokens = tmp[j][tmp[j].find('(') + 1:-1].split(',')
            while len(tokens) != 1:
                a = tokens.pop(0)
                b = tokens.pop(0)
                tokens.insert(0, op + '(' + a + ',' + b + ')')
            tmp[j] = tokens[0]

            # while len(tokens) != 1:
            #     a = tokens.pop()
            #     b = tokens.pop()
            #     tokens.append(op + '(' + b + ',' + a + ')')
            # tmp[j] = tokens[0]

    res = tmp[-1]
    loop = 0
    while res.find('result') != -1 and loop < 25:
        loop = loop + 1
        idx = res.find('result')
        if len(res) <= idx + 7 or not res[idx + 6].isdigit():
            return ' '
        if (res[idx + 7].isdigit()):
            p = int(res[idx + 6:idx + 8])
            if p > len(tmp):
                return ' '
            res = res[:idx] + tmp[p - 1] + res[idx + 8:]
        else:
            p = int(res[idx + 6])
            if p > len(tmp):
                return ' '
            res = res[:idx] + tmp[p - 1] + res[idx + 7:]
    # print(res)
    res = res.replace("=",'')
    res = res.replace('tmpToken>',',>')
    return res


def eval_Scratachpad2(regex):
    reg_args = regex.split(' Arguments: ')
    if len(reg_args) != 2:
        return ' '
    regex = reg_args[0]
    args = reg_args[1]
    # print(regex)
    args = args.split(' ')
    tmp = regex.split(' ')
    if len(args) == 0:
        return ' '
    if len(tmp) == 0:
        return ' '
    # print(tmp)
    for j in range(len(tmp)):
        tmp[j] = norm3(tmp[j][8:])
        # print(tmp[j])
        # print(tmp[j])
        if ('result' + str(j + 1)) in tmp[j]:
            return ' '

        # multi to binary
        if tmp[j].count(',') >= 2 and ('or' in tmp[j] or 'and' in tmp[j] or 'concat' in tmp[j]):
            op = tmp[j][:tmp[j].find('(')]
            tokens = tmp[j][tmp[j].find('(') + 1:-1].split(',')
            while len(tokens) != 1:
                a = tokens.pop(0)
                b = tokens.pop(0)
                tokens.insert(0, op + '(' + a + ',' + b + ')')
            tmp[j] = tokens[0]

    res = tmp[-1]
    # print(res)
    while res.find('result') != -1:

        idx = res.find('result')
        if len(res) <= idx + 6 or not res[idx + 6].isdigit():
            return ' '
        p = int(res[idx + 6])
        res = res[:idx] + tmp[p - 1] + res[idx + 7:]
    # print(args)
    for i in range(len(args)):
        res = res.replace('argument' + str(i + 1), args[i][10:])
        pass
    res = norm3(res)
    # print(res)
    return res


def eval_Scratachpad3(regex):
    tmp = regex.split(' ')
    for j in range(len(tmp)):
        tmp[j] = norm3(tmp[j][8:])
        # print(tmp[j])
        if ('result' + str(j + 1)) in tmp[j]:
            return ' '
        # multi to binary
        if tmp[j].count(',') >= 2 and ('or' in tmp[j] or 'and' in tmp[j] or 'concat' in tmp[j]):
            op = tmp[j][:tmp[j].find('(')]
            tokens = tmp[j][tmp[j].find('(') + 1:-1].split(',')
            while len(tokens) != 1:
                a = tokens.pop(0)
                b = tokens.pop(0)
                tokens.insert(0, op + '(' + a + ',' + b + ')')
            tmp[j] = tokens[0]

    res = tmp[-1]

    while res.find('result') != -1:

        idx = res.find('result')
        if len(res) <= idx + 6 or not res[idx + 6].isdigit():
            return ' '
        p = int(res[idx + 6])
        res = res[:idx] + tmp[p - 1] + res[idx + 7:]
    # print(res)
    for key, value in argument_Dict.items():
        res = res.replace(key, value)
        pass
    # print(res)
    return res


def eval_Scratachpad5(regex):
    if not ':' in regex:
        return regex
    for i in range(len(regex) - 1, 8, -1):
        if regex[i] == ':':
            regex = regex[:i - 8] + '<STEP>' + regex[i - 8:]
    tmp = regex.split('<STEP> ')
    # tmp = tmp[:-1]
    if len(tmp) == 0 and len(regex) > 8:
        return regex[8:]
    elif len(tmp) == 0:
        return " "
    for j in range(len(tmp)):
        tmp[j] = parse_Scratachpad(tmp[j])
        tmp[j] = norm3(tmp[j][8:])
        # print(tmp[j])
        if ('result' + str(j + 1)) in tmp[j]:
            return ' '
        # multi to binary
        if tmp[j].count(',') >= 2 and ('or' in tmp[j] or 'and' in tmp[j] or 'concat' in tmp[j]):
            op = tmp[j][:tmp[j].find('(')]
            tokens = tmp[j][tmp[j].find('(') + 1:-1].split(',')
            while len(tokens) != 1:
                a = tokens.pop(0)
                b = tokens.pop(0)
                tokens.insert(0, op + '(' + a + ',' + b + ')')
            tmp[j] = tokens[0]

    res = tmp[-1]

    while res.find('result') != -1:

        idx = res.find('result')
        if len(res) <= idx + 6 or not res[idx + 6].isdigit():
            return ' '
        p = int(res[idx + 6])
        if len(res) <= idx + 7:
            return " "
        res = res[:idx] + tmp[p - 1] + res[idx + 7:]
    # print(res)
    return res


def eval_Scratachpad6(regex):
    if not ':' in regex:
        return regex
    output = regex.split('; Output:')[1]
    regex = regex.split('; Output:')[0]
    for i in range(len(regex) - 1, 8, -1):
        if regex[i] == ':':
            regex = regex[:i - 8] + '<STEP>' + regex[i - 8:]
    tmp = regex.split('<STEP> ')
    # tmp = tmp[:-1]
    if len(tmp) == 0 and len(regex) > 8:
        return regex[8:]
    elif len(tmp) == 0:
        return " "
    for j in range(len(tmp)):
        tmp[j] = parse_Scratachpad(tmp[j])
        tmp[j] = norm3(tmp[j][8:])
        # print(tmp[j])
        if ('result' + str(j + 1)) in tmp[j]:
            return ' '
        # multi to binary
        if tmp[j].count(',') >= 2 and ('or' in tmp[j] or 'and' in tmp[j] or 'concat' in tmp[j]):
            op = tmp[j][:tmp[j].find('(')]
            tokens = tmp[j][tmp[j].find('(') + 1:-1].split(',')
            while len(tokens) != 1:
                a = tokens.pop(0)
                b = tokens.pop(0)
                tokens.insert(0, op + '(' + a + ',' + b + ')')
            tmp[j] = tokens[0]
    res = tmp[-1]

    while res.find('result') != -1:

        idx = res.find('result')
        if len(res) <= idx + 6 or not res[idx + 6].isdigit():
            return ' '
        p = int(res[idx + 6])
        if len(res) <= idx + 7:
            return " "
        res = res[:idx] + tmp[p - 1] + res[idx + 7:]
    # print(res)
    return res


def eval_Scratachpad7(regex):
    if ': <' in regex:
        return regex[regex.find(': <') + 2: -1]
    tmp = regex.split(' ; ')
    # tmp = tmp[:-1]
    if len(tmp) == 0 and len(regex) > 8:
        return regex[8:]
    elif len(tmp) == 0:
        return " "
    for j in range(len(tmp)):
        tmp[j] = parse_Scratachpad7(tmp[j])
        tmp[j] = norm3(tmp[j][8:])
        # print(tmp[j])
        if ('result' + str(j + 1)) in tmp[j]:
            return ' '
        # multi to binary
        if tmp[j].count(',') >= 2 and ('or' in tmp[j] or 'and' in tmp[j] or 'concat' in tmp[j]):
            op = tmp[j][:tmp[j].find('(')]
            tokens = tmp[j][tmp[j].find('(') + 1:-1].split(',')
            while len(tokens) != 1:
                a = tokens.pop(0)
                b = tokens.pop(0)
                tokens.insert(0, op + '(' + a + ',' + b + ')')
            tmp[j] = tokens[0]

    res = tmp[-1]

    while res.find('result') != -1:

        idx = res.find('result')
        if len(res) <= idx + 6 or not res[idx + 6].isdigit():
            return ' '
        # p = int(res[idx + 6])
        if len(res) <= idx + 7:
            return " "
        # res = res[:idx] + tmp[p - 1] + res[idx + 7:]

        if (res[idx + 7].isdigit()):
            p = int(res[idx + 6:idx + 8])
            if p > len(tmp) or len(res) <= idx + 8:
                return ' '
            res = res[:idx] + tmp[p - 1] + res[idx + 8:]
        else:
            p = int(res[idx + 6])
            if p > len(tmp) or len(res) <= idx + 7:
                return ' '
            res = res[:idx] + tmp[p - 1] + res[idx + 7:]

    # print(res)
    return res


# def eval_Scratachpad4(regex):
#
#     tmp = regex.split(';')
#     tmp = tmp[:-1]
#
#
#     for i in range(len(tmp)):
#         tmp[i] = tmp[i][tmp[i].find('=')-7:]
#     #print(tmp)
#     if len(tmp) == 0 and len(regex)>8:
#         return regex[8:]
#     elif len(tmp) == 0:
#         return " "
#     for j in range(len(tmp)):
#
#         tmp[j] = norm3(tmp[j][8:])
#         # print(tmp[j])
#         if ('result' + str(j+1)) in tmp[j]:
#             return ' '
#         #multi to binary
#         if tmp[j].count(',') >= 2 and ('or' in tmp[j] or 'and' in tmp[j] or 'concat' in tmp[j]):
#             op = tmp[j][:tmp[j].find('(')]
#             tokens = tmp[j][tmp[j].find('(')+1:-1].split(',')
#             while len(tokens) != 1:
#                 a = tokens.pop(0)
#                 b = tokens.pop(0)
#                 tokens.insert(0, op + '(' + a + ',' + b + ')')
#             tmp[j] = tokens[0]
#
#     res = tmp[-1]
#
#     while res.find('result') != -1:
#
#         idx = res.find('result')
#         if len(res) <= idx+6 or not res[idx+6].isdigit():
#             return ' '
#         p = int(res[idx+6])
#         if len(res) <= idx + 7:
#             return " "
#         res = res[:idx] + tmp[p-1] + res[idx+7:]
#     #print(res)
#     return res

def bina_post2Original(post):
    stack = []

    res = ""
    if len(post) == 1:
        return post[0]
    # while True:

    for i in range(0, len(post)):
        op = post[i]
        if bina_opDict.__contains__(op):
            tmpStr = ""
            for _stack_index in range(bina_opDict[op]):
                if len(stack) == 0:
                    return " "
                tmpStr = stack.pop() + ' ' + tmpStr
            tmpStr = op + ' ' + tmpStr
            stack.append(tmpStr)
        else:
            stack.append(op)
    if len(stack) != 1:
        print('error stack')
        return " "
    else:
        res = stack[0]
    res = res.replace(",  )", ")")
    res = res.replace("   ", " ")
    res = res.replace("  ", " ")
    if len(res) >= 1 and res[len(res) - 1] == ' ':
        res = res[0:len(res) - 1]
    res = res.split(" ")
    # print(post)
    res = bina_pre2Original(res)
    # print('res',res)
    return res


class CustomTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        print("outputs________________________________________________________")
        print(outputs[2])
        print(len(outputs[2]))
        # tokenizer = model.tokenizer
        logits = outputs.get("logits")
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        # loss = 0
        '''
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0]))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        '''

        loss = None
        # tokenizer = T5Tokenizer.from_pretrained("t5-base")
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            tmp_sum = 0
            for idx in range(len(logits)):
                tmp = loss_fct(logits[idx].view(-1, logits[idx].size(-1)), labels[idx].view(-1))

                # print('??', len(logits[idx]))
                lo = []
                la = []
                lo = torch.argmax(logits[idx], axis=1)
                # print(tokenizer.convert_ids_to_tokens([1, 2, 3, 4, 5, 6, 20, 21, 22, 23, 24, 25, 26, 27, 28]))
                '''
                for i in range(len(lo)):
                    #if logits[idx][i] != -100:
                    lo.append(logits[idx][i])
                '''
                for j in range(len(labels[idx])):
                    if labels[idx][j] != -100:
                        la.append(labels[idx][j])
                # print('lo', len(lo))
                # print('lo___', lo)
                # print('la', len(la))
                decoded_preds = tokenizer.decode(lo, skip_special_tokens=True)
                decoded_labels = tokenizer.decode(la, skip_special_tokens=True)
                # print('preds', decoded_preds)
                # print('labels', decoded_labels)
                # print(idx,tmp)
                tmp_sum = tmp + tmp_sum
            # print('sum___',tmp_sum)
            # print(loss)
            loss = tmp_sum / len(logits)

        return (loss, outputs) if return_outputs else loss


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    source_lang: str = field(default=None, metadata={"help": "Source language id for translation."})
    target_lang: str = field(default=None, metadata={"help": "Target language id for translation."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a jsonlines)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (sacreblue) on "
                    "a jsonlines file."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (sacreblue) on " "a jsonlines file."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the :obj:`decoder_start_token_id`."
                    "Useful for multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token "
                    "needs to be the target language token.(Usually it is the target language token)"
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        elif self.source_lang is None or self.target_lang is None:
            raise ValueError("Need to specify the source language and the target language.")

        # accepting both json and jsonl file extensions, as
        # many jsonlines files actually have a .json extension
        valid_extensions = ["json", "jsonl"]

        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in valid_extensions, "`train_file` should be a jsonlines file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in valid_extensions, "`validation_file` should be a jsonlines file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is expected, e.g. with "
            "`--source_prefix 'translate English to German: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    new_tokens = ['~']
    tokenizer.add_tokens(new_tokens)
    special_tokens_dict = {'additional_special_tokens': ['<STEP>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    model.resize_token_embeddings(len(tokenizer))

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.target_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.target_lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # For translation we set the codes of our source and target languages (only useful for mBART, the others will
    # ignore those attributes).
    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert data_args.target_lang is not None and data_args.source_lang is not None, (
            f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --source_lang and "
            "--target_lang arguments."
        )

        tokenizer.src_lang = data_args.source_lang
        tokenizer.tgt_lang = data_args.target_lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id

    # Get the language codes for input/target.
    source_lang = data_args.source_lang.split("_")[0]
    target_lang = data_args.target_lang.split("_")[0]

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):

        inputs = [ex[source_lang] for ex in examples["translation"]]
        # print(raw_datasets["train"])
        all_train_f = []
        idx_point = 0
        locate = False
        # if len(inputs) < 450:
        #     print('structed regex should relpace the src.txt')
        #     exit(1)
        with open('/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/src-train.txt', "r") as train_f:
            line = train_f.readline()
            # check line is not empty
            while line:
                # line = line.replace('\n','')
                all_train_f.append(line)
                line = train_f.readline()
        while idx_point < len(all_train_f):
            print(all_train_f[idx_point])
            print(inputs[0])
            # print(idx_point)
            if inputs[0] == all_train_f[idx_point]:
                locate = False
                break
            idx_point = idx_point + 1000
        print(idx_point)
        with open('/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/me.txt', "a") as f3:
            f3.write(str(idx_point) + '\n')
        '''
        if locate == False:
            print('begin error', inputs[0])
            print(all_train_f[0])
            exit(1)
        for tmp_in_idx in range(len(inputs)):
            if inputs[tmp_in_idx] != all_train_f[idx_point + tmp_in_idx]:
                print('middle error', inputs[tmp_in_idx])
                exit(1)
        print('finish', idx_point)
        print('len_inputs',len(inputs))
        '''

        targets = [ex['regex'] for ex in examples["translation"]]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        if locate:
            for input_idx in range(len(model_inputs['input_ids'])):
                model_inputs['input_ids'][input_idx].append(11)
                model_inputs['attention_mask'][input_idx].append(0)

        # model_inputs['input_ids'][0].append(0)
        # model_inputs['attention_mask'][0].append(0)
        # print(model_inputs['input_ids'])
        # print('___________________________________')
        # print(tokenizer.pad_token_id)
        # print(len(model_inputs['input_ids']))
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
            print(1111111111111111111111111111111111111111111111112222222222222222222222)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            print(train_dataset)

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        print(predict_dataset[0])
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
            #print(predict_dataset)

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # Metric
    metric = load_metric("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        # Swriter = SummaryWriter('./logs/63')

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        print('preds', preds)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        print('labels', labels)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        reg_eq = 0
        reg_accr = 0
        perf = 0
        print('____________________________________________')
        equiv_flag = False

        if 'AAAAAAAAStrScratachpad' in data_args.train_file:
            for idx in range(len(decoded_preds)):
                aaa = ''.join(decoded_preds[idx].replace('\n', ''))
                bbb = ''.join(decoded_labels[idx][0].replace('\n', ''))
                # print(idx)
                if idx % 200 == 0:
                    print(idx)
                    print(reg_eq)
                # print('____________________________________________')
                f1 = False
                f2 = False
                if aaa == bbb:
                    perf = perf + 1
                    f1 = True
                else:
                    # print('pred', '  '.join(decoded_preds[idx].replace('\n', '')))
                    # print('label', '  '.join(decoded_labels[idx][0].replace('\n', '')))
                    pass
                aaa = eval_StrScratachpad(aaa)
                bbb = eval_StrScratachpad(bbb)
                aaa = aaa.replace(' ', '')
                bbb = bbb.replace(' ', '')
                if training_args.do_eval:
                    with open('/mnt/sda/zs/ge/transformers/examples/pytorch/translation/tran_predict.txt',
                              "a+") as f1, open(
                        '/mnt/sda/zs/ge/transformers/examples/pytorch/translation/tran_labels.txt', "a+") as f2:
                        f1.write(aaa + '\n')
                        f2.write(bbb + '\n')

                # print(aaa)
                # print(bbb)

                # print(idx)
                # print('aaa', aaa)
                # print('bbb',bbb)
                # if dfa_eual_test(aaa, bbb):

                if equiv_flag and check_equiv(aaa, bbb):
                    reg_eq = reg_eq + 1
                    f2 = True
        elif 'PostScratachpad7' in data_args.train_file:
            for idx in range(len(decoded_preds)):
                aaa = ''.join(decoded_preds[idx].replace('', ''))
                bbb = ''.join(decoded_labels[idx][0].replace('', ''))
                # print(idx)
                if idx % 200 == 0:
                    print(idx)
                    print(reg_eq)
                # print('____________________________________________')
                f1 = False
                f2 = False
                if aaa == bbb:
                    perf = perf + 1
                    f1 = True
                else:
                    # print('pred', '  '.join(decoded_preds[idx].replace('\n', '')))
                    # print('label', '  '.join(decoded_labels[idx][0].replace('\n', '')))
                    pass

                # print(aaa)
                # print(bbb)
                aaa = eval_Scratachpad7(aaa)
                bbb = eval_Scratachpad7(bbb)
                if training_args.do_eval:
                    with open('/mnt/sda/zs/ge/transformers/examples/pytorch/translation/tran_predict.txt',
                              "a+") as f1, open(
                        '/mnt/sda/zs/ge/transformers/examples/pytorch/translation/tran_labels.txt', "a+") as f2:
                        f1.write(aaa + '\n')
                        f2.write(bbb + '\n')
                # print(aaa)
                # print(bbb)
                aaa = aaa.replace(' ', '')
                bbb = bbb.replace(' ', '')

                # print(idx)
                # print('aaa', aaa)
                # print('bbb',bbb)
                # if dfa_eual_test(aaa, bbb):

                if equiv_flag and check_equiv(aaa, bbb):
                    reg_eq = reg_eq + 1
                    f2 = True
        elif 'PostScratachpad3' in data_args.train_file:
            for idx in range(len(decoded_preds)):
                aaa = ''.join(decoded_preds[idx].replace('\n', ''))
                bbb = ''.join(decoded_labels[idx][0].replace('\n', ''))
                # print(idx)
                if idx % 200 == 0:
                    print(idx)
                    print(reg_eq)
                # print('____________________________________________')
                f1 = False
                f2 = False
                if aaa == bbb:
                    perf = perf + 1
                    f1 = True
                else:
                    # print('pred', '  '.join(decoded_preds[idx].replace('\n', '')))
                    # print('label', '  '.join(decoded_labels[idx][0].replace('\n', '')))
                    pass
                if training_args.do_eval:
                    with open('/mnt/sda/zs/ge/transformers/examples/pytorch/translation/tran_predict.txt',
                              "a+") as f1, open(
                        '/mnt/sda/zs/ge/transformers/examples/pytorch/translation/tran_labels.txt', "a+") as f2:
                        f1.write(aaa + '\n')
                        f2.write(bbb + '\n')

                # print(aaa)
                # print(bbb)
                aaa = eval_Scratachpad3(aaa)
                bbb = eval_Scratachpad3(bbb)
                aaa = aaa.replace(' ', '')
                bbb = bbb.replace(' ', '')
                # print(idx)
                # print('aaa', aaa)
                # print('bbb',bbb)
                # if dfa_eual_test(aaa, bbb):

                if equiv_flag and check_equiv(aaa, bbb):
                    reg_eq = reg_eq + 1
                    f2 = True
        elif 'PostScratachpad2' in data_args.train_file and not 'new' in data_args.train_file:
            for idx in range(len(decoded_preds)):
                aaa = ''.join(decoded_preds[idx].replace('\n', ''))
                bbb = ''.join(decoded_labels[idx][0].replace('\n', ''))
                # print(idx)
                if idx % 200 == 0:
                    print(idx)
                    print(reg_eq)
                # print('____________________________________________')
                f1 = False
                f2 = False
                if aaa == bbb:
                    perf = perf + 1
                    f1 = True
                else:
                    # print('pred', '  '.join(decoded_preds[idx].replace('\n', '')))
                    # print('label', '  '.join(decoded_labels[idx][0].replace('\n', '')))
                    pass
                if training_args.do_eval:
                    with open('/mnt/sda/zs/ge/transformers/examples/pytorch/translation/tran_predict.txt',
                              "a+") as f1, open(
                        '/mnt/sda/zs/ge/transformers/examples/pytorch/translation/tran_labels.txt', "a+") as f2:
                        f1.write(aaa + '\n')
                        f2.write(bbb + '\n')

                # print(aaa)
                # print(bbb)
                aaa = eval_Scratachpad2(aaa)
                bbb = eval_Scratachpad2(bbb)
                aaa = aaa.replace(' ', '')
                bbb = bbb.replace(' ', '')
                # print(idx)
                # print('aaa', aaa)
                # print('bbb',bbb)
                # if dfa_eual_test(aaa, bbb):

                if equiv_flag and check_equiv(aaa, bbb):
                    reg_eq = reg_eq + 1
                    f2 = True

        elif 'PostScratachpad' in data_args.train_file:
            for idx in range(len(decoded_preds)):
                aaa = ''.join(decoded_preds[idx].replace('\n', ''))
                bbb = ''.join(decoded_labels[idx][0].replace('\n', ''))
                # print(idx)
                if idx % 200 == 0:
                    print(idx)
                    print(reg_eq)
                # print('____________________________________________')
                f1 = False
                f2 = False
                if aaa == bbb:
                    perf = perf + 1
                    f1 = True
                else:
                    # print('pred', '  '.join(decoded_preds[idx].replace('\n', '')))
                    # print('label', '  '.join(decoded_labels[idx][0].replace('\n', '')))
                    pass

                to_repl = {'int0': '2',
                           'int1': '3',
                           'int2': '4',
                           'int3': '5',
                           'int4': '6',
                           'int5': '7',
                           'int6': '8',
                           'int7': '9',
                           'int8': '10'}
                for key, val in to_repl.items():
                    aaa = aaa.replace(key, val)
                    bbb = bbb.replace(key, val)



                # print(aaa)
                # print(bbb)
                aaa = eval_Scratachpad(aaa)
                bbb = eval_Scratachpad(bbb)
                aaa = aaa.replace(' ', '')
                bbb = bbb.replace(' ', '')
                if training_args.do_eval:
                    with open('/mnt/sda/zs/ge/transformers/examples/pytorch/translation/tran_predict.txt',
                              "a+") as f1, open(
                        '/mnt/sda/zs/ge/transformers/examples/pytorch/translation/tran_labels.txt', "a+") as f2:
                        f1.write(aaa + '\n')
                        f2.write(bbb + '\n')
                # print(idx)
                # print('aaa', aaa)
                # print('bbb',bbb)
                # if dfa_eual_test(aaa, bbb):

                if equiv_flag and check_equiv(aaa, bbb):
                    reg_eq = reg_eq + 1
                    f2 = True

        elif 'AST' in data_args.train_file:
            for idx in range(len(decoded_preds)):

                aaa = norm3(''.join(decoded_preds[idx].replace('\n', '')))
                bbb = norm3(''.join(decoded_labels[idx][0].replace('\n', '')))
                if idx % 200 == 0:
                    print(idx)
                # print('____________________________________________')
                f1 = False
                f2 = False
                if aaa == bbb:
                    perf = perf + 1
                    f1 = True
                else:
                    # print('pred', '  '.join(decoded_preds[idx].replace('\n', '')))
                    # print('label', '  '.join(decoded_labels[idx][0].replace('\n', '')))
                    pass
                # print(aaa)
                # print(bbb)
                # clear
                to_repl = {'int0': '2',
                           'int1': '3',
                           'int2': '4',
                           'int3': '5',
                           'int4': '6',
                           'int5': '7',
                           'int6': '8',
                           'int7': '9',
                           'int8': '10'}
                for key, val in to_repl.items():
                    aaa = aaa.replace(key, val)
                    bbb = bbb.replace(key, val)

                if training_args.do_eval:
                    with open('/mnt/sda/zs/ge/transformers/examples/pytorch/translation/tran_predict.txt',
                              "a+") as f1, open(
                        '/mnt/sda/zs/ge/transformers/examples/pytorch/translation/tran_labels.txt', "a+") as f2:
                        f1.write(aaa + '\n')
                        f2.write(bbb + '\n')

                # if dfa_eual_test(aaa, bbb):

                if equiv_flag and check_equiv(aaa, bbb):
                    reg_eq = reg_eq + 1
                    f2 = True


        elif 'Pre' in data_args.train_file:
            for idx in range(len(decoded_preds)):
                aaa = norm2(''.join(decoded_preds[idx].replace('\n', '')))
                bbb = norm2(''.join(decoded_labels[idx][0].replace('\n', '')))
                if idx % 200 == 0:
                    print(idx)
                # print('____________________________________________')
                f1 = False
                f2 = False
                if aaa == bbb:
                    perf = perf + 1
                    f1 = True
                else:
                    # print('pred', '  '.join(decoded_preds[idx].replace('\n', '')))
                    # print('label', '  '.join(decoded_labels[idx][0].replace('\n', '')))
                    pass
                if training_args.do_eval:
                    with open('/mnt/sda/zs/ge/transformers/examples/pytorch/translation/tran_predict.txt',
                              "a+") as f1, open(
                        '/mnt/sda/zs/ge/transformers/examples/pytorch/translation/tran_labels.txt', "a+") as f2:
                        f1.write(aaa + '\n')
                        f2.write(bbb + '\n')

                aaa = aaa.split()
                bbb = bbb.split()
                aaa = bina_pre2Original(aaa)
                bbb = bina_pre2Original(bbb)
                aaa = aaa.replace(' ', '')
                bbb = bbb.replace(' ', '')
                # print('aaa', aaa)
                # print(bbb)
                # if dfa_eual_test(aaa, bbb):

                if equiv_flag and check_equiv(aaa, bbb):
                    reg_eq = reg_eq + 1
                    f2 = True

        else:
            for idx in range(len(decoded_preds)):
                aaa = norm2(''.join(decoded_preds[idx].replace('\n', '')))
                bbb = norm2(''.join(decoded_labels[idx][0].replace('\n', '')))
                if idx % 200 == 0:
                    print(idx)
                # print('____________________________________________')
                f1 = False
                f2 = False
                if aaa == bbb:
                    perf = perf + 1
                    f1 = True
                else:
                    # print('pred', '  '.join(decoded_preds[idx].replace('\n', '')))
                    # print('label', '  '.join(decoded_labels[idx][0].replace('\n', '')))
                    pass
                if training_args.do_eval:
                    with open('/mnt/sda/zs/ge/transformers/examples/pytorch/translation/tran_predict.txt',
                              "a+") as f1, open(
                        '/mnt/sda/zs/ge/transformers/examples/pytorch/translation/tran_labels.txt', "a+") as f2:
                        f1.write(aaa + '\n')
                        f2.write(bbb + '\n')
                aaa = aaa.split()
                bbb = bbb.split()
                aaa = bina_post2Original(aaa)
                bbb = bina_post2Original(bbb)
                aaa = aaa.replace(' ', '')
                bbb = bbb.replace(' ', '')
                # if dfa_eual_test(aaa, bbb):

                if equiv_flag and check_equiv(aaa, bbb):
                    reg_eq = reg_eq + 1
                    f2 = True

        print('____________________________________________')

        print('reg_eq: ', reg_eq)
        print('perf:', perf)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result["reg"] = reg_eq / len(decoded_preds)
        result["perf"] = perf / len(decoded_preds)
        result = {k: round(v, 4) for k, v in result.items()}
        # print(result)

        return result

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    writer.write("\n".join(predictions))

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "translation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    languages = [l for l in [data_args.source_lang, data_args.target_lang] if l is not None]
    if len(languages) > 0:
        kwargs["language"] = languages

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    '''
    if dfa_eual_test('[< NUM> ] | [ LET> ]',
                     '[ LET> ] | [ NUM> ]'):
        print('1111')
    else:
        print('no')
    '''
    # print(norm('( \ b ( [ <NUM> ] ) & ( <M0> ) \ b ) ( . * )'))
    main()
    # eval_Scratachpad2('result1=repeat(argument1,argument2) result2=or(result1,argument3) Arguments: argument1=m0> argument2=vow> argument3=any>')
    # print(dfa_eual_test(aaa, bbb))
    # check equivalance
    # print('EXPECTED TRUE', check_equiv('concat(<any>,and(and(<vow>,<m0>),<m1>))', 'concat(<any>,and(and(<m0>,<m1>),<num>))'))
    # print('end..................')
    '''
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    #new_tokens = ['<', '{', '}', '~', '\\']
    #tokenizer.add_tokens(new_tokens)

    print(tokenizer.encode("items with a letter and a numeral ."))
    print(tokenizer.encode("items with a letter and a numeral . 0")) 
    print(tokenizer.encode("items with a letter and a numeral . 1"))    
    print(tokenizer.encode("items with a letter and a numeral . 2")) 
    print(tokenizer.encode("items with a letter and a numeral . 3"))
    print(tokenizer.encode("items with a letter and a numeral . 4")) 
    print(tokenizer.encode("items with a letter and a numeral . 5"))
    print(tokenizer.encode("items with a letter and a numeral . 6")) 
    print(tokenizer.encode("items with a letter and a numeral . 7")) 
    print(tokenizer.encode("items with a letter and a numeral . 8")) 
    print(tokenizer.encode("items with a letter and a numeral . 9")) 
    print(tokenizer.encode("items with a letter and a numeral . 1 0")) 
    print(tokenizer.encode("items with a letter and a numeral . 1 1"))   
    '''
    '''
    with open("/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tran_test.txt", "r") as f1:
        # Read next line
        i = 0
        line = f1.readline()
        # check line is not empty
        while line:
            line = f1.readline()
            ids = tokenizer.encode(line)
            de = tokenizer.decode(ids, skip_special_tokens=True)
            de = de.replace(" ","")
            line = line.replace(" ","")
            line = line.replace("\n","")
            if de != line:
                print(line)
                print(de)
                break
    '''
