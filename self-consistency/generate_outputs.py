from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
from datasets import load_dataset
import numpy as np
import re
from easy_eval.eval import check_equiv
from easy_eval.streg_utils import parse_spec_to_ast

def norm3(regex):
    regex = regex.replace('(', ' ( ')
    regex = regex.replace(')', ' ) ')
    regex = regex.replace(',', ' , ')
    if '   ' in regex:
        print('error blank')
        return ' '
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

def eval_Scratachpad(regex):
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
    return res


def eval_Scratachpad2(regex):
    if regex == '':
        return ''
    #regex = re.sub("^[A-Za-z0-9]", " ", regex)
    # tmp_regex = regex
    # regex = ''
    # for i in range(len(tmp_regex)):
    #     if tmp_regex[i].isdigit() or tmp_regex[i].isalpha() or tmp_regex[i] == '(' or tmp_regex[i] == ')' or tmp_regex[i] == ','\
    #             or tmp_regex[i] == '=' or tmp_regex[i] == ' ':
    #         regex = regex + tmp_regex[i]
    # print('regex:', regex)
    regex = regex.replace('  ', ' ')
    regex = regex.replace(',>', 'tmpToken>')
    tmp = regex.split(' ')
    #print(tmp)
    tmp2 = tmp.copy()
    tmp = []
    #print(tmp2)
    for i in range(len(tmp2)):
        if 'result' in tmp2[i]:
            tmp.append(tmp2[i])
    print('tmp',tmp)

    for j in range(len(tmp)):
        #tmp[j] = norm3(tmp[j][8:])
        tmp[j] = tmp[j][8:]
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
        if len(res) <= idx + 6 or not res[idx + 6].isdigit():
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
            #print(res)
    # print(res)
    res = res.replace("=",'')
    res = res.replace('tmpToken>',',>')
    return res


def GetResult(path,file, tar_cache,num_return_sequences):
    batch_size = 12
    gens = []
    gens2 = []
    # tokenizer = T5Tokenizer.from_pretrained(path)
    #model = T5ForConditionalGeneration.from_pretrained(path)
    tokenizer = BartTokenizer.from_pretrained(path)
    model = BartForConditionalGeneration.from_pretrained(path)
    #t5-small
    # tokenizer = T5Tokenizer.from_pretrained("t5-base")
    # model = T5ForConditionalGeneration.from_pretrained("t5-base")

    # # training
    # input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
    # >> > labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
    # >> > outputs = model(input_ids=input_ids, labels=labels)
    # >> > loss = outputs.loss
    # >> > logits = outputs.logits

    # inference
    #input_ids = tokenizer("translate English to English: lines with words and a vowel before lower-case letter or string <m0>", return_tensors="pt").input_ids  # Batch size 1

    raw_datasets = load_dataset('json', data_files=file)
    #print(raw_datasets['train'][0])
    inputs = [ex["translation"]['en'] for ex in raw_datasets['train']]
    prefix = 'translate English to English: '
    inputs = [prefix + inp for inp in inputs]
    #print(len(inputs))
    length = len(inputs)
    idx = 0
    while idx < length:
        input_ids = tokenizer(inputs[idx:min(length,idx+batch_size)],
                              max_length=1024, padding=True, truncation=True,
                              return_tensors="pt").input_ids  # Batch size 1
        #print(input_ids)
        outputs = np.array(model.generate(input_ids, max_length=1024, do_sample=True, num_return_sequences=num_return_sequences,  output_scores=True))
        # outputs = np.array(model.generate(input_ids, max_length=1024, do_sample=False, num_return_sequences=num_return_sequences,  output_scores=True))
        #outputs = outputs.reshape(batch_size,num_return_sequences,-1)
        # outputs = np.array(
        #    model.generate(input_ids, max_length=1024, num_beams=num_return_sequences, num_return_sequences=num_return_sequences,
        #                   output_scores=False,early_stopping=True))
        # print(outputs)
        #outputs = np.array(tokenizer.batch_decode(outputs,skip_special_tokens=True))

        tmp_outputs = []
        for i in range(len(outputs)):
            #print(outputs[i])
            tmp_outputs.append(tokenizer.decode(outputs[i], skip_special_tokens=True))
        outputs = np.array(tmp_outputs)

        #print(outputs)
        for i in range(len(outputs)):
            outputs[i] = outputs[i].strip()
            outputs[i] = re.sub(r'<extra_id_..>', "", outputs[i])
            outputs[i] = re.sub('Its', "", outputs[i])
        tmpO = outputs
        tmpO = tmpO.reshape(-1, num_return_sequences)
        tmpO = tmpO.tolist()
        gens2.extend(tmpO)
        #outputs = outputs.tolist()
        for i in range(len(outputs)):
            #print(outputs[i])
            #print(outputs[i][-1])
            if 'bart' not in path:
                if 'PostScratachpad' in path:
                    outputs[i] = eval_Scratachpad(outputs[i])
                    #pass
                else:
                    outputs[i] = norm3(outputs[i])
            else:
                if 'PostScratachpad' in path:
                    outputs[i] = eval_Scratachpad2(outputs[i])
                    #pass


            #print(outputs[i])
            # if outputs[i][-1] != ")":
            #     print(outputs[i][-1]+'1111')
            #     exit(0)
        outputs = np.array(outputs)
        #print(outputs)
        print(outputs.shape)
        outputs = outputs.reshape(-1, num_return_sequences)
        outputs = outputs.tolist()
        gens.extend(outputs)
        idx += batch_size
        print(idx)
    print('res', len(gens))
    with open(path+'/res2.txt', 'w') as f1:
        for g in gens2:
            f1.write("\n".join(g))
            f1.write('\n')
    #print(gens[0])
    return gens


#,do_sample = True,
# def check():
#     res1 = GetResult("/mnt/sda/zs/ge/transformers/examples/pytorch/translation/sc_models/base_2AST_seed2_AST_bs12_withlog/checkpoint-6233",
#               "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/AST_test.json",
#               "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tar_cache/base_2AST_seed2_AST_bs12_withlog.txt",
#               40)
#     res2 = GetResult(
#         "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/sc_models/only_PostScratachpad_seed3_PostScratachpad_bs12_withlog/checkpoint-5691",
#         "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/PostScratachpad_test.json",
#         "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tar_cache/only_PostScratachpad_seed3_PostScratachpad_bs12_withlog.txt",
#         40)
#     # res1 = GetResult("/mnt/sda/zs/ge/transformers/examples/pytorch/translation/sc_models/base_2AST_seed2_AST_bs12_withlog/checkpoint-6233",
#     #           "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/small.json",
#     #           "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tar_cache/base_2AST_seed2_AST_bs12_withlog.txt",
#     #           1)
#     # res2 = GetResult(
#     #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/sc_models/only_PostScratachpad_seed3_PostScratachpad_bs12_withlog/checkpoint-5691",
#     #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/small.json",
#     #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tar_cache/only_PostScratachpad_seed3_PostScratachpad_bs12_withlog.txt",
#     #     1)
#
#     all_freq = []
#     for i in range(len(res1)):
#         freq = {}
#         single_line_res = res1[i] + res2[i]
#         for cur in single_line_res:
#             match_flag = False
#             for key in freq.keys():
#                 if check_equiv(cur, key):
#                     freq[key] = freq[key] + 1
#                     match_flag = True
#                     break
#             if not match_flag:
#                 freq[cur] = 1
#         freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
#         all_freq.append(freq)
#         if i % 10 == 0:
#             print(i)
#
#     #print(all_freq)
#     with open('/mnt/sda/zs/ge/transformers/examples/pytorch/translation/most_freq_80.txt', 'w') as f1:
#         #for g in gens:
#         for f in all_freq:
#             f1.write(str(f[0][0])+'\n')
#     all_freq = [str(f) for f in all_freq]
#     with open('/mnt/sda/zs/ge/transformers/examples/pytorch/translation/consistency_dict_80.txt', 'w') as f2:
#         #for g in gens:
#         f2.write("\n".join(all_freq))



    # all_freq = []
    # for i in range(len(res1)):
    #     freq = {}
    #     single_line_res = res2[i]# + res1c[i]
    #     for cur in single_line_res:
    #         match_flag = False
    #         for key in freq.keys():
    #             if check_equiv(cur, key):
    #                 freq[key] = freq[key] + 1
    #                 match_flag = True
    #                 break
    #         if not match_flag:
    #             freq[cur] = 1
    #     freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    #     all_freq.append(freq)
    #     if i % 10 == 0:
    #         print(i)
    #
    # #print(all_freq)
    # with open('/mnt/sda/zs/ge/transformers/examples/pytorch/translation/most_freq_onlyPostScratachpad_80.txt', 'w') as f1:
    #     #for g in gens:
    #     for f in all_freq:
    #         f1.write(str(f[0][0])+'\n')
    # all_freq = [str(f) for f in all_freq]
    # with open('/mnt/sda/zs/ge/transformers/examples/pytorch/translation/consistency_dict_onlyPostScratachpad_80.txt', 'w') as f2:
    #     #for g in gens:
    #     f2.write("\n".join(all_freq))


def reinforce():
    res1 = GetResult(
        "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/kb_models/seed1_AST_split1_bs12_withlog/checkpoint-300",
        "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/kb13/AST_train.json",
        "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tar_cache/base_2AST_seed2_AST_bs12_withlog.txt",
        80)
    res2 = GetResult(
        "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/kb_models/seed1_AST_split2_bs12_withlog/checkpoint-300",
        "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/kb13/AST_train.json",
        "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tar_cache/base_2AST_seed2_AST_bs12_withlog.txt",
        80)
    res3 = GetResult(
        "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/kb_models/seed1_AST_split3_bs12_withlog/checkpoint-300",
        "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/kb13/AST_train.json",
        "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tar_cache/base_2AST_seed2_AST_bs12_withlog.txt",
        80)
    res4 = GetResult(
        "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/kb_models/seed1_AST_split4_bs12_withlog/checkpoint-300",
        "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/kb13/AST_train.json",
        "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tar_cache/base_2AST_seed2_AST_bs12_withlog.txt",
        80)
    raw_datasets = load_dataset('json', data_files="/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/kb13/AST_train.json")
    #print(raw_datasets['train'][0])
    outputs = [ex["translation"]['regex'] for ex in raw_datasets['train']]
    all_freq = []
    for i in range(len(outputs)):
        freq = []
        single_line_res = res1[i] + res2[i] + res3[i] + res4[i]
        for cur in single_line_res:
            match_flag = False
            for key in freq:
                if key == cur:
                    match_flag = True
            if not match_flag and cur != outputs[i] and check_equiv(cur,outputs[i]):
                freq.append(cur)
        all_freq.append(freq)
        if i % 10 == 0:
            print(i)
    all_freq = [' || '.join(f) for f in all_freq]
    with open('/mnt/sda/zs/ge/transformers/examples/pytorch/translation/reinforce.txt', 'w') as f2:
        #for g in gens:
        f2.write("\n".join(all_freq))



def check():
    res1 = GetResult(
        "/mnt/sdc/zs/ge/transformers/examples/pytorch/translation/bart-base_models/seed3_AST_bs12_withlog/checkpoint-3523",
        "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/AST_test.json",
        "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tar_cache/only_PostScratachpad_seed3_PostScratachpad_bs12_withlog.txt",
        40)
    res2 = GetResult(
        "/mnt/sdc/zs/ge/transformers/examples/pytorch/translation/bart-base_models/seed2_PostScratachpad_bs12_withlog/checkpoint-6775",
        "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/PostScratachpad_test.json",
        "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tar_cache/only_PostScratachpad_seed3_PostScratachpad_bs12_withlog.txt",
        40)


    # res1 = GetResult("/mnt/sda/zs/ge/transformers/examples/pytorch/translation/sc_models/seed1_AST_100_bs12_withlog/checkpoint-145",
    #           "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/AST_test.json",
    #           "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tar_cache/base_2AST_seed2_AST_bs12_withlog.txt",
    #           40)
    # res2 = GetResult(
    #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/sc_models/seed2_PostScratachpad_100_bs12_withlog/checkpoint-150",
    #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/PostScratachpad_test.json",
    #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tar_cache/only_PostScratachpad_seed3_PostScratachpad_bs12_withlog.txt",
    #     40)
    # res1 = GetResult(
    #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/sc_models/seed2_AST_1000_bs12_withlog/checkpoint-1218",
    #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/AST_test.json",
    #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tar_cache/base_2AST_seed2_AST_bs12_withlog.txt",
    #     40)
    # res2 = GetResult(
    #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/sc_models/seed2_PostScratachpad_1000_bs12_withlog/checkpoint-1008",
    #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/PostScratachpad_test.json",
    #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tar_cache/only_PostScratachpad_seed3_PostScratachpad_bs12_withlog.txt",
    #     40)
    # res2 = GetResult(
    #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/sc_models/only_PostScratachpad_seed3_PostScratachpad_bs12_withlog/checkpoint-5691",
    #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/PostScratachpad_test.json",
    #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tar_cache/only_PostScratachpad_seed3_PostScratachpad_bs12_withlog.txt",
    #     1)
    # res1 = GetResult("/mnt/sda/zs/ge/transformers/examples/pytorch/translation/sc_models/base_2AST_seed2_AST_bs12_withlog/checkpoint-6233",
    #           "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/AST_test.json",
    #           "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tar_cache/base_2AST_seed2_AST_bs12_withlog.txt",
    #           2)

    # res1 = GetResult("/mnt/sda/zs/ge/transformers/examples/pytorch/translation/sc_models/seed2_AST_3000_bs12_withlog/checkpoint-2875",
    #           "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/AST_test.json",
    #           "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tar_cache/base_2AST_seed2_AST_bs12_withlog.txt",
    #           40)
    # res2 = GetResult(
    #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/sc_models/seed1_PostScratachpad_3000_bs12_withlog/checkpoint-3625",
    #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/PostScratachpad_test.json",
    #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tar_cache/only_PostScratachpad_seed3_PostScratachpad_bs12_withlog.txt",
    #     40)
    # res1 = GetResult(
    #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/sc_models/seed3_AST_4500_bs12_withlog/checkpoint-5640",
    #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/AST_test.json",
    #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tar_cache/base_2AST_seed2_AST_bs12_withlog.txt",
    #     40)
    # res2 = GetResult(
    #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/sc_models/seed3_PostScratachpad_4500_bs12_withlog/checkpoint-3008",
    #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/PostScratachpad_test.json",
    #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tar_cache/only_PostScratachpad_seed3_PostScratachpad_bs12_withlog.txt",
    #     40)
    all_freq = []
    for i in range(len(res1)):
        freq = {}
        single_line_res = res1[i] + res2[i]
        for cur in single_line_res:
            if cur == ' ' or cur == '':
                continue
            match_flag = False
            for key in freq.keys():
                if check_equiv(cur, key):
                #if cur == key:
                    freq[key] = freq[key] + 1
                    match_flag = True
                    break
            if not match_flag:
                freq[cur] = 1
        freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        all_freq.append(freq)
        if i % 10 == 0:
            print(i)

    #print(all_freq)
    with open('/mnt/sda/zs/ge/transformers/examples/pytorch/translation/bartbase_80.txt', 'w') as f1:
        #for g in gens:
        for f in all_freq:
            if len(f) == 0:
                f1.write('\n')
            else:
                f1.write(str(f[0][0])+'\n')
    all_freq = [str(f) for f in all_freq]
    with open('/mnt/sda/zs/ge/transformers/examples/pytorch/translation/bartbase_dict_80.txt', 'w') as f2:
        #for g in gens:
        f2.write("\n".join(all_freq))






def checkkb():
    res1 = GetResult(
        #"/mnt/sda/zs/ge/transformers/examples/pytorch/translation/kb_models/2seed2_AST_bs12_withlog/checkpoint-792",
        "/mnt/sdc/zs/ge/transformers/examples/pytorch/translation/kbC_models/80seed2_Complete_AST_bs12_withlog/checkpoint-1196",
              #"/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/kb13/AST_val.json",
                "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/kb13/complete/Complete_AST_val.json",
              "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tar_cache/base_2AST_seed2_AST_bs12_withlog.txt",
              1)
    res2 = GetResult(
        #"/mnt/sda/zs/ge/transformers/examples/pytorch/translation/kb_models/2seed3_PostScratachpad_bs12_withlog/checkpoint-936",
        "/mnt/sdc/zs/ge/transformers/examples/pytorch/translation/kbC_models/80seed1_Complete_PostScratachpad_bs12_withlog/checkpoint-1586",
        "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/kb13/complete/Complete_PostScratachpad_val.json",
        #"/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/kb13/complete/Complete_AST_val.json",
        "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tar_cache/only_PostScratachpad_seed3_PostScratachpad_bs12_withlog.txt",
        1)
    # res1 = GetResult("/mnt/sda/zs/ge/transformers/examples/pytorch/translation/sc_models/base_2AST_seed2_AST_bs12_withlog/checkpoint-6233",
    #           "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/AST_test.json",
    #           "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tar_cache/base_2AST_seed2_AST_bs12_withlog.txt",
    #           5)
    # res1 = GetResult(
    #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/sc_models/only_PostScratachpad_seed3_PostScratachpad_bs12_withlog/checkpoint-5691",
    #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/PostScratachpad_test.json",
    #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tar_cache/only_PostScratachpad_seed3_PostScratachpad_bs12_withlog.txt",
    #     5)

    all_freq = []
    for i in range(len(res1)):
        freq = {}
        single_line_res = res1[i] + res2[i]
        for cur in single_line_res:
            if cur == ' ' or cur == '':
                continue
            match_flag = False
            for key in freq.keys():
                if check_equiv(cur, key):
                #if cur == key:
                    #here1
                    #break
                    freq[key] = freq[key] + 1
                    match_flag = True
                    break
            if not match_flag:
                freq[cur] = 1
        # here2
        freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        #freq = sorted(freq.items())
        #freq = tuple(freq.items())
        all_freq.append(freq)
        if i % 10 == 0:
            print(i)

    #print(all_freq)

    with open('/mnt/sda/zs/ge/transformers/examples/pytorch/translation/kb_2.txt', 'w') as f1:
        #for g in gens:
        for f in all_freq:
            print(f)
            if len(f) == 0 or len(f[0]) == 0:
                f1.write('\n')
            else:
                f1.write(str(f[0][0])+'\n')
    all_freq = [str(f) for f in all_freq]
    with open('/mnt/sda/zs/ge/transformers/examples/pytorch/translation/kb_dict_2.txt', 'w') as f2:
        #for g in gens:
        f2.write("\n".join(all_freq))
#
#     for i in range(len(res1)):
#         res1[i] = re.sub(r'<extra_id_..>', "", res1[i])


def checksynth():
    res1 = GetResult(
        #"/mnt/sda/zs/ge/transformers/examples/pytorch/translation/kb_models/2seed2_AST_bs12_withlog/checkpoint-792",
        "/mnt/sdc/zs/ge/transformers/examples/pytorch/translation/synth_models/seed2_AST_bs12_withlog/checkpoint-4607",
              #"/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/kb13/AST_val.json",
                "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/synth_turk/AST_test.json",
              "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tar_cache/base_2AST_seed2_AST_bs12_withlog.txt",
              40)
    res2 = GetResult(
        #"/mnt/sda/zs/ge/transformers/examples/pytorch/translation/kb_models/2seed3_PostScratachpad_bs12_withlog/checkpoint-936",
       "/mnt/sdc/zs/ge/transformers/examples/pytorch/translation/synth_models/seed1_PostScratachpad_bs12_withlog/checkpoint-5962",
        #"/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/kb13/PostScratachpad_val.json",
        "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/synth_turk/PostScratachpad_test.json",
        "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tar_cache/only_PostScratachpad_seed3_PostScratachpad_bs12_withlog.txt",
        40)
    # res1 = GetResult("/mnt/sda/zs/ge/transformers/examples/pytorch/translation/sc_models/base_2AST_seed2_AST_bs12_withlog/checkpoint-6233",
    #           "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/small.json",
    #           "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tar_cache/base_2AST_seed2_AST_bs12_withlog.txt",
    #           1)
    # res2 = GetResult(
    #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/sc_models/only_PostScratachpad_seed3_PostScratachpad_bs12_withlog/checkpoint-5691",
    #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/small.json",
    #     "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/tar_cache/only_PostScratachpad_seed3_PostScratachpad_bs12_withlog.txt",
    #     1)

    all_freq = []
    for i in range(len(res1)):
        freq = {}
        single_line_res = res1[i] + res2[i]
        for cur in single_line_res:
            if cur == ' ':
                continue
            match_flag = False
            for key in freq.keys():
                if check_equiv(cur, key):
                    freq[key] = freq[key] + 1
                    match_flag = True
                    break
            if not match_flag:
                freq[cur] = 1
        freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        all_freq.append(freq)
        if i % 10 == 0:
            print(i)

    #print(all_freq)
    with open('/mnt/sda/zs/ge/transformers/examples/pytorch/translation/synth_most_freq_80.txt', 'w') as f1:
        #for g in gens:
        for f in all_freq:
            f1.write(str(f[0][0])+'\n')
    all_freq = [str(f) for f in all_freq]
    with open('/mnt/sda/zs/ge/transformers/examples/pytorch/translation/synth_consistency_dict_80.txt', 'w') as f2:
        #for g in gens:
        f2.write("\n".join(all_freq))
#
#     for i in range(len(res1)):
#         res1[i] = re.sub(r'<extra_id_..>', "", res1[i])
#print(norm3("or(repeatatleast(low>,4),concat(vow>,cap>))"))
#checksynth()
#reinforce()
#checkkb()
#check()
print(check_equiv('concat(<m0>,endwith(star(<cap>)))', 'concat(<m0>,star(endwith(<cap>)))'))
print(parse_spec_to_ast('star(endwith(contain(<m0>)))').standard_regex())