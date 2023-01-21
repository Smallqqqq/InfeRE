from easy_eval.eval import check_equiv, check_io_consistency
from easy_eval.streg_utils import parse_spec_to_ast

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
        if len(item) <= idx + 1:
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
        if len(item) <= idx + 1:
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


def eval_Scratachpad(regex):
    regex = regex.replace('<,>','<tmpToken>')
    tmp = regex.split(' ')
    for j in range(len(tmp)):
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

    while res.find('result') != -1:

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
    # print(res)
    res = res.replace("=", '')
    res = res.replace('<tmpToken>','<,>')
    return res


def norm3(regex):
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


def GetDiff(file1, file2):
    ori = []
    gen = []

    with open(file1, "r") as f1, open(file2, "r") as f2:
        lineA = f1.readline()
        lineB = f2.readline()
        # check line is not empty
        while lineA and lineB:
            ori.append(lineA)
            gen.append(lineB)
            lineA = f1.readline()
            lineB = f2.readline()

    print(len(ori))
    print(len(gen))
    if (len(lineA) != len(lineB)):
        print('error')
        return
    num = 0
    with open("/mnt/sda/zs/ge/transformers/examples/pytorch/translation/kb_models/diffScratachpa.txt", "w") as f4:
        for idx in range(len(ori)):

            #ori[idx] = ori[idx].replace(' <STEP> ', ' ')
            #gen[idx] = gen[idx].replace(' <STEP> ', ' ')
            #ori[idx] = ori[idx].replace('=', '')
            ori[idx] = ori[idx].replace('\n', '')
            gen[idx] = gen[idx].replace('\n', '')
            #gen[idx] = gen[idx].replace('notcc', 'not')
            #tmp = ori[idx]
            # print(ori[idx])
            # print(gen[idx])
            #ori[idx] = eval_Scratachpad(ori[idx])
            #gen[idx] = eval_Scratachpad(gen[idx])
            #tmp2 = ori[idx]
            #ori[idx] = norm3(ori[idx])
            #gen[idx] = norm3(gen[idx])

            if (check_equiv(ori[idx], gen[idx])):
            #if ori[idx] == gen[idx]:
                num += 1
            else:
                print(ori[idx])
                print(gen[idx])
                pass

                #print('wrong')
                #if (ori[idx] == ""):
                #    print(tmp)
                #    # print(eval_Scratachpad(tmp))
                # print(tmp)
                # print(tmp2)
                #print(ori[idx])
                #print(gen[idx])

                #f4.write(str(idx) + ' ')
                #f4.write(str(ori[idx]) + ' ')
                #f4.write(str(gen[idx]) + ' ')
                #f4.write('\n')
            if idx % 200 == 0:
                print(idx)
                print(num)
    print(num / len(ori))



def GetDiff2(file1, file2, k):
    ori = []
    gen = []

    with open(file1, "r") as f1, open(file2, "r") as f2:
        lineA = f1.readline()
        lineB = f2.readline()
        # check line is not empty
        while lineA and lineB:
            ori.append(lineA)
            gen.append(lineB)
            lineA = f1.readline()
            lineB = f2.readline()

    for i in range(len(ori)):
        x = ori[i][1:-1]
        x = x.split('\',')
        print(x)
        print('_________')
        tmp_list = []
        for j in range(len(x)):
            tmpx = x[j][x[j].find('(\'')+2:]
            tmp_list.append(tmpx)
        print(tmp_list)
        ori[i] = tmp_list

    print(len(ori))
    print(len(gen))
    if (len(lineA) != len(lineB)):
        print('error')
        return
    num = 0
    with open("/mnt/sda/zs/ge/transformers/examples/pytorch/translation/kb_models/diffScratachpa.txt", "w") as f4:
        for idx in range(len(ori)):

            #ori[idx] = ori[idx].replace(' <STEP> ', ' ')
            #gen[idx] = gen[idx].replace(' ', '')

            gen[idx] = gen[idx].replace('\n', '')

            #gen[idx] = gen[idx].replace('notcc', 'not')
            #tmp = ori[idx]
            # print(ori[idx])
            # print(gen[idx])
            #ori[idx] = eval_Scratachpad(ori[idx])
            #gen[idx] = eval_Scratachpad(gen[idx])
            #tmp2 = ori[idx]
            #ori[idx] = norm3(ori[idx])
            #gen[idx] = norm3(gen[idx])

            print(ori[idx])
            print(gen[idx])
            for ii in range(min(len(ori[idx]),k)):
                ori[idx][ii] = ori[idx][ii].replace('\n', '')
                ori[idx][ii] = ori[idx][ii].replace('=', '')
                print(ori[idx][ii])
                #if (check_equiv(ori[idx][ii], gen[idx])):
                if ori[idx][ii] == gen[idx]:
                    num += 1
                    break
                else:
                    pass

                #print('wrong')
                #if (ori[idx] == ""):
                #    print(tmp)
                #    # print(eval_Scratachpad(tmp))
                # print(tmp)
                # print(tmp2)
                #print(ori[idx])
                #print(gen[idx])

                #f4.write(str(idx) + ' ')
                #f4.write(str(ori[idx]) + ' ')
                #f4.write(str(gen[idx]) + ' ')
                #f4.write('\n')
            if idx % 200 == 0:
                print(idx)
                print(num)
    print(num / len(ori))


def GetDiff3(file1, file2, k):
    ori = []
    gen = []

    with open(file1, "r") as f1, open(file2, "r") as f2:
        lineA = f1.readline()
        lineB = f2.readline()
        # check line is not empty
        while lineA and lineB:
            ori.append(lineA)
            gen.append(lineB)
            lineA = f1.readline()
            lineB = f2.readline()

    for i in range(len(ori)):
        x = ori[i][1:-1]
        x = x.split('\',')
        print(x)
        print('_________')
        tmp_list = []
        for j in range(len(x)):
            tmpx = x[j][x[j].find('(\'')+2:]
            tmp_list.append(tmpx)
        print(tmp_list)
        ori[i] = tmp_list

    print(len(ori))
    print(len(gen))
    if (len(lineA) != len(lineB)):
        print('error')
        return
    num = 0
    with open("/mnt/sda/zs/ge/transformers/examples/pytorch/translation/kb_models/diffScratachpa.txt", "w") as f4:
        for idx in range(len(ori)):

            #ori[idx] = ori[idx].replace(' <STEP> ', ' ')
            gen[idx] = gen[idx].replace(' <STEP> ', ' ')

            gen[idx] = gen[idx].replace('\n', '')

            #gen[idx] = gen[idx].replace('notcc', 'not')
            #tmp = ori[idx]
            # print(ori[idx])
            # print(gen[idx])
            #ori[idx] = eval_Scratachpad(ori[idx])
            gen[idx] = eval_Scratachpad(gen[idx])
            #tmp2 = ori[idx]
            #ori[idx] = norm3(ori[idx])
            #gen[idx] = norm3(gen[idx])

            # print(ori[idx])
            print(gen[idx])
            for ii in range(min(len(ori[idx]),k)):
                print(ori[idx])
                ori[idx][ii] = ori[idx][ii].replace('\n', '')
                ori[idx][ii] = ori[idx][ii].replace('=', '')
                if (check_equiv(ori[idx][ii], gen[idx])):
                #if ori[idx][ii] == gen[idx]:
                    num += 1
                    break
                else:
                    pass

                #print('wrong')
                #if (ori[idx] == ""):
                #    print(tmp)
                #    # print(eval_Scratachpad(tmp))
                # print(tmp)
                # print(tmp2)
                #print(ori[idx])
                #print(gen[idx])

                #f4.write(str(idx) + ' ')
                #f4.write(str(ori[idx]) + ' ')
                #f4.write(str(gen[idx]) + ' ')
                #f4.write('\n')
            if idx % 200 == 0:
                print(idx)
                print(num)
    print(num / len(ori))



def GetDiff4(file1, file2, file3, k):
    ori = []
    gen = []

    with open(file1, "r") as f1, open(file2, "r") as f2:
        lineA = f1.readline()
        lineB = f2.readline()
        # check line is not empty
        while lineA and lineB:
            ori.append(lineA)
            gen.append(lineB)
            lineA = f1.readline()
            lineB = f2.readline()

    for i in range(len(ori)):
        x = ori[i][1:-1]
        x = x.split('\',')
        print(x)
        print('_________')
        tmp_list = []
        for j in range(len(x)):
            tmpx = x[j][x[j].find('(\'')+2:]
            tmp_list.append(tmpx)
        print(tmp_list)
        ori[i] = tmp_list

    print(len(ori))
    print(len(gen))
    if (len(lineA) != len(lineB)):
        print('error')
        return
    num = 0

    jj = 0
    with open(file3, "r") as f3:
        lineA = f3.readline()
        # check line is not empty
        while lineA:
            lineA = lineA.replace('\n', '')
            lineA = eval_Scratachpad(lineA)
            lineA = norm3(lineA)
            print(lineA)
            ori[jj].insert(0,lineA)
            jj = jj + 1
            lineA = f3.readline()


    with open("/mnt/sda/zs/ge/transformers/examples/pytorch/translation/kb_models/diffScratachpa.txt", "w") as f4:
        for idx in range(len(ori)):

            #ori[idx] = ori[idx].replace(' <STEP> ', ' ')
            gen[idx] = gen[idx].replace(' <STEP> ', ' ')

            gen[idx] = gen[idx].replace('\n', '')

            #gen[idx] = gen[idx].replace('notcc', 'not')
            #tmp = ori[idx]
            # print(ori[idx])
            # print(gen[idx])
            #ori[idx] = eval_Scratachpad(ori[idx])
            gen[idx] = eval_Scratachpad(gen[idx])
            #tmp2 = ori[idx]
            #ori[idx] = norm3(ori[idx])
            #gen[idx] = norm3(gen[idx])

            # print(ori[idx])
            print(gen[idx])
            for ii in range(min(len(ori[idx]),k)):
                print(ori[idx])
                ori[idx][ii] = ori[idx][ii].replace('\n', '')
                ori[idx][ii] = ori[idx][ii].replace('=', '')
                if (check_equiv(ori[idx][ii], gen[idx])):
                #if ori[idx][ii] == gen[idx]:
                    num += 1
                    break
                else:
                    pass

                #print('wrong')
                #if (ori[idx] == ""):
                #    print(tmp)
                #    # print(eval_Scratachpad(tmp))
                # print(tmp)
                # print(tmp2)
                #print(ori[idx])
                #print(gen[idx])

                #f4.write(str(idx) + ' ')
                #f4.write(str(ori[idx]) + ' ')
                #f4.write(str(gen[idx]) + ' ')
                #f4.write('\n')
            if idx % 200 == 0:
                print(idx)
                print(num)
    print(num / len(ori))



def GetDiff5(file1, file2, file3, k):
    ori = []
    gen = []

    with open(file1, "r") as f1, open(file2, "r") as f2:
        lineA = f1.readline()
        lineB = f2.readline()
        # check line is not empty
        while lineA and lineB:
            ori.append(lineA)
            gen.append(lineB)
            lineA = f1.readline()
            lineB = f2.readline()

    for i in range(len(ori)):
        x = ori[i][1:-1]
        x = x.split('\',')
        print(x)
        print('_________')
        tmp_list = []
        for j in range(len(x)):
            tmpx = x[j][x[j].find('(\'')+2:]
            tmp_list.append(tmpx)
        print(tmp_list)
        ori[i] = tmp_list

    print(len(ori))
    print(len(gen))
    if (len(lineA) != len(lineB)):
        print('error')
        return
    num = 0

    jj = 0
    with open(file3, "r") as f3:
        lineA = f3.readline()
        # check line is not empty
        while lineA:
            lineA = lineA.replace('\n', '')
            # lineA = eval_Scratachpad(lineA)
            lineA = norm3(lineA)
            print(lineA)
            ori[jj].insert(0,lineA)
            jj = jj + 1
            lineA = f3.readline()

    with open("/mnt/sda/zs/ge/transformers/examples/pytorch/translation/kb_models/diffScratachpa.txt", "w") as f4:
        for idx in range(len(ori)):

            #ori[idx] = ori[idx].replace(' <STEP> ', ' ')
            #gen[idx] = gen[idx].replace(' ', '')

            gen[idx] = gen[idx].replace('\n', '')

            #gen[idx] = gen[idx].replace('notcc', 'not')
            #tmp = ori[idx]
            # print(ori[idx])
            # print(gen[idx])
            #ori[idx] = eval_Scratachpad(ori[idx])
            #gen[idx] = eval_Scratachpad(gen[idx])
            #tmp2 = ori[idx]
            #ori[idx] = norm3(ori[idx])
            #gen[idx] = norm3(gen[idx])

            print(ori[idx])
            print(gen[idx])
            for ii in range(min(len(ori[idx]),k)):
                ori[idx][ii] = ori[idx][ii].replace('\n', '')
                ori[idx][ii] = ori[idx][ii].replace('=', '')
                print(ori[idx][ii])
                if (check_equiv(ori[idx][ii], gen[idx])):
                #if ori[idx][ii] == gen[idx]:
                    num += 1
                    break
                else:
                    pass

                #print('wrong')
                #if (ori[idx] == ""):
                #    print(tmp)
                #    # print(eval_Scratachpad(tmp))
                # print(tmp)
                # print(tmp2)
                #print(ori[idx])
                #print(gen[idx])

                #f4.write(str(idx) + ' ')
                #f4.write(str(ori[idx]) + ' ')
                #f4.write(str(gen[idx]) + ' ')
                #f4.write('\n')
            if idx % 200 == 0:
                print(idx)
                print(num)
    print(num / len(ori))
# print(eval_Scratachpad("result1=<m0>"))
#print(check_equiv('contain(concat(star(<let>),<vow>))', 'contain(repeatatleast(<vow>,1))'))
#print(parse_spec_to_ast('contain(concat(<M0>,concat(startwith(<let>),<M1>)))').standard_regex())
#GetDiff("/mnt/sda/zs/ge/transformers/examples/pytorch/translation/completeval_kb_freq.txt",
#       "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/Complete_AST_val.txt")
# GetDiff("/mnt/sda/zs/ge/transformers/examples/pytorch/translation/bartsmall_80.txt",
#          "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/AST_test.txt")
#GetDiff("/mnt/sda/zs/ge/transformers/examples/pytorch/translation/sc_models/base_2AST_seed2_AST_bs12_withlog/checkpoint-6233/res2.txt",
#        "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/AST_test.txt")
#GetDiff("/mnt/sda/zs/ge/transformers/examples/pytorch/translation/ps_kb_2.txt",
#      "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/Complete_PostScratachpad_val.txt")

GetDiff3("/mnt/sda/zs/ge/transformers/examples/pytorch/translation/t5small_dict_80.txt",
#      "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/AST_test.txt",

      "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/PostScratachpad_test.txt",
      5)

# GetDiff4("/mnt/sda/zs/ge/transformers/examples/pytorch/translation/2ps_kb_dict_5.txt",
# #      "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/AST_test.txt",
#
#       "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/Complete_PostScratachpad_val.txt",
#          "/mnt/sdc/zs/ge/transformers/examples/pytorch/translation/kbC_models/80seed1_Complete_PostScratachpad_bs12_withlog/checkpoint-1586/generated_predictions.txt",
#       5)

# GetDiff5("/mnt/sda/zs/ge/transformers/examples/pytorch/translation/2ast_kb_dict_5.txt",
# #      "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/AST_test.txt",
#
#       "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/Complete_AST_val.txt",
#          "/mnt/sdc/zs/ge/transformers/examples/pytorch/translation/kbC_models/80seed2_Complete_AST_bs12_withlog/checkpoint-1196/generated_predictions.txt",
#       5)

#GetDiff("/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/str/StrPostScratachpad_teste.txt",
#       "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/data/str/StrAST_teste.txt")
# GetDiff("/mnt/sda/zs/ge/transformers/examples/pytorch/translation/kb_models/generated_predictions.txt",
#         "/mnt/sda/zs/ge/transformers/examples/pytorch/translation/kb_models/PostScratachpad_val.txt")