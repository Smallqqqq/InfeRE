# This is a sample Python script.
from eval import check_equiv, check_io_consistency
from streg_utils import parse_spec_to_ast
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
'''
import network as nx
import matplotlib.pyplot as plt

def create_graph(G, node, pos={}, x=0, y=0, layer=1):
    pos[node.key] = (x, y)
    if len(node.child) == 3:
            G.add_edge(node.key, node.child[0].key)
            l_x, l_y = x - 1 / 3 ** layer, y - 1
            l_layer = layer + 1
            create_graph(G, node.child[0], x=l_x, y=l_y, pos=pos, layer=l_layer)

            G.add_edge(node.key, node.child[1].key)
            r_x, r_y = x, y - 1
            r_layer = layer + 1
            create_graph(G, node.child[1], x=r_x, y=r_y, pos=pos, layer=r_layer)

            G.add_edge(node.key, node.child[2].key)
            r_x, r_y = x + 1 / 3 ** layer, y - 1
            r_layer = layer + 1
            create_graph(G, node.child[2], x=r_x, y=r_y, pos=pos, layer=r_layer)
    else:
        if len(node.child) > 0:
            G.add_edge(node.key, node.child[0].key)
            l_x, l_y = x - 1 / 2 ** layer, y - 1
            l_layer = layer + 1
            create_graph(G, node.child[0], x=l_x, y=l_y, pos=pos, layer=l_layer)
        if len(node.child) > 1:
            G.add_edge(node.key, node.child[1].key)
            r_x, r_y = x + 1 / 2 ** layer, y - 1
            r_layer = layer + 1
            create_graph(G, node.child[1], x=r_x, y=r_y, pos=pos, layer=r_layer)
    return (G, pos)

def draw(node):   # 以某个节点为根画图
    graph = nx.DiGraph()
    graph, pos = create_graph(graph, node)
    fig, ax = plt.subplots(figsize=(8, 10))  # 比例可以根据树的深度适当调节
    nx.draw_networkx(graph, pos, ax=ax, node_size=300)
    plt.show()
'''
setMulti = {
    'and',
    'or',
    'concat',
}
opDict ={
    'optional':1,
    'endwith':1,
    'startwith':1,
    'not':1,
    'repeatatleast':2,
    'contain':1,
    'repeat':2,
    'star':1,
    'repeatrange':3,

}





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

    def postorderVisit(self):
        post = []
        # print(self.key)
        for i in range(len(self.child)):
            post.extend(self.child[i].preorderVisit())
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


def process_AST(src_file, tar_file):
    key_set = set()
    with open(src_file, "r") as f1, open(tar_file, "w") as f2:
        # Read next line
        i = 0
        line = f1.readline()
        # check line is not empty
        while line:
            if i != 0:
                f2.write('\n')
            i = i + 1
            if line == "":
                print("error line")
                print(i)
                return
            # process regex

            regex = line.replace('\n', '')
            regex = regex.replace('notcc', 'not')
            '''
            regex = regex.replace('(', ' ( ')
            regex = regex.replace(')', ' ) ')
            regex = regex.replace(',', ' , ')
            if '   ' in regex:
                print('error blank')
                return
            else:
                regex = ' '.join(regex.split())
            '''
            regex = regex.split(" ")

            for k in range(len(regex)):
                key_set.add(regex[k])
            # print(regex)
            # generate Tree
            _Regextree = build_tree(regex)
            _Regextree.simplify()
            # draw(RegexTree)
            track = _Regextree.preorderVisit()
            track = ' '.join(track)
            # print("track: ",track)
            # f2.write(track)
            # print(RegexTree.child[0])
            #

            line = f1.readline()
            if i % 500 == 0:
                print(i)
    #print("set:", key_set)


def post2Original(post):
    stack = []
    def compress_stack_pre():
        pass

    for i in range(0, len(post)):
        op = post[i]
        if (not opDict.__contains__(op)) and (not setMulti.__contains__(op)):
            stack.append(op)
        else:
            compress_stack_pre()
            stack.append(op)



def pre2Original(pre):
    stack = []
    res = ""
    for i in range(0, len(pre)):
        op = pre[i]
        if opDict.__contains__(op):
            stack.append(opDict[op])
            res = res + op + ' ( '
        elif setMulti.__contains__(op):
            stack.append(-3) #mutli
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
            if i != len(pre)-1:
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
                    while stack[-1] == 0:
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
    if res[len(res)-1] == ' ':
        res = res[0:len(res)-1]
    #print('res',res)
    return res
    # for i in range

def multipre2Original(pre):

    pre = pre2Original(pre)
    pre = pre.split(" ")
    pre = build_tree(pre)
    pre.binarify()
    pre = pre.preorderVisit()
    pre = pre2Original(pre)
    return pre


def aaa():
    with open("/mnt/sda/zs/easy_eval/targ-train.txt", "r") as f1:
        # Read next line
        i = 0
        line = f1.readline()
        # check line is not empty
        while line:
            #print(i)
            i = i + 1
            if line == "":
                print("error line")
                print(i)
                return
            # process regex
            regex = line.replace('\n', '')
            regex = regex.replace('notcc', 'not')
            tmp = regex
            regex = regex.split(" ")
            # print(regex)
            aRegexTree = build_tree(regex)
            aRegexTree.simplify()
            pret = aRegexTree.preorderVisit()
            back = multipre2Original(pret)
            if not check_equiv(tmp,back):
                print('not_equal')
                print(tmp)
                print(pret)
                print(back)
                print(i)
            line = f1.readline()
            if i % 20 == 0:
                print(i)
        print("yeeeeee")



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # #regex = 'concat ( repeatrange ( <let> , 3 , 6 ) , optional ( repeat ( const0 , 4 ) ) )'
    # regex= 'and ( not ( contain ( <spec> ) ) , and ( startwith ( repeat ( <cap> , 3 ) ) , not ( startwith ( repeat ( <H> , 3 ) ) ) ) )'
    # #regex = 'repeatrange ( <let> , 4 , 3 )'
    # regex = regex.split(" ")
    # #print(regex)
    # RegexTree = build_tree(regex)
    # RegexTree.preorderVisit(0,0)
    # #print(RegexTree.child[0])
    # draw(RegexTree)
    aaa()

    '''
    regex = "concat ( repeatrange ( const0 , 1 , 3 ) , concat ( or ( <a> , const1 ) , optional ( repeatrange ( const0 , 1 , 3 ) ) ) )"
    print(regex)
    regex = regex.split(" ")
    # print(regex)
    aRegexTree = build_tree(regex)
    aRegexTree.simplify()
    pret = aRegexTree.preorderVisit()
    print(pret)
    print(multipre2Original(pret))
    '''

    # draw(RegexTree)
    '''
    process_AST(
        "C:\\Users\\s1mpleQ\\Desktop\\ZS\\regex\\StructuredRegex-master\\code\\datasets\\StReg\\targ-train.txt",
        "C:\\Users\\s1mpleQ\\Desktop\\ZS\\regex\\StructuredRegex-master\\code\\datasets\\StReg\\Tree\\StrPreMultiTree_train.txt")
    process_AST(
        "C:\\Users\\s1mpleQ\\Desktop\\ZS\\regex\\StructuredRegex-master\\code\\datasets\\StReg\\targ-val.txt",
        "C:\\Users\\s1mpleQ\\Desktop\\ZS\\regex\\StructuredRegex-master\\code\\datasets\\StReg\\Tree\\StrPreMultiTree_val.txt")
    process_AST(
        "C:\\Users\\s1mpleQ\\Desktop\\ZS\\regex\\StructuredRegex-master\\code\\datasets\\StReg\\targ-testi.txt",
        "C:\\Users\\s1mpleQ\\Desktop\\ZS\\regex\\StructuredRegex-master\\code\\datasets\\StReg\\Tree\\StrPreMultiTree_testi.txt")
    process_AST(
        "C:\\Users\\s1mpleQ\\Desktop\\ZS\\regex\\StructuredRegex-master\\code\\datasets\\StReg\\targ-teste.txt",
        "C:\\Users\\s1mpleQ\\Desktop\\ZS\\regex\\StructuredRegex-master\\code\\datasets\\StReg\\Tree\\StrPreMultiTree_teste.txt")
    '''
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
