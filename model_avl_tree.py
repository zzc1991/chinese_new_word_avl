import math


class TreeNode:
    '''
    Definition of Balanced Binary Search Tree Nodes
    平衡二叉搜索树节点的定义
    '''

    def __init__(self, val, count=0, rank=0):
        # 节点值
        # node value
        self.val = val
        # 左子树
        # left subtree
        self.left = None
        # 右子树
        # right subtree
        self.right = None
        # 用来判断统计是否完成
        # Used to determine whether the statistics are complete
        self.word_finish = False
        # 用来计数
        # to count
        self.count = count
        # 1为一阶节点 2为2阶节点 3为3阶节点
        # 1 is a first-order node 2 is a second-order node 3 is a third-order node
        self.rank = rank
        # 新增儿子平衡二叉搜索树
        # Add son balanced binary search tree
        self.child = None
        # 记录父亲节点以方便计算信息熵
        # Record the parent node to facilitate the calculation of information entropy
        self.parent = None


class OperationTree:

    def add(self, root, word):
        '''
        Balanced binary tree node added
        平衡二叉树节点新增
        :param root:
        :param word:
        :return:
        '''
        for count, char in enumerate(word):
            found_in_child = False
            # 在节点中找字符
            flag, treenode = self.query_foo(root, char)
            if flag:
                root = treenode.child
                found_in_child = True
            # 顺序在节点后面添加节点。 a->b->c
            if not found_in_child:
                parent = root.parent
                root = self.insert_bal_foo(root, char, rank=count + 1)
                parent.child = root
                flag, parent_node = self.query_foo(root, char)
                parent_node.parent = parent
                if flag:
                    if count + 2 < 4:
                        node = TreeNode(0, rank=count + 2)
                        node.parent = parent_node
                        parent_node.child = node
                if parent_node.child is not None:
                    root = parent_node.child
                else:
                    root = parent_node
            # 判断是否是最后一个节点，这个词每出现一次就+1
            if count == len(word) - 1:
                if not found_in_child:
                    root.count += 1
                    root.word_finish = True
                else:
                    treenode.count += 1
                    treenode.word_finish = True

    def add_suffix(self, root, word):
        '''
        Calculate the left entropy and rearrange the word array.
        In order to solve the conflict of inserting the balanced binary tree,
        the code number of the last word is negative.
        对左熵进行计算，将单词数组重新排列，为了解决插入平衡二叉树的冲突，将最后一个单词的code号取负值
        :param root:
        :param word:
        :return:
        '''
        length = len(word)
        if length == 3:
            word = list(word)
            word[0], word[1], word[2] = word[1], word[2], -word[0]
            for count, char in enumerate(word):
                found_in_child = False
                # 在节点中找字符（不是最后的后缀词）
                flag, treenode = self.query_foo(root, char)
                if flag:
                    root = treenode.child
                    found_in_child = True
                if not found_in_child:
                    parent = root.parent
                    root = self.insert_bal_foo(root, char, rank=count + 1)
                    parent.child = root
                    flag, parent_node = self.query_foo(root, char)
                    parent_node.parent = parent
                    if flag:
                        if count + 2 < 4:
                            node = TreeNode(0, rank=count + 2)
                            node.parent = parent_node
                            parent_node.child = node
                    if parent_node.child is not None:
                        root = parent_node.child
                    else:
                        root = parent_node

                # 判断是否是最后一个节点，这个词每出现一次就+1
                if count == len(word) - 1:
                    if not found_in_child:
                        root.count += 1
                        root.isback = True
                        root.word_finish = True
                    else:
                        treenode.count += 1
                        treenode.isback = True
                        treenode.word_finish = True

    '''二叉搜索树操作'''

    def search_one(self, root):
        """
        Calculate mutual information:
        find first-order co-occurrences, and return word probabilities
        计算互信息: 寻找一阶共现，并返回词概率
        :return:
        """
        one_dict = {}
        total_one = 0
        if root is not None:
            total_one = self.one_count_foo(root)
            one_dict = self.one_count_dict_foo(root, total_one)

        return one_dict, total_one

    def search_bi(self, root, PMI_limit):
        """
        Calculate mutual information:
        find second-order co-occurrences and return log2( P(X,Y) / (P(X) * P(Y)) and word probability
        计算互信息: 寻找二阶共现，并返回 log2( P(X,Y) / (P(X) * P(Y)) 和词概率
        :return:
        """
        # 1 grem The proportion of each word, and the total number of 1 grem
        # 1 grem 各词的占比，和 1 grem 的总次数
        one_dict, total_one = self.search_one(root)
        total = self.bi_count_foo(root)
        result = self.bi_count_dict_foo(root, one_dict, total, PMI_limit)

        return result

    def search_left(self, root):
        """
        Find the left frequency,
        count the left entropy, and return the left entropy (bc - a this is abc|bc so it is the left entropy)
        寻找左频次,统计左熵， 并返回左熵 (bc - a 这个算的是 abc|bc 所以是左熵)
        :return:
        """

        total = self.left_count_foo(root)
        result = self.left_count_dict_foo(root, total)
        for item in result:
            result[item] = -result[item]
        return result

    def search_right(self, root):
        """
        Find the right frequency,
        count the right entropy, and return the right entropy (ab - c is abc|ab so it is the right entropy)
        寻找右频次,统计右熵，并返回右熵 (ab - c 这个算的是 abc|ab 所以是右熵)
        :return:
        """
        total = self.right_count_foo(root)
        result = self.right_count_dict_foo(root, total)
        for item in result:
            result[item] = -result[item]
        return result

    def find_word_tfidf(self, root, number_word,idf_list, N=0):
        '''
        Calculate the new word and output it, convert it to Chinese characters using the number_word list
        计算新词并输出，用number_word列表将其转换成中文字符
        :param root:
        :param number_word:
        :param N:
        :return:
        '''
        # Get mutual information by searching
        # 通过搜索得到互信息
        bi = self.search_bi(root, 0)
        # 通过搜索得到左右熵
        left = self.search_left(root)
        right = self.search_right(root)
        result = {}
        for item in bi:
            values = bi[item]
            if item in left.keys() and item in right.keys():
                word_list = item.split('_')
                # Calculate left and right information entropy with tf-idf
                # 计算带有tf-idf的左右信息熵
                result[number_word[int(word_list[0])] + '_' + number_word[int(word_list[1])]] = (values[0] + min(
                    left[item], right[item])) * values[1]*(idf_list[int(word_list[0])]+idf_list[int(word_list[1])])
        # Arrange in reverse order from largest to smallest, the larger the value, the greater the probability of being a compound word
        # 按照 大到小倒序排列，value 值越大，说明是组合词的概率越大
        result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        dict_list = [result[0][0]]
        add_word = {}
        new_word = "".join(dict_list[0].split('_'))
        # get probability
        # 获得概率
        add_word[new_word] = result[0][1]
        # Take the first N
        # 取前N个
        for d in result[1: N]:
            flag = True
            for tmp in dict_list:
                pre = tmp.split('_')[0]
                # The new word suffix appears in the prefix of the old word or if a new word is found, it appears in the list; then jump out of the loop
                # 新出现单词后缀，再老词的前缀中 or 如果发现新词，出现在列表中; 则跳出循环
                if d[0].split('_')[-1] == pre or "".join(tmp.split('_')) in "".join(d[0].split('_')):
                    flag = False
                    break
            if flag:
                new_word = "".join(d[0].split('_'))
                add_word[new_word] = d[1]
                dict_list.append(d[0])

        return result, add_word

    def query_foo(self, root, val):
        '''
        Iterative version of balanced binary tree query with log2n complexity
        平衡二叉树查询的迭代版本，复杂度为log2n
        :param root:
        :param val:
        :return:
        '''
        if root is None:
            return False, None
        res_root = None
        while root is not None:
            # get the top element of the stack
            # 取栈顶元素
            if val < root.val:
                root = root.left
            elif val > root.val:
                root = root.right
            else:
                res_root = root
                break
        if res_root is None:
            return False, None
        else:
            return True, res_root

    def insert_bal_foo(self, root, val, count=0, rank=0):
        '''
        Recursive version of balanced binary tree insertion process with log2n complexity
        平衡二叉树插入过程的递归版本，复杂度为log2n
        :param root:
        :param val:
        :param count:
        :param rank:
        :return:
        '''
        if root is None:
            root = TreeNode(val, count, rank)
            return root
        else:
            if val < root.val:
                root.left = self.insert_bal_foo(root.left, val, count, rank)
            else:
                root.right = self.insert_bal_foo(root.right, val, count, rank)
            if self.depth(root.left) - self.depth(root.right) > 1:
                # LL type, that is, the inserted value is on the left side of the left side of the a node
                # LL型,即插入的数字在a节点的左边的左边
                if val < root.left.val:
                    root = self.LL(root)
                # LR type, that is, the inserted value is on the left and right of the a node
                # LR型,即插入的数字在a节点的左边的右边
                else:
                    root = self.LR(root)
            elif self.depth(root.right) - self.depth(root.left) > 1:
                # RR type, that is, the inserted value is on the right side of the right side of the a node
                # RR型,即插入的数字在a节点的右边的右边
                if val > root.right.val:
                    root = self.RR(root)
                # RL type, that is, the inserted number is on the left side of the right side of the a node
                # RL型,即插入的数字在a节点的右边的左边
                else:
                    root = self.RL(root)
        return root

    def one_count_foo(self, root):
        '''
        Calculate the total count of nodes of order 1
        计算1阶节点的总count
        :param root:
        :return:
        '''
        if not root:
            return
        res, stack = [], [root]
        while stack:
            # Get the top element of the stack
            # 取栈顶元素
            s = stack.pop()
            if s:
                # Due to the FIFO feature of the stack,
                # put the right child first and then the left child
                # 由于栈的先进后出特性 先放入右孩子 再放入左孩子
                stack.append(s.right)
                stack.append(s.left)
                res.append(s.count)
        return sum(res)

    def one_count_dict_foo(self, root, total):
        '''
        Normalize the 1st-order node frequency
        对1阶节点频度进行归一化处理
        :param root:
        :param total:
        :return:
        '''
        if not root:
            return
        res, stack = {}, [root]
        while stack:
            # Get the top element of the stack
            # 取栈顶元素
            s = stack.pop()
            if s:
                # Due to the FIFO feature of the stack,
                # put the right child first and then the left child
                # 由于栈的先进后出特性 先放入右孩子 再放入左孩子
                stack.append(s.right)
                stack.append(s.left)
                res[s.val] = s.count / total
        return (res)

    def bi_count_foo(self, root):
        '''
        Calculate the total count of nodes of order 2
        计算2阶节点的总count
        :param root:
        :return:
        '''
        if not root:
            return
        res, stack = [], [root]
        while stack:
            # Get the top element of the stack
            # 取栈顶元素
            s = stack.pop()
            if s:
                # If it is a 1st-order node, take the child node
                # 如果是1阶节点取子节点
                if s.rank == 1:
                    stack.append(s.child)
                stack.append(s.right)
                stack.append(s.left)
                # If it is a node of order 2 and is not the one used to build a balanced binary tree
                # 如果为2阶节点 且不是用于构建平衡二叉树的那个节点
                if s.val != 0 and s.rank == 2:
                    res.append(s.count)
        return sum(res)

    def bi_count_dict_foo(self, root, one_dict, total, PMI_limit):
        '''
        Normalize the 2st-order node frequency
        对2阶节点频度进行归一化处理
        :param root:
        :param one_dict:
        :param total:
        :param PMI_limit:
        :return:
        '''
        if not root:
            return
        res, stack = {}, [root]
        while stack:
            # 取栈顶元素
            s = stack.pop()
            if s:
                # 由于栈的先进后出特性 先放入右孩子 再放入左孩子
                if s.rank == 1:
                    stack.append(s.child)
                stack.append(s.right)
                stack.append(s.left)
                if s.val != 0 and s.rank == 2 and s.word_finish:
                    PMI = math.log(max(s.count, 1), 2) - math.log(total, 2) - math.log(one_dict[s.parent.val],
                                                                                       2) - math.log(
                        one_dict[s.val],
                        2)
                    # 这里做了PMI阈值约束
                    if PMI > PMI_limit:
                        res[str(s.parent.val) + '_' + str(s.val)] = (PMI, s.count / total)

        return res

    def left_count_foo(self, root):
        '''
        Calculate total left entropy
        计算左熵总数
        :param root:
        :return:
        '''
        if not root:
            return
        res, stack = [], [root]
        while stack:
            s = stack.pop()
            if s:
                stack.append(s.child)
                stack.append(s.right)
                stack.append(s.left)
                if s.rank == 3 and s.val < 0 and s.word_finish:
                    res.append(s.count)
        return sum(res)

    def left_count_dict_foo(self, root, total):
        '''
        Calculate the left entropy for each combination
        计算每个组合的左熵
        :param root:
        :param total:
        :return:
        '''
        if not root:
            return

        if not root:
            return
        res, stack = {}, [root]
        while stack:
            s = stack.pop()
            if s:
                stack.append(s.child)
                stack.append(s.right)
                stack.append(s.left)
                if s.rank == 3 and s.val < 0 and s.word_finish:
                    key_word = str(s.parent.parent.val) + '_' + str(s.parent.val)
                    if key_word in res.keys():
                        p = res[key_word]
                        p += (s.count / total) * math.log(s.count / total, 2)
                        res[key_word] = p
                    else:
                        res[key_word] = (s.count / total) * math.log(s.count / total, 2)
        return res

    def right_count_foo(self, root):
        '''
        Calculate the total right entropy
        计算右熵总数
        :param root:
        :return:
        '''
        if not root:
            return
        res, stack = [], [root]
        while stack:
            s = stack.pop()
            if s:
                stack.append(s.right)
                stack.append(s.left)
                stack.append(s.child)
                if s.rank == 3 and s.val > 0 and s.word_finish:
                    res.append(s.count)
        return sum(res)

    def right_count_dict_foo(self, root, total):
        '''
        Calculate the right entropy for each combination
        计算每个组合的右熵
        :param root:
        :param total:
        :return:
        '''
        if not root:
            return

        if not root:
            return
        res,stack = {},[root]
        while stack:
            # 取栈顶元素
            s = stack.pop()
            if s:
                # 由于栈的先进后出特性 先放入右孩子 再放入左孩子
                stack.append(s.child)
                stack.append(s.right)
                stack.append(s.left)
                if s.rank == 3 and s.val > 0 and s.word_finish:
                    key_word = str(s.parent.parent.val) + '_' + str(s.parent.val)
                    if key_word in res.keys():
                        p = res[key_word]
                        p += (s.count / total) * math.log(s.count / total, 2)
                        res[key_word] = p
                    else:
                        res[key_word] = (s.count / total) * math.log(s.count / total, 2)
        return res

    def depth(self, root_node):
        '''
        Returns the node depth
        返回节点深度
        :param root_node:
        :return:
        '''
        # Define stack, save the current node and its depth
        # 定义stack，保存当前节点及其深度
        stack = []
        if root_node:
            # Push the root node and its depth 1
            # 入栈根节点及其深度1
            stack.append((root_node, 1))
        # 返回值，最大深度
        max_depth = 0
        while stack:
            # Traverse, pop the current node and its depth
            # 遍历，弹出当前节点及其深度
            tree_node, cur_depth = stack.pop()
            # 节点为空就遍历其他节点
            if tree_node:
                # If the node is empty, traverse other nodes
                # 判断是否更新节点深度最大值
                max_depth = max(max_depth, cur_depth)
                # Add left and right nodes and their depth
                # 加入左右节点及其深度
                stack.append((tree_node.left, cur_depth + 1))
                stack.append((tree_node.right, cur_depth + 1))

        return max_depth

    def LL(self, node):
        '''
        left-left rotation operation
        left-left 型旋转操作
        :param node:
        :return:
        '''
        temp = node.left.right
        temp1 = node.left
        temp1.right = node
        node.left = temp
        return temp1

    def RR(self, node):
        '''
        right-right rotation operation
        right-right 型旋转操作
        :param node:
        :return:
        '''
        temp = node.right.left
        temp1 = node.right
        temp1.left = node
        node.right = temp
        return temp1

    def LR(self, node):
        '''
        left-right rotation operation
        left-right 型旋转操作
        :param node:
        :return:
        '''
        node.left = self.RR(node.left)
        return self.LL(node)

    def RL(self, node):
        '''
        right-left rotation operation
        right-left 型旋转操作
        :param node:
        :return:
        '''
        node.right = self.LL(node.right)
        return self.RR(node)
