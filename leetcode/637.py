# Definition for a binary tree node.
from typing import List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
        
        '''
        parameters:
            root: 【3,9,20，null，null，15,7】
        需要自己构建二叉树
        '''
        # 思路
        # 构建二叉树
        # 遍历二叉树
        # 记录每一层的节点数和节点值之和
        # 计算每一层的平均值
        # 返回结果
        
        # 构建二叉树
        # for i in range(len(root)):
        #     if root[i] is not None:
        #         root[i] = TreeNode(root[i])
        
        if not root:
            return []
        res = []
        queue = [root]
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.pop(0)
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(sum(level) / len(level))
        return res
    

import pytest
# from learn_sl_ai.leetcode import Solution, TreeNode

# 测试空树
def test_averageOfLevels_empty_tree():
    root = None
    solution = Solution()
    assert solution.averageOfLevels(root) == []

# 测试只有一个节点的树
def test_averageOfLevels_single_node_tree():
    root = TreeNode(1)
    solution = Solution()
    assert solution.averageOfLevels(root) == [1.0]

# 测试具有多个节点的树
def test_averageOfLevels_multiple_nodes_tree():
    root = TreeNode(3)
    root.left = TreeNode(9)
    root.right = TreeNode(20)
    root.right.left = TreeNode(15)
    root.right.right = TreeNode(7)
    solution = Solution()
    expected = [3.0, 14.5, 11.0]
    assert solution.averageOfLevels(root) == expected
    
if __name__ == '__main__':
    import os
    pytest.main(["-q", os.path.join(os.path.dirname(__file__),"637.py") ])
