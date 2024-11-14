# Definition for a binary tree node.
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True
        def dfs(node, lower, upper):
            if not node:
                return True
            if node.val <= lower or node.val >= upper:
                return False
            return dfs(node.left, lower, node.val) and dfs(node.right, node.val, upper)
        return dfs(root, float('-inf'), float('inf'))

if __name__ == '__main__':
    # test code
    root = TreeNode(2)
    root.left = TreeNode(1)
    root.right = TreeNode(3)
    assert Solution().isValidBST(root) == True
    root = TreeNode(5)
    root.left = TreeNode(1)
    root.right = TreeNode(4)
    root.right.left = TreeNode(3)
    root.right.right = TreeNode(6)
    assert Solution().isValidBST(root) == False


# 我们用递归来遍历整个树，并逐步缩小每个节点的有效值范围：

# 	1.	从根节点开始：
# 	•	当前节点是 5，它是树的根节点，因此它没有初始的上下界限制。
# 	•	对左子树，我们将 5 作为上界，所有左子树节点必须小于 5。
# 	•	对右子树，我们将 5 作为下界，所有右子树节点必须大于 5。
# 	2.	左子树的递归：
# 	•	当前节点是 3，其上界为 5（因为它在 5 的左子树上），下界为负无穷大（-∞）。
# 	•	检查 3 是否符合范围条件：-∞ < 3 < 5，符合。
# 	•	继续检查 3 的左子树，最大值为 3，右子树最小值也为 3。
# 	3.	检查 3 的左子树：
# 	•	当前节点是 2，其上界是 3，下界是 -∞。
# 	•	检查 2 是否符合范围条件：-∞ < 2 < 3，符合。
# 	•	2 没有左右子树，递归返回。
# 	4.	检查 3 的右子树：
# 	•	当前节点是 4，其上界是 5，下界是 3。
# 	•	检查 4 是否符合范围条件：3 < 4 < 5，符合。
# 	•	4 没有左右子树，递归返回。
# 	5.	右子树的递归：
# 	•	回到根节点 5，检查它的右子树。
# 	•	当前节点是 7，其下界是 5，上界为正无穷大（+∞）。
# 	•	检查 7 是否符合范围条件：5 < 7 < +∞，符合。
# 	•	继续检查 7 的右子树，最小值为 7。
# 	6.	检查 7 的右子树：
# 	•	当前节点是 8，其上界是 +∞，下界是 7。
# 	•	检查 8 是否符合范围条件：7 < 8 < +∞，符合。
# 	•	8 没有左右子树，递归返回。

# 递归调用流程

# 每次递归，我们会更新当前节点的上下界，逐层深入，直到叶节点。这种递归机制保证了每个节点都满足二叉搜索树的定义。
# 当前节点	下界	上界	是否满足条件
# 5	        -∞	    +∞	    是
# 3	        -∞	    5	    是
# 2	        -∞	    3	    是
# 4	        3	    5	    是
# 7	        5	    +∞	    是
# 8	        7	    +∞	    是