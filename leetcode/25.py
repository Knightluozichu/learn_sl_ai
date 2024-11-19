# Definition for singly-linked list.
# %%
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        # 辅助函数：翻转链表的一部分，并返回新的头和尾
        def reverse(start, end):
            prev = end.next
            curr = start
            while curr != end.next:
                temp = curr.next
                curr.next = prev
                prev = curr
                curr = temp
            return end, start
        
        # 计算链表的长度
        length = 0
        current = head
        while current:
            length += 1
            current = current.next
        
        # 创建虚拟头节点，便于操作
        dummy = ListNode(0)
        dummy.next = head
        prev_group_end = dummy
        
        # 遍历链表，按每 k 个节点进行翻转
        while length >= k:
            # 找到当前组的头和尾
            group_start = prev_group_end.next
            group_end = prev_group_end
            for _ in range(k):
                group_end = group_end.next
            
            # 翻转当前组并更新连接关系
            new_start, new_end = reverse(group_start, group_end)
            prev_group_end.next = new_start
            new_end.next = group_end.next
            prev_group_end = new_end
            
            # 减少剩余长度
            length -= k
        
        return dummy.next
    
# test [1,2,3,4,5] k=2
# 生成测试案例
head = ListNode(1)
current = head
for i in range(2, 6):
    current.next = ListNode(i)
    current = current.next
k = 2

# 调用函数
solution = Solution()
new_head = solution.reverseKGroup(head, k)

# 打印结果
while new_head:
    print(new_head.val)
    new_head = new_head.next
# 2

