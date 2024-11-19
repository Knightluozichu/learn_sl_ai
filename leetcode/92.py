# 92. 反转链表 II
# 中等
# 相关标签
# 相关企业
# 给你单链表的头指针 head 和两个整数 left 和 right ，其中 left <= right 。请你反转从位置 left 到位置 right 的链表节点，返回 反转后的链表 。


# Definition for singly-linked list.
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        newHead = ListNode(0)
        newHead.next = head
        pre = newHead
        for i in range(left - 1):
            pre = pre.next
        cur = pre.next
        for i in range(right - left):
            temp = pre.next
            pre.next = cur.next
            cur.next = cur.next.next
            pre.next.next = temp
        return newHead.next

            

s = Solution()
head = ListNode(1)
current = head
for i in range(2, 6):
    # print(i)
    current.next = ListNode(i)
    current = current.next
s.reverseBetween(head, 2, 4)