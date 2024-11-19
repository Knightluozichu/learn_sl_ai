class Solution:
    def romanToInt(self, s: str) -> int:
        """
        将罗马数字转换为整数

        参数:
            s (str): 输入的罗马数字字符串.

        返回:
            int: 转换后的整数值.

        示例:
            >>> romanToInt('III')
            3
            >>> romanToInt('IV')
            4
        """
        # 定义一个字典，用于存储罗马数字字符及其对应的整数值
        dic = {'I':1, 'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}

        # 如果输入的字符串长度为 1，则直接返回对应的值
        if len(s) == 1:
            return dic[s[0]]

        # 将字符串中的每个字符转换为对应的整数值，并存储在列表中
        numlist = []
        for sr in s:
            numlist.append(dic[sr])

        # 初始化一个变量，用于存储转换后的整数值
        num_sum = 0

        # 遍历列表中的每个元素
        for i in range(len(numlist)):
            # 获取当前元素的值
            cur = numlist[i]
            # 如果当前元素小于下一个元素，则减去当前元素的值
            if i < len(numlist) - 1 and cur < numlist[i + 1]:
                num_sum -= cur
            # 否则，加上当前元素的值
            else:
                num_sum += cur

        # 返回最终的整数值
        return num_sum

            

import unittest

class TestRomanToInt(unittest.TestCase):

    def test_single_roman_digit(self):
        # 测试单个罗马数字字符
        sol = Solution()
        self.assertEqual(sol.romanToInt('I'), 1)
        self.assertEqual(sol.romanToInt('V'), 5)
        self.assertEqual(sol.romanToInt('X'), 10)
        self.assertEqual(sol.romanToInt('L'), 50)
        self.assertEqual(sol.romanToInt('C'), 100)
        self.assertEqual(sol.romanToInt('D'), 500)
        self.assertEqual(sol.romanToInt('M'), 1000)

    def test_multiple_roman_digits(self):
        # 测试多个罗马数字字符
        sol = Solution()
        self.assertEqual(sol.romanToInt('II'), 2)
        self.assertEqual(sol.romanToInt('III'), 3)
        self.assertEqual(sol.romanToInt('IV'), 4)
        self.assertEqual(sol.romanToInt('IX'), 9)
        self.assertEqual(sol.romanToInt('XL'), 40)
        self.assertEqual(sol.romanToInt('XC'), 90)
        self.assertEqual(sol.romanToInt('CD'), 400)
        self.assertEqual(sol.romanToInt('CM'), 900)

    def test_long_roman_number(self):
        # 测试长罗马数字
        sol = Solution()
        self.assertEqual(sol.romanToInt('MCMXCIV'), 1994)
        self.assertEqual(sol.romanToInt('MMMCMXCIX'), 3999)

if __name__ == '__main__':
    unittest.main()
