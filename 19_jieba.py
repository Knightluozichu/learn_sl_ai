# %%
import jieba
from nltk.tokenize import punkt

text = '我来到北京清华大学'
seg_list = jieba.cut(text, cut_all=True)
print('全模式：', '/'.join(seg_list))

seg_list = jieba.cut(text, cut_all=False)
print('精确模式：', '/'.join(seg_list))

seg_list = jieba.cut_for_search(text)
print('搜索引擎模式：', '/'.join(seg_list))

words = jieba.lcut("他来到了网易杭研大厦")
print(words)

jieba.suggest_freq(('中', '将'), True)
words = jieba.lcut("如果放到post中将出错。")
print(words)

# 关键词提取
sentence = """
以下是 2024 年 11 月 11 日的一些热点新闻：
1. **中国政府公布黄岩岛领海基线**：10 日，中国政府发表声明，宣布中华人民共和国黄岩岛的领海基线，这是中方对黄岩岛主权的明确宣示。针对此事件，外交部发言人表示，黄岩岛是中国固有领土，菲方所谓“海洋区域法”严重侵犯中方，中方对此坚决反对。
2. **第七届进博会意向成交金额突破 800 亿美元**：11 月 10 日，第七届进博会闭幕，数据显示，本届进博会按一年计意向成交金额超 800 亿美元，比上届增长 2.0%。
3. **人民空军成立 75 周年纪念日**：11 月 11 日是人民空军成立 75 周年纪念日。
4. **自然资源部、民政部公布我国南海部分岛礁标准名称**：11 月 10 日，自然资源部、民政部公布了我国南海部分岛礁标准名称。
5. **天舟七号货运飞船顺利撤离空间站组合体**：11 月 10 日 16 时 30 分，天舟七号货运飞船顺利撤离空间站组合体，转入独立飞行阶段，将于近期择机受控再入大气层。
6. **全国总工会等四部门联合印发《新就业形态劳动者权益协商指引》**：该指引指导平台企业合理制定涉及劳动者权益的制度规则和平台算法。
7. **多起文体赛事结果**：
    - **2024 世界斯诺克国际锦标赛决赛**：11 月 10 日，中国选手丁俊晖战胜英格兰选手韦克林夺冠，这是丁俊晖 2019 年英锦赛夺冠后，时隔五年再次获得排名赛冠军。
    - **2024WTT 法兰克福冠军赛**：11 月 10 日，中国选手王曼昱 4 比 2 战胜王艺迪夺冠，国乒包揽女单冠亚军；林诗栋 4 比 1 战胜瑞典选手卡尔伯格，取得男单冠军。
8. **一些国际事件**：
    - **内塔尼亚胡承认制造黎巴嫩传呼机爆炸事件**：当地时间 11 月 10 日，以色列总理内塔尼亚胡表示，他坚持主张实施针对黎巴嫩真主党的寻呼机爆炸计划及暗杀其时任领导人的行动，尽管有安全部门高级官员反对。
    - **美国商务部要求台积电对运往大陆的部分芯片实施出口限制**：路透社报道，美国商务部已要求台积电对运往大陆的部分 7 纳米或更先进芯片实施出口限制，起因是加拿大拆解机构称在华为昇腾 910B 中发现疑似台积电 7 纳米芯片组。
9. **国内其他事件**：
    - **河南安阳妇幼保健院排查偷拍摄像头事件**：安阳妇幼保健院就医院被传暗藏摄像头一事发布通报，已抓获违法行为人并对全院进行全面排查，未发现其他监视监听设备。
    - **安徽六安一幼儿园教师因体罚学生被采取强制措施**：六安市金安区教育局 11 月 10 日发布通报，幼儿园教师靳某因体罚学生被采取强制措施，该幼儿园被责令全面整顿。
"""

# 关键词提取
import jieba.analyse
keywords = jieba.analyse.extract_tags(sentence, topK=20, withWeight=True)

print(keywords)

# add word
jieba.add_word('黄岩岛')

# freq
jieba.del_word('勇敢的人')

# 词性标注
import jieba.posseg as pseg
words = pseg.cut("他绝对是一个勇敢的人")
for word, flag in words:
    print('%s %s' % (word, flag))

pseg.lcut("他绝对是一个勇敢的人")
# %%
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize, word_tokenize

EXAMPLE_TEXT = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."
print(sent_tokenize(EXAMPLE_TEXT))