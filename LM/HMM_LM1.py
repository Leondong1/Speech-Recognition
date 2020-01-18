'''
基于马尔可夫模型的语言模型

功能：给定一个拼音序列，返回对应的文字序列
'''


import time


class LanguageModel(): # 语言模型类
    def __init__(self, pny_vocab_dict_file, one_gram_file, two_gram_file, two_gram_pny_file):
        '''
        保存 2gram 字词统计文件  以及 拼音 路径
        :param modelpath:   2gram 字词统计文件路径
        '''

        self.pinyin_dict = self.readPinyinDict(pny_vocab_dict_file)
        self.one_gram_count = self.readNgramfile(one_gram_file)
        self.two_gram_count = self.readNgramfile(two_gram_file)
        self.two_gram_pinyin_count = self.twoGramPinyinCount(two_gram_pny_file)


    def readPinyinDict(self,dictfilename):
        '''
        读入拼音汉字的字典文件
        返回读取后的字典， 形式是  拼音: 汉字
        '''
        dic_symbol = {} # 初始化符号字典
        for i in open(dictfilename, 'r', encoding='utf8'):
            list_symbol = [] # 初始化符号列表
            if i != "":
                txt_l = i.split('\t')
                pinyin = txt_l[0]
                for word in txt_l[1]:
                    list_symbol.append(word)
                dic_symbol[pinyin] = list_symbol

        return dic_symbol

    def readNgramfile(self, filename):
        # 读词语出现的词频  用数据集统计出来的词频数据
        dic_model = {}
        for i in open(filename, 'r', encoding='utf8'):
            if i != "":
                txt_l = i.split('\t')
                if len(txt_l) == 1:
                    continue
                dic_model[txt_l[0]] = txt_l[1]
        return dic_model

    def twoGramPinyinCount(self, filename):
        # 拼音词语 后面是数字  用数据集统计出来的2元拼音词频数据
        dic={}

        for line in open(filename, 'r', encoding='utf8'):
            if line == '':
                continue
            pinyin_split = line.split('\t')
            # print(pinyin_split)

            list_pinyin = pinyin_split[0]
            if list_pinyin not in dic and int(pinyin_split[1]) > 1:
                dic[list_pinyin] = pinyin_split[1]
        return dic



    def pinyin2Text(self, list_pinyin):
        '''
        拼音 转 文字
        尚未完成
        :param list_pinyin:
        :return:
        '''
        text = ''
        length = len(list_pinyin)
        if length == 0: # 如果传入的长度是0
            return ''

        # 先取出一个字，即拼音列表中第一个字
        str_tmp = [list_pinyin[0]]
        # print(str_tmp, 124)

        count = length
        for i in range(0, length - 1):
        # while count > 0:
            # 依次从第一个字开始每次连续取两个字拼音
            # i = length - count
            str_split = list_pinyin[i] + ' ' + list_pinyin[i+1]
            # print(str_split, str_split in self.words_pinyin)
            # 如果这个拼音在汉语拼音状态转移字典的话
            if str_split in self.two_gram_pinyin_count:
                str_tmp.append(list_pinyin[i+1])
                # print("if", str_tmp)
                # count -= 1
            else:
                # 否则不加入，然后直接将现有的拼音序列解码
                # print("else str_tmp", str_tmp)
                str_decode = self.decode(str_tmp, 0.0001)
                # print("else", str_decode)
                print(i)
                if len(str_decode) > 0:
                    text += str_decode[0][0]
                    # print("text", text)
                str_tmp = [list_pinyin[i+1]]


        str_decode = self.decode(str_tmp,0.000001)
        if len(str_decode) > 0:
            text += str_decode[0][0]

        return text


    def decode(self, words_pinyin, threshold=0.000001):
        '''
        给定拼音词语，进行解码， 基于马尔可夫链
        '''
        words_list = []
        pinyin_count = len(words_pinyin)
        # print('pinyin_count',pinyin_count)
        # logging.info('pinyin_count')
        # print(words_pinyin)

        for i in range(pinyin_count):
            # 如果字典中有收录该字
            # 取出所有的候选字
            word_candidates = self.pinyin_dict.get(words_pinyin[i])
            if not word_candidates:
                break

            word_candidates_count = len(word_candidates)
            if i == 0:
                # 第一个字做初始化处理
                for j in range(word_candidates_count):
                    tuple_word = ['', 0.0]
                    # 设置马尔可夫模型初始状态值
                    # 设置初始概率，1.0
                    tuple_word = [word_candidates[j], 1.0]
                    # 添加到可能的句子列表
                    words_list.append(tuple_word)

            else:
                # 处理后面的字
                new_words_list = []
                words_num = len(words_list)
                for j in range(words_num): # 遍历已有的所有可能的短语序列

                    for k in range(word_candidates_count): # 根据当前位置的拼音遍历当前位置所有可取的汉字
                        tuple_word = list(words_list[j])  # 取出第j个已有的短语序列
                        # print('tuple_word1: ',tuple_word)
                        tuple_word[0] = tuple_word[0] + word_candidates[k]  # 尝试按照下一个音可能对应的全部的字进行组合
                        # print('tuple_word[0]  ',tuple_word[0])
                        # print('word_candidates[%d]  '%k,word_candidates[k])

                        tmp_words = tuple_word[0][-2:]  # 取出用于计算的最后两个字
                        # print('tmp_words: ',tmp_words,tmp_words in self.two_gram_count)
                        if tmp_words in self.two_gram_count:  # 判断它们是不是再状态转移表里
                            tuple_word[1] = tuple_word[1] * float(self.two_gram_count[tmp_words]) / float(self.one_gram_count[tmp_words[-2]])
                        # 当前概率上乘转移概率，公式化简后为第n-1和n个字出现的次数除以第n-1个字出现的次数
                        else:
                            tuple_word[1] = .0
                        # print('tuple_word2: ',tuple_word)
                        # print(tuple_word[1] >= pow(threshold, i))
                        if tuple_word[1] >= pow(threshold, i):
                            # 大于阈值之后保留，否则丢弃
                            # 由于文字长度增加之后相乘的概率会越来越低，所以这里的阈值是幂次方
                            new_words_list.append(tuple_word)

                words_list = new_words_list  # 新的短语序列tuple列表替换掉原有的短语序列tuple列表
            # print(words_list,'\n')  # 当前位置的候选短语序列tuple列表
        # print(words_list) # 全部拼音的候选短语序列tuple列表
        return sorted(words_list, key=(lambda x: x[-1]), reverse=True)






if __name__ == '__main__':
    pny_vocab_dict_file = '../Pny_2gram_files/pny_dict/pny_vocab_dict.txt'
    one_gram_file = '../Pny_2gram_files/1gram_count.txt'
    two_gram_file = '../Pny_2gram_files/2gram_count.txt'
    two_gram_pny_file = '../Pny_2gram_files/2gram_pinyin_count.txt'
    ml = LanguageModel(pny_vocab_dict_file, one_gram_file, two_gram_file, two_gram_pny_file)

    # str_pinyin = ['kao3', 'yan2', 'ying1', 'yu3', 'ci2', 'hui4']
    # str_pinyin = ['da4', 'jia1', 'hao3']
    # str_pinyin = ['kao3', 'yan2', 'yan1', 'yu3', 'ci2', 'hui4']
    str_pinyin = ['kao3', 'yan2', 'ying1', 'yu3', 'ci2', 'hui4']
    # str_pinyin = ['zhe4', 'zhen1', 'shi4', 'ji2', 'hao3', 'de5']
    # str_pinyin = ['jin1', 'tian1', 'shi4', 'xing1', 'qi1', 'san1']
    # str_pinyin = ['ni3', 'hao3', 'a1']
    # str_pinyin = ['wo3', 'dui4', 'shi4', 'mei2', 'cuo4', 'ni3', 'hao3']
    # str_pinyin = ['wo3', 'dui4', 'shi4', 'tian1', 'mei2', 'na5', 'li3', 'hai4']
    # str_pinyin = ['ba3', 'zhe4', 'xie1', 'zuo4', 'wan2', 'wo3', 'jiu4', 'qu4', 'shui4', 'jiao4']
    # str_pinyin = ['wo3', 'qu4', 'a4', 'mei2', 'shi4', 'er2', 'la1']
    # str_pinyin = ['wo3', 'men5', 'qun2', 'li3', 'xiong1', 'di4', 'jian4', 'mei4', 'dou1', 'zai4', 'shuo1']
    # str_pinyin = "ta1 jin3 ping2 yao1 bu4 de li4 liang4 zai4 yong3 dao4 shang4 xia4 fan1 teng2 yong3 dong4 she2 xing2 zhuang4 ru2 hai3 tun2 yi4 zhi2 yi3 yi1 tou2 de you1 shi4 ling3 xian1".split()

    t1 = time.time()
    r = ml.pinyin2Text(str_pinyin)
    # r = ml.decode(str_pinyin)
    print('语音转文字结果：\n', r)
    print(time.time() - t1)



