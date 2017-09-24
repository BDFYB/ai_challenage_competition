# -*- coding: UTF-8 -*-
import jieba


def separate_by_words_and_letter(sentences):
    letters = {}
    words = {}
    letters['<s>'] = 0
    letters['</s>'] = 0
    """
    根据sentences list做分词，并统计词频
    """
    for sentence in sentences:
        letters['<s>'] += 1
        words['</s>'] += 1
        for letter in sentence:
            if letter in letters:
                letters[letter] += 1
            else:
                letters[letter] = 1
        seg_list = jieba.lcut(sentence, cut_all=False)
        for word in seg_list:
            if word in words:
                words[word] += 1
            else:
                words[word] = 1
    sorted_letters = sorted(letters.items(), key=lambda letters:letters[1], reverse=True)
    sorted_words = sorted(words.items(), key=lambda words:words[1], reverse=True)
    return sorted_letters, sorted_words

def jieba_decode(sentence):
    seg_list = jieba.lcut(sentence, cut_all=False)
    content = " ".join(seg_list)
    content = '<s> ' + content + ' </s>'
    return content

if __name__ == "__main__":
    data = u'\u4e00\u4e2a\u7a7f\u7740\u7070\u8272\u88e4\u5b50\u7684\u5973\u4eba\u548c\u4e00\u4e2a\u5de6\u624b\u62ff\u7740\u592a\u9633\u955c\u7684\u7537\u4eba\u8d70\u5728\u9053\u8def\u4e0a'
    print("oridata: ", data)
    print("decoded: ", jieba_decode(data))


