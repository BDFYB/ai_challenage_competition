#!/usr/bin/python
# -*- coding: UTF-8 -*-

import json
import os
import jieba
import sys
import pickle

"""
reload(sys)
sys.setdefaultencoding('utf8')
"""

sys.path.append(sys.path[0] + '/./')
sys.path.append(sys.path[0] + '/module')

from module import word_frequency
from module import tf_record_maker
from module import pickle_record_maker


def load_data(file_name):
    ret_lines = []
    if os.path.exists(file_name) and os.path.isfile(file_name):
        fd = open(file_name)
        for line in fd.readlines():
            ret_lines.append(line)
    else:
        print("%s file is not exsists!" % file_name)
        exit(1)

    return ret_lines
    
def persists_to_file(data_map, file_name):
    if os.path.exists(file_name):
        if os.path.isfile(file_name):
            os.remove(file_name)
        elif os.path.isdir(file_name):
            print("same name dir exsists! " + file_name)
            return 1

    fd = open(file_name, 'w')
    for (key, value) in data_map:
        string = str(key) + ' ' + str(value) + '\n'
        fd.write(string)
    fd.close()

if __name__ == "__main__":
    #功能开关
    generate_word_frequency = 1
    generate_TFRecord = 0
    generate_PICKLE_record = 1
    """
    读取描述文件，获得sentences list及照片描述的list
    """
    """
    #测试数据
    current_dir = os.path.split(os.path.realpath(sys.argv[0]))[0]
    input_annotions_file = current_dir + '/ori_data/caption_train_annotations_part.json'
    input_image_dir = current_dir + '/ori_data/caption_train_images_part'
    output_processed_word_jieba = current_dir + '/processed_data/word_frequency_jieba'
    output_processed_word_raw = current_dir + '/processed_data/word_frequency_single'
    output_word_dict = current_dir + '/processed_data/word_dict_pickle'
    """

    #正式数据
    current_dir = '/home/fzy/sea/challenger/ai_challenger_caption_train_20170902/'
    input_annotions_file = current_dir + 'caption_train_annotations_20170902.json'
    input_image_dir = current_dir + 'caption_train_images_20170902/'
    output_processed_word_jieba = current_dir + 'processed_data/word_frequency_jieba'
    output_processed_word_raw = current_dir + 'processed_data/word_frequency_single'
    output_word_dict = current_dir + 'processed_data/word_dict_pickle'

    json_str = load_data(input_annotions_file)
    image_desc = []
    sentences = []
    for a_str in json_str:
        data = json.loads(a_str)
        for dat in data:
            image_desc.append(dat)
            sentences.extend(dat['caption'])
    print("picture num: %s" % len(image_desc))
    print("sentence num: %s" % len(sentences))


    word_dicts = {}
    #调用词频统计模块
    if generate_word_frequency:
        sorted_letters, sorted_words = word_frequency.separate_by_words_and_letter(sentences)
        #制作word的词典文件
        count = 1
        for sorted_word in sorted_words:
            count += 1
            word_dicts[sorted_word[0]] = count
        """
        持久化word_dict结果
        """
        persists_to_file(sorted_words, output_processed_word_jieba)
        persists_to_file(sorted_letters, output_processed_word_raw)
        fd = open(output_word_dict, 'wb')
        pickle.dump(word_dicts, fd)
        fd.close()

    #生成TFRecord文件，复杂模式（最终）
    if generate_TFRecord:
        result_file = current_dir + "/processed_data/img_TFRecord"
        decoder = tf_record_maker.ImageTFREncoder(data, input_image_dir, result_file)
        decoder.make_tf_record()
    #生成pickle文件持久化数据，临时模式
    if generate_PICKLE_record:
        result_file = current_dir + "/processed_data/img_PKL_record"
        decoder = pickle_record_maker.ImagePKLEncoder(data, input_image_dir, result_file)
        decoder.persists()
