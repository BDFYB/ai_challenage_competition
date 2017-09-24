#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf
import os
import sys
import pickle
import random


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

class data_reader(object):
    """
    file_type: 
       TFR:0 
       PKL:1
    """
    def __init__(self, data_file_name, word_dict_file_name, batch_size, file_type=1):
        #文件名请填写绝对路径
        self.file_type = file_type
        self.data_file_name = data_file_name
        self.word_dict_file_name = word_dict_file_name
        self.batch_size = batch_size

        #读取word dict
        word_file_fd = open(word_dict_file_name, "rb")
        self.word_dict = pickle.load(word_file_fd)
        word_file_fd.close()

        self.data_list = []
        if file_type == 0:
            #TFR
            raise Exception("TFR was not supported")
        elif file_type == 1:
            #Pickle
            file_fd = open(data_file_name, "rb")
            self.data_list = pickle.load(file_fd)
            file_fd.close()

    def get_word_dict(self):
        if not self.word_dict:
            print("Warning: %s word_dict 为空数据" % self.word_dict_file_name)

        return self.word_dict

    def get_main_data_in_PKL(self):
        #返回一个list，元素为一个map，key分别为图片路径和caption（已分词后）
        if self.file_type == 0:
            raise Exception("实例文件类型为TFR，请调用get_main_data_in_TFR()获取数据")
        return self.data_list

    def get_main_data_in_TFR(self):
        #todo
        if self.file_type == 1:
            raise Exception("实例文件类型为PKL，请调用get_main_data_in_PKL()获取数据")
        with tf.Session() as sess: #开始一个会话
            tf.global_variables_initializer().run()
            img_data, caption = _read_and_decode_in_example_proto(file_name)
            return img_data, caption

    def get_data_in_batch(self, changed_batch_size = -1):
        if self.file_type == 0:
            raise Exception("实例文件类型为TFR，暂未支持用batch获取")

        size = self.batch_size
        if changed_batch_size > 0:
            size = changed_batch_size

        num_range = range(len(self.data_list))
        index_list = random.sample(num_range, size)
        ret_batch = []
        for i in index_list:
            ret_batch.append(self.data_list[i])
        return ret_batch


    def _read_and_decode_in_example_proto(self):
        #根据文件名生成一个队列
        filename_queue = tf.train.string_input_producer([self.data_file_name])

        reader = tf.TFRecordReader()
        _, serialized_data = reader.read(filename_queue)   #返回文件名和文件内容
        features = tf.parse_single_example(serialized_data,
                                           features={
                                               "image_id": tf.FixedLenFeature([], tf.string),
    										   "data": tf.FixedLenFeature([], tf.string),
    										   "caption": tf.FixedLenFeature([], tf.string),
                                           })
        img = tf.decode_raw(features['data'], tf.uint8)
        #img = tf.reshape(img, [224, 224, 3])
        caption = features['caption']
        print(caption)
        return img, caption

           
    def _read_and_decode_in_sequence_example_proto(self):
        #解析采用sequence_example协议持久化的数据
        sequence_features = {
            "image/image_id": tf.FixedLenFeature([],dtype=tf.string),
            "image/data": tf.FixedLenFeature([],dtype=tf.int64)
        }
        context_features = {
            "image/caption": tf.FixedLenSequenceFeature([3], dtype=tf.float32,allow_missing=True)
        }

        #根据文件名生成一个队列
        filename_queue = tf.train.string_input_producer([self.data_file_name])

        reader = tf.TFRecordReader()
        _, serialized_data = reader.read(filename_queue)   #返回文件名和文件内容
        features = tf.parse_single_example(serialized_data,
                                           features={
                                               "image/image_id": _bytes_feature(image_name),
    										   "image/data": _bytes_feature(decoded_image),
                                           })

        img = tf.decode_raw(features['img/data'], tf.uint8)
        #img = tf.reshape(img, [224, 224, 3])
        #img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
        label = tf.cast(features['label'], tf.int32)

        return img, label


if __name__ == "__main__":
    """
    file_type: 
       TFR:0 暂未支持
       PKL:1 默认
    """
    file_type=1
    current_dir = os.path.split( os.path.realpath(sys.argv[0]))[0]
    word_dict_file_name = current_dir + "/processed_data/word_dict_pickle"


    #读取data文件例子
    #   TFR文件格式例子
    if file_type == 0:
        data_file_name = current_dir + "/processed_data/img_TFRecord"
        reader = data_reader(data_file_name, word_dict_file_name, batch_size = 2, file_type=0)
        
        #读取Word词典文件
        word_dict = reader.get_word_dict()
        print(word_dict)

        img_data, caption = reader.get_main_data_in_TFR()
        #队列读取使用示例
        coord=tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(20):
            example, l = sess.run([img_data, caption])#在会话中取出image和label
            #example, l= sess.run(reader._read_and_decode_in_example_proto())
            print(example, l)
        coord.request_stop()
        coord.join(threads)
        sess.close()

    #    Pickle文件格式例子
    elif file_type == 1:
        data_file_name = current_dir + "/processed_data/img_PKL_record"
        reader = data_reader(data_file_name, word_dict_file_name, batch_size = 2, file_type=1)
        
        #读取Word词典文件
        word_dict = reader.get_word_dict()
        print("word dict: \n %s" % word_dict)

        image_info_list = reader.get_main_data_in_PKL()
        print("data info: \n %s" % image_info_list)

        data_batch = reader.get_data_in_batch()
        print ("batched data info: \n %s" % data_batch)
