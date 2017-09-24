#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
图片转tf_record类，输入imageinfo为list，每个元素为一个map，包含url（原始地址）、image_id（图片名称）、caption（图片描述）
caption为一个数组，每个图片可对应多个描述。
存储时将存储image_id,caption（图片及描述一对一），因此一个图片对应多条记录
建议读取时shuffle

"""

import tensorflow as tf
import numpy as np
import os
import sys
"""
for python2
reload(sys) # Python2.5 初始化后会删除 sys.setdefaultencoding 这个方法，我们需要重新载入 
sys.setdefaultencoding('utf-8') 
"""

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


class ImageDecoder(object):
  """Decoding images in TensorFlow."""

  def __init__(self):
    # Create a single TensorFlow Session for all image decoding calls.
    self._sess = tf.Session()

    # TensorFlow ops for JPEG decoding.
    self._encoded_jpeg = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)

  def decode_jpeg(self, encoded_jpeg):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._encoded_jpeg: encoded_jpeg})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

class ImageTFREncoder(object):
    def __init__(self, image_info, jpg_dir, result_file):
        self.image_desc = image_info
        self.image_dir = jpg_dir
        self.decoder = ImageDecoder()
        self.writer = tf.python_io.TFRecordWriter(result_file)

    def __del__(self):
        self.writer.close()

    def make_tf_record(self):
        file_list = os.listdir(self.image_dir)

        for single_record in self.image_desc:
            url = single_record['url']
            image_file_name = single_record['image_id']
            caption_list = single_record['caption']
            if image_file_name not in file_list:
                print("[WARNING] file not exsists! %s is not in %s!" % (image_file_name, self.image_dir))
                continue
            real_file_name = os.path.join(self.image_dir, image_file_name)
            for caption in caption_list:
                #利用example协议持久化
                example = self._to_example(real_file_name, image_file_name, caption)
                if example is not None:
                    self.writer.write(example.SerializeToString())

                """
                #利用sequence_example协议持久化
                sequence_example = self._to_sequence_example(real_file_name, image_file_name, caption)
                if sequence_example is not None:
                    self.writer.write(sequence_example.SerializeToString())
                """

    def _to_example(self, image, image_name, caption):
        #利用tfrecord文件的example协议持久化
        with tf.gfile.FastGFile(image, "r") as f:
            encoded_image = f.read()
        try:
            decoded_image = self.decoder.decode_jpeg(encoded_image)
        except (tf.errors.InvalidArgumentError, AssertionError):
            print("Skipping file with invalid JPEG data: %s" % image)
            return 

        example = tf.train.Example(features=tf.train.Features(feature={
            'caption': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(caption)])),
            'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(decoded_image)])),
            'image_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(image_name)])),
        }))
        return example

    def _to_sequence_example(self, image, image_name, caption):
        #利用tfrecord文件的sequence_example协议持久化
        with tf.gfile.FastGFile(image, "r") as f:
            encoded_image = f.read()
        try:
            decoded_image = self.decoder.decode_jpeg(encoded_image)
        except (tf.errors.InvalidArgumentError, AssertionError):
            print("Skipping file with invalid JPEG data: %s" % image)
            return 

        context = tf.train.Features(feature={
            "image/image_id": _bytes_feature(image_name),
            "image/data": _bytes_feature(decoded_image),
        })

        feature_lists = tf.train.FeatureLists(feature_list={
            "image/caption": _bytes_feature_list(caption),
        })
        sequence_example = tf.train.SequenceExample(
            context=context, feature_lists=feature_lists)

        return sequence_example


if __name__ == "__main__":
    #example
    current_dir = os.path.split( os.path.realpath( sys.argv[0] ) )[0]
    jpg_dir = current_dir + "/../ori_data/caption_train_images_part"
    result_file = current_dir + "/../processed_data/img_TFRecord"
    data = [{u'url': u'http://www.sinaimg.cn/dy/slidenews/4_img/2015_21/704_1634177_196978.jpg', 
             u'image_id': u'ad0bbf05ad434f4028191c0e2cf7e2d6f5f31ea8.jpg', 
             u'caption': [u'\u4e00\u4e2a\u7a7f\u7740\u7070\u8272\u88e4\u5b50\u7684\u5973\u4eba\u548c\u4e00\u4e2a\u5de6\u624b\u62ff\u7740\u592a\u9633\u955c\u7684\u7537\u4eba\u8d70\u5728\u9053\u8def\u4e0a', 
                          u'\u4e00\u4e2a\u6234\u7740\u58a8\u955c\u7684\u5973\u4eba\u548c\u4e00\u4e2a\u7a7f\u7740\u7070\u8272\u4e0a\u8863\u7684\u7537\u4eba\u62ff\u7740\u51b0\u6fc0\u51cc\u8d70\u5728\u9053\u8def\u4e0a', 
                          u'\u4e00\u4e2a\u5de6\u624b\u62ff\u7740\u58a8\u955c\u7684\u7537\u4eba\u548c\u4e00\u4e2a\u6234\u7740\u58a8\u955c\u7684\u5973\u4eba\u8d70\u5728\u9053\u8def\u4e0a', 
                          u'\u4e00\u4e2a\u5de6\u624b\u62ff\u7740\u58a8\u955c\u7684\u7537\u4eba\u548c\u4e00\u4e2a\u5de6\u624b\u62ff\u7740\u51b0\u6dc7\u6dcb\u7684\u5973\u4eba\u8d70\u5728\u9053\u8def\u4e0a', 
                          u'\u9053\u8def\u4e0a\u4e00\u4e2a\u6234\u7740\u5e3d\u5b50\u7684\u7537\u4eba\u5728\u8ddf\u4e00\u4e2a\u5de6\u624b\u62ff\u7740\u51b0\u6fc0\u51cc\u7684\u5973\u4eba\u8bb2\u8bdd'
                          ]}]
    decoder = ImageTFREncoder(data, jpg_dir, result_file)
    decoder.make_tf_record()

