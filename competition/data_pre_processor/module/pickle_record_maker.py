#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
图片转pickle类，输入imageinfo为list，每个元素为一个map，包含url（原始地址）、image_id（图片名称）、caption（图片描述）
caption为一个数组，每个图片可对应多个描述。
存储时将存储image_id,caption（图片及描述一对一），因此一个图片对应多条记录
建议读取时shuffle

"""

import tensorflow as tf
import numpy as np
import pickle
import os
import sys

import word_frequency

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

class ImagePKLEncoder(object):
    def __init__(self, image_info, jpg_dir, result):
        self.image_desc = image_info
        self.image_dir = jpg_dir
        self.decoder = ImageDecoder()
        self.result_file = result

    def __del__(self):
        self.writer.close()

    def persists(self):
        file_list = os.listdir(self.image_dir)
        image_decoded_list = []
        #singel_decoded_image = {}

        for single_record in self.image_desc:
            url = single_record['url']
            image_file_name = single_record['image_id']
            caption_list = single_record['caption']
            if image_file_name not in file_list:
                print("[WARNING] file not exsists! %s is not in %s!" % (image_file_name, self.image_dir))
                continue
            real_file_name = os.path.join(self.image_dir, image_file_name)
            for caption in caption_list:
                #做成map持久化
                singel_decoded_image = self._to_map(real_file_name, image_file_name, caption)
                image_decoded_list.append(singel_decoded_image)

        ret_file = open(self.result_file, 'wb')
        pickle.dump(image_decoded_list, ret_file)
        ret_file.close()

    def _to_map(self, image, image_name, caption):
        #利用tfrecord文件的example协议持久化
        #simple形式（利用pickle）不进行image解析
        """
        with tf.gfile.FastGFile(image, "r") as f:
            encoded_image = f.read()
        try:
            decoded_image = self.decoder.decode_jpeg(encoded_image)
        except (tf.errors.InvalidArgumentError, AssertionError):
            print("Skipping file with invalid JPEG data: %s" % image)
            return 
        """
        ret_map = {
            'caption': word_frequency.jieba_decode(caption),
            'image_id': image_name,
        }
        return ret_map


if __name__ == "__main__":
    #example
    current_dir = os.path.split( os.path.realpath( sys.argv[0] ) )[0]
    jpg_dir = current_dir + "/../ori_data/caption_train_images_part"
    result_file = current_dir + "/../processed_data/img_Pickle"
    data = [{u'url': u'http://www.sinaimg.cn/dy/slidenews/4_img/2015_21/704_1634177_196978.jpg', 
             u'image_id': u'ad0bbf05ad434f4028191c0e2cf7e2d6f5f31ea8.jpg', 
             u'caption': [u'\u4e00\u4e2a\u7a7f\u7740\u7070\u8272\u88e4\u5b50\u7684\u5973\u4eba\u548c\u4e00\u4e2a\u5de6\u624b\u62ff\u7740\u592a\u9633\u955c\u7684\u7537\u4eba\u8d70\u5728\u9053\u8def\u4e0a', 
                          u'\u4e00\u4e2a\u6234\u7740\u58a8\u955c\u7684\u5973\u4eba\u548c\u4e00\u4e2a\u7a7f\u7740\u7070\u8272\u4e0a\u8863\u7684\u7537\u4eba\u62ff\u7740\u51b0\u6fc0\u51cc\u8d70\u5728\u9053\u8def\u4e0a', 
                          u'\u4e00\u4e2a\u5de6\u624b\u62ff\u7740\u58a8\u955c\u7684\u7537\u4eba\u548c\u4e00\u4e2a\u6234\u7740\u58a8\u955c\u7684\u5973\u4eba\u8d70\u5728\u9053\u8def\u4e0a', 
                          u'\u4e00\u4e2a\u5de6\u624b\u62ff\u7740\u58a8\u955c\u7684\u7537\u4eba\u548c\u4e00\u4e2a\u5de6\u624b\u62ff\u7740\u51b0\u6dc7\u6dcb\u7684\u5973\u4eba\u8d70\u5728\u9053\u8def\u4e0a', 
                          u'\u9053\u8def\u4e0a\u4e00\u4e2a\u6234\u7740\u5e3d\u5b50\u7684\u7537\u4eba\u5728\u8ddf\u4e00\u4e2a\u5de6\u624b\u62ff\u7740\u51b0\u6fc0\u51cc\u7684\u5973\u4eba\u8bb2\u8bdd'
                          ]}]
    decoder = ImagePKLEncoder(data, jpg_dir, result_file)
    decoder.persists()

