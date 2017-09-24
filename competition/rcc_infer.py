import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
# from matplotlib import pyplot as plt
from PIL import Image

# from utils import label_map_util

from utils import visualization_utils as vis_util

MODEL_NAME = '/Users/baidu/AI/ai_challenge_competition/competition/faster_RCNN_module/faster_rcnn_resnet101_coco_11_06_2017'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

# NUM_CLASSES = 90


# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
# category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# PATH_TO_TEST_IMAGES_DIR = 'test_images'
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

PATH_TO_TEST_IMAGES_DIR = '/Users/baidu/AI/ai_challenge_competition/competition/data_pre_processing/data_process_tf_record/ori_data/caption_train_images_part'
IMAGE_NAMES = [
	'e684e76b579d79ab8de43364f816ba8f642e4a1b.jpg',
	]
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, image_name) for image_name in IMAGE_NAMES ]

TEST_IMAGES = []
for image_path in TEST_IMAGE_PATHS:
  image = Image.open(image_path)
  image_np = load_image_into_numpy_array(image)
  TEST_IMAGES.append(image_np)


# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    #train_summary_writer = tf.summary.FileWriter('/home/fzy/project/ai', sess.graph)

    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    feature_map = detection_graph.get_tensor_by_name('SecondStageBoxPredictor/AvgPool:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    for index, image_path in enumerate(TEST_IMAGE_PATHS):
      # image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      # image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      # image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      (features, boxes, scores, classes, num) = sess.run(
          [feature_map,  detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: np.array(TEST_IMAGES)})
      
      print('features[0]: %d' % len(features))
      print(features)
      print('boxes[0]: %d' % len(boxes))
      print('boxes[0]: %d' % len(boxes[0]))
      print('boxes[0]: %d' % len(boxes[0][0]))
#      print('boxes[0]: %d' % len(boxes[0][0][0]))
#      print('boxes[2]: %d' % len(boxes.tolist()[2]))
      print(boxes[0, 0:3])
#      print(boxes[0, 1])
#      print(boxes)
      print(boxes[0][0])
      print(boxes[0][0][0])
#      exit()

      print('scores')
      print(scores)

      print('classes')
      print(classes)

      print('num')
      print(num)
      break

      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          None,#category_index,
          use_normalized_coordinates=True,
          agnostic_mode=True,
          line_thickness=8)
      
      box_image = Image.fromarray(image_np)
      box_image.save('/Users/baidu/AI/ai_challenge_competition/competition/faster_RCNN_module/result.jpg' % index)
      # plt.figure(figsize=IMAGE_SIZE)
      # plt.imshow(image_np)
