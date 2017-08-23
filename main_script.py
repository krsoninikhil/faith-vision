
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[10]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


# ## Env setup

# In[11]:


# This is needed to display the images.
#%matplotlib inline

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

# In[12]:


from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[13]:


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ## Download Model

# In[14]:


opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

# In[15]:


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[16]:


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[17]:

while(True):
    def load_image_into_numpy_array(image):
      (im_width, im_height) = image.size
      return np.array(image.getdata()).reshape(
          (im_height, im_width, 3)).astype(np.uint8)

    # In[18]:
    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    PATH_TO_TEST_IMAGES_DIR = '/home/ubuntu/server'
    TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'shot.jpg') ]
    print(TEST_IMAGE_PATHS)
    # Size, in inches, of the output images.
    IMAGE_SIZE = (12, 8)
    # In[19]:

    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        for image_path in TEST_IMAGE_PATHS:
          image = Image.open(image_path)
          print(image)
          #exit()
          # the array based representation of the image will be used later in order to prepare the
          # result image with boxes and labels on it.
          image_np = load_image_into_numpy_array(image)
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          # Each box represents a part of the image where a particular object was detected.
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          # Each score represent how level of confidence for each of the objects.
          # Score is shown on the result image, together with the class label.
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          # Actual detection.
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          # Visualization of the results of a detection.
        #   vis_util.visualize_boxes_and_labels_on_image_array(
        #       image_np,
        #       np.squeeze(boxes),
        #       np.squeeze(classes).astype(np.int32),
        #       np.squeeze(scores),
        #       category_index,
        #       use_normalized_coordinates=True,
        #       line_thickness=8)
        #   plt.figure(figsize=IMAGE_SIZE)
        #   plt.imshow(image_np)
    # In[20]:
    classes_name = []
    # In[21]:

    classes_indices = classes[0].astype(np.int32)
    # In[22]:
    for i in classes_indices:
        for j in categories:
            if i == j['id']:
                classes_name.append(j['name'])
                break

    # In[23]:
    scores_list = list(scores[0])

    # In[24]:
    rel_classes_name = [classes_name[i] for i in range(len(scores_list)) if scores_list[i] > 0.7]

    # In[25]:
    boxes_list = list(boxes[0])

    # In[26]:
    rel_boxes_list = [boxes_list[i] for i in range(len(scores_list)) if scores_list[i] > 0.7]
    # In[27]:
    rel_position_list = []
    def positionLoc(rel_boxes_list):
        for box in rel_boxes_list:
            if box[3] < 0.5:
                if box[2] < 0.5:
                    rel_position_list.append("Top Left")
                elif box[0] > 0.5:
                    rel_position_list.append("Bottom Left")
                else:
                    rel_position_list.append("Left")
            elif box[1] > 0.5:
                if box[2] < 0.5:
                    rel_position_list.append("Top Right")
                elif box[0] > 0.5:
                    rel_position_list.append("Bottom Right")
                else:
                    rel_position_list.append("Right")
            else:
                rel_position_list.append("Center")

    # In[28]:
    positionLoc(rel_boxes_list)

    # In[29]:
    output_dict = {}
    for i in range(len(rel_classes_name)):
        output_dict.update({'item'+str(i+1): 'There is a %s to the %s.' % (rel_classes_name[i], rel_position_list[i])})

    # In[31]:

    import time
    import json
    timeStamp = str(time.time())
    textstring = ""
    out = '/home/ubuntu/server/text.json'
    for v in output_dict:
        textstring += output_dict[v] + "."
    data = {'timestamp': timeStamp, 'text': textstring}
    with open(out,'w') as dat:
        json.dump(data, dat, sort_keys = False, indent=4)

    time.sleep(3)

# In[ ]:
