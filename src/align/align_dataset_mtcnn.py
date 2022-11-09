"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imageio
from PIL import Image
import os
import tensorflow as tf
import numpy as np
import facenet
from src.align import detect_face
import random
from time import sleep


#  python align/align_dataset_mtcnn.py data/new_data data/new_data_160 --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.4
# has_class_directories == true表示对文件夹内的所有class的图片重新对齐
# has_class_directories == false表示只对当前class的图片对齐
def process_all_160(input_dir, output_dir):
    random_order = True
    detect_multiple_faces = False
    margin = 32
    image_size = 160
    sleep(random.random())
    output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Store some git revision info in a text file in the log directory
    # src_path = D:\Py_projects\final_facenet
    # src_path, _ = os.path.split(os.path.realpath(__file__))
    # facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))

    # 若has_class_directories == True，则返回的是分类好的所有人的图片，否则只是返回在系统界面录入的人的图片
    dataset = facenet.get_dataset(input_dir)

    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)  #0.25
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor


    nrof_images_total = 0
    nrof_successfully_aligned = 0
    if random_order:
        random.shuffle(dataset)     #打乱原始顺序
    for cls in dataset:
        output_class_dir = os.path.join(output_dir, cls.name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
            if random_order:
                random.shuffle(cls.image_paths)
        for image_path in cls.image_paths:
            nrof_images_total += 1
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            output_filename = os.path.join(output_class_dir, filename + '.png')
            print(image_path)
            if not os.path.exists(output_filename):
                try:
                    img = imageio.imread(image_path)
                except (IOError, ValueError, IndexError) as e:
                    errorMessage = '{}: {}'.format(image_path, e)
                    print(errorMessage)
                else:
                    if img.ndim < 2:
                        print('Unable to align "%s"' % image_path)
                        continue
                    if img.ndim == 2:
                        img = facenet.to_rgb(img)
                    img = img[:, :, 0:3]

                    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold,
                                                                factor)
                    nrof_faces = bounding_boxes.shape[0]
                    if nrof_faces > 0:
                        det = bounding_boxes[:, 0:4]
                        det_arr = []
                        img_size = np.asarray(img.shape)[0:2]
                        if nrof_faces > 1:
                            if detect_multiple_faces:
                                for i in range(nrof_faces):
                                    det_arr.append(np.squeeze(det[i]))
                            else:
                                bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                                img_center = img_size / 2
                                offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                     (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                                index = np.argmax(
                                    bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                                det_arr.append(det[index, :])
                        else:
                            det_arr.append(np.squeeze(det))

                        for i, det in enumerate(det_arr):
                            det = np.squeeze(det)
                            bb = np.zeros(4, dtype=np.int32)
                            bb[0] = np.maximum(det[0] - margin / 2, 0)
                            bb[1] = np.maximum(det[1] - margin / 2, 0)
                            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                            # scaled = misc.imresize(cropped, (image_size, image_size),
                            #                        interp='bilinear')
                            scaled = np.array(Image.fromarray(cropped).resize((image_size, image_size), resample=2))
                            nrof_successfully_aligned += 1
                            filename_base, file_extension = os.path.splitext(output_filename)
                            if detect_multiple_faces:
                                output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                            else:
                                output_filename_n = "{}{}".format(filename_base, file_extension)
                            imageio.imwrite(output_filename_n, scaled)
                    else:
                        print('Unable to align "%s"' % image_path)
    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)


def process_this_160(input_dir, output_dir):
    random_order = True
    detect_multiple_faces = False
    margin = 32
    image_size = 160
    sleep(random.random())
    # output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = os.listdir(input_dir)

    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)  # 0.25
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    nrof_images_total = 0
    nrof_successfully_aligned = 0
    if random_order:
        random.shuffle(dataset)  # 打乱原始顺序

    for image in dataset:     #image表示图片本身
        nrof_images_total += 1
        output_filename = os.path.join(output_dir, str(image))
        print(output_filename)
        # if not xx 用于判断xx是否为None，当xx为None时执行  也就是说对于已经对齐过的图片不再重新对齐
        if not os.path.exists(output_filename):
            try:
                # img = misc.imread(image)
                img = imageio.imread(input_dir + '/' + str(image))
            except (IOError, ValueError, IndexError) as e:
                errorMessage = '{}: {}'.format(image, e)
                print(errorMessage)
            else:
                if img.ndim < 2:
                    print('Unable to align "%s"' % image)
                    continue
                if img.ndim == 2:
                    img = facenet.to_rgb(img)
                img = img[:, :, 0:3]

                bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold,
                                                            factor)
                nrof_faces = bounding_boxes.shape[0]
                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    det_arr = []
                    img_size = np.asarray(img.shape)[0:2]
                    if nrof_faces > 1:
                        if detect_multiple_faces:
                            for i in range(nrof_faces):
                                det_arr.append(np.squeeze(det[i]))
                        else:
                            bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                            img_center = img_size / 2
                            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                            index = np.argmax(
                                bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                            det_arr.append(det[index, :])
                    else:
                        det_arr.append(np.squeeze(det))

                    for i, det in enumerate(det_arr):
                        det = np.squeeze(det)
                        bb = np.zeros(4, dtype=np.int32)
                        bb[0] = np.maximum(det[0] - margin / 2, 0)
                        bb[1] = np.maximum(det[1] - margin / 2, 0)
                        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                        # scaled = misc.imresize(cropped, (image_size, image_size),
                        #                        interp='bilinear')
                        scaled = np.array(Image.fromarray(cropped).resize((image_size, image_size), resample=2))
                        nrof_successfully_aligned += 1
                        filename_base, file_extension = os.path.splitext(output_filename)
                        if detect_multiple_faces:
                            output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                        else:
                            output_filename_n = "{}{}".format(filename_base, file_extension)
                        # misc.imsave(output_filename_n, scaled)
                        imageio.imwrite(output_filename_n, scaled)
                else:
                    print('Unable to align "%s"' % image)

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

#
# if __name__ == '__main__':
#     process_all_160('D:/Py_projects/final_facenet/data/images','D:/Py_projects/final_facenet/data/images_160')
#
