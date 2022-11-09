# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Contains the definition of the Inception Resnet V1 architecture.
As described in http://arxiv.org/abs/1602.07261.
  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

import math

import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, Concatenate, BatchNormalization, DepthwiseConv2D, Lambda, Reshape, Layer, Activation, add
from math import ceil




def SEModule(inputs, filters, ratio):
    x = inputs
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, int(x.shape[1])))(x)
    x = Conv2D(int(filters / ratio), (1, 1), strides=(1, 1), padding='same', use_bias=False, activation=None)(x)
    x = Activation('relu')(x)
    x = Conv2D(int(filters), (1, 1), strides=(1, 1), padding='same', use_bias=False, activation=None)(x)
    excitation = Activation('hard_sigmoid')(x)
    x = inputs * excitation

    return x


def GhostModule(inputs, out, ratio, convkernel, dwkernel,stride):
    x = inputs
    conv_out_channel = ceil(out * 1.0 / ratio)
    x = Conv2D(int(conv_out_channel), (convkernel, convkernel), use_bias=False,
               strides=(1, 1), padding='same', activation=None)(x)
    if ratio == 1:
        return x
    else:
        dw = DepthwiseConv2D(dwkernel, 1, padding='same', use_bias=False,
                             depth_multiplier=ratio - 1, activation=None)(x)
        dw = dw[:, :, :, :int(out - conv_out_channel)]
        output = Concatenate()([x, dw])
        return output

def GhostModule_v(inputs, out, ratio, convkernel, dwkernel,stride,padding):
    x = inputs
    conv_out_channel = ceil(out * 1.0 / ratio)
    if stride==2:
        strides = (2, 2)
    else:
        strides = (1, 1)
    if padding == 2:
        padding = 'VALID'
    else:
        padding = 'same'
    x = Conv2D(int(conv_out_channel), (convkernel, convkernel), use_bias=False,
               strides=strides, padding=padding, activation=None)(x)
    if ratio == 1:
        return x
    else:
        dw = DepthwiseConv2D(dwkernel, 1, padding='same', use_bias=False,
                             depth_multiplier=ratio - 1, activation=None)(x)

        dw = dw[:, :, :, :int(out - conv_out_channel)]
        # print(x.shape)
        # print(dw.shape)
        output = Concatenate()([x, dw])
        print(output.shape)
        return output

# def GBNeck(inputs, dwkernel, strides, exp, out, ratio, use_se):
#     x = DepthwiseConv2D(dwkernel, strides, padding='same', depth_multiplier=ratio - 1,
#                         activation=None, use_bias=False)(inputs)
#     x = BatchNormalization()(x)
#     x = Conv2D(out, (1, 1), strides=(1, 1), padding='same',
#                activation=None, use_bias=False)(x)
#     x = BatchNormalization()(x)
#     y = GhostModule(inputs, exp, ratio, 1, 3)
#     y = BatchNormalization()(y)
#     y = Activation('relu')(y)
#     if strides > 1:
#         y = DepthwiseConv2D(dwkernel, strides, padding='same', depth_multiplier=ratio - 1,
#                             activation=None, use_bias=False)(y)
#         y = BatchNormalization()(y)
#         y = Activation('relu')(y)
#     if use_se:
#         SEModule(y, exp, ratio)
#     y = GhostModule(y, out, ratio, 1, 3)
#     y = BatchNormalization()(y)
#
#     return add([x, y])
#
#
#
# def GBNeck_v(inputs, dwkernel, strides, exp, out, ratio, use_se):
#     print(inputs.shape)
#     # x = DepthwiseConv2D(dwkernel, strides, padding='VALID', depth_multiplier=ratio - 1,   
#     #                     activation=None, use_bias=False)(inputs)
#     # x = BatchNormalization()(x)
#     # x = Conv2D(out, (1, 1), strides=(1, 1), padding='VALID', activation=None, use_bias=False)(x)
#     # x = BatchNormalization()(x)
#     x = GhostModule_v(inputs, exp, ratio, 1, 3)
#     print("...")
#     # print(y.shape)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     if strides > 1:
#         x = DepthwiseConv2D(dwkernel, strides, padding='VALID', depth_multiplier=ratio - 1,
#                             activation=None, use_bias=False)(x)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#     if use_se:
#         SEModule(x, exp, ratio)
#     y = GhostModule_v(inputs, out, ratio, 1, 3)
#     y = BatchNormalization()(y)
#     # print(x.shape)
#     print(y.shape)
#     # print((add([x, y]).shape))
#     print("**********************")
#     return add([x, y])



def cbam_block(input_feature, name, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    with tf.variable_scope(name):
        attention_feature = channel_attention(input_feature, 'ch_at', ratio)
        attention_feature = spatial_attention(attention_feature, 'sp_at')
        print("CBAM Hello")
    return attention_feature


def channel_attention(input_feature, name, ratio=8):
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.variable_scope(name):
        channel = input_feature.get_shape()[-1]
        avg_pool = tf.reduce_mean(input_feature, axis=[1, 2], keepdims=True)

        assert avg_pool.get_shape()[1:] == (1, 1, channel)
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                   units=channel // ratio,
                                   activation=tf.nn.relu,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   name='mlp_0',
                                   reuse=None)
        assert avg_pool.get_shape()[1:] == (1, 1, channel // ratio)
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                   units=channel,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   name='mlp_1',
                                   reuse=None)
        assert avg_pool.get_shape()[1:] == (1, 1, channel)

        max_pool = tf.reduce_max(input_feature, axis=[1, 2], keepdims=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)
        max_pool = tf.layers.dense(inputs=max_pool,
                                   units=channel // ratio,
                                   activation=tf.nn.relu,
                                   name='mlp_0',
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel // ratio)
        max_pool = tf.layers.dense(inputs=max_pool,
                                   units=channel,
                                   name='mlp_1',
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)

        scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')

    return input_feature * scale


def spatial_attention(input_feature, name):
    kernel_size = 7
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    with tf.variable_scope(name):
        avg_pool = tf.reduce_mean(input_feature, axis=[3], keepdims=True)
        assert avg_pool.get_shape()[-1] == 1
        max_pool = tf.reduce_max(input_feature, axis=[3], keepdims=True)
        assert max_pool.get_shape()[-1] == 1
        concat = tf.concat([avg_pool, max_pool], 3)
        assert concat.get_shape()[-1] == 2

        concat = tf.layers.conv2d(concat,
                                  filters=1,
                                  kernel_size=[kernel_size, kernel_size],
                                  strides=[1, 1],
                                  padding="same",
                                  activation=None,
                                  kernel_initializer=kernel_initializer,
                                  use_bias=False,
                                  name='conv')
        assert concat.get_shape()[-1] == 1
        concat = tf.sigmoid(concat, 'sigmoid')

    return input_feature * concat


# Inception-A   96->64 64->32
def block_inception_a(inputs, scope=None, reuse=None):
  """Builds Inception-A block for Inception v4 network."""
  # By default use stride=1 and SAME padding
  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='SAME'):
    with tf.variable_scope(
        scope, 'BlockInceptionA', [inputs], reuse=reuse):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(inputs, 64, [1, 1], scope='Conv2d_0a_1x1')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(inputs, 32, [1, 1], scope='Conv2d_0a_1x1')
        branch_1 = slim.conv2d(branch_1, 64, [1, 3], scope='Conv2d_0b_1x3')
        branch_1 = slim.conv2d(branch_1, 64, [3, 1], scope='Conv2d_0c_3x1')
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.conv2d(inputs, 32, [1, 1], scope='Conv2d_0a_1x1')
        branch_2 = slim.conv2d(branch_2, 64, [7, 1], scope='Conv2d_0b_7x1')
        branch_2 = slim.conv2d(branch_2, 64, [1, 7], scope='Conv2d_0c_1x7')
      with tf.variable_scope('Branch_3'):
        branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
        branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
      return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])




# Inception-Resnet-A
def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 35x35 resnet block."""
    with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 32, 3, scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 32, 3, scope='Conv2d_0c_3x3')
        mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        cbam =cbam_block(up, '_cbam_block', ratio=8)
        net += scale * cbam
        # net = net + net1
        if activation_fn:
            net = activation_fn(net)
    return net

# Inception-Resnet-B
def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 17x17 resnet block."""
    with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 128, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 128, [1, 7],
                                        scope='Conv2d_0b_1x7')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 128, [7, 1],
                                        scope='Conv2d_0c_7x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        cbam =cbam_block(up, '_cbam_block', ratio=8)
        net += scale * cbam

        # net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


# Inception-Resnet-C
def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 8x8 resnet block."""
    with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 192, [1, 3],
                                        scope='Conv2d_0b_1x3')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [3, 1],
                                        scope='Conv2d_0c_3x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        cbam =cbam_block(up, '_cbam_block', ratio=8)
        net += scale * cbam

        # net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net

def reduction_a(net, k, l, m, n):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(net, n, 3, stride=2, padding='VALID',
                                 scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
        tower_conv1_0 = slim.conv2d(net, k, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1_0, l, 3,
                                    scope='Conv2d_0b_3x3')
        tower_conv1_2 = slim.conv2d(tower_conv1_1, m, 3,
                                    stride=2, padding='VALID',
                                    scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
    net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)
    return net

def reduction_b(net):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                   padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
        tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1, 256, 3, stride=2,
                                    padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv2_1 = slim.conv2d(tower_conv2, 256, 3,
                                    scope='Conv2d_0b_3x3')
        tower_conv2_2 = slim.conv2d(tower_conv2_1, 256, 3, stride=2,
                                    padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_3'):
        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
    net = tf.concat([tower_conv_1, tower_conv1_1,
                        tower_conv2_2, tower_pool], 3)
    return net

def inference(images, keep_probability, phase_train=True,
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=slim.initializers.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return inception_resnet_v1(images, is_training=phase_train,
              dropout_keep_prob=keep_probability, bottleneck_layer_size=bottleneck_layer_size, reuse=reuse)


def inception_resnet_v1(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None,
                        scope='InceptionResnetV1'):
    """Creates the Inception Resnet V1 model.
    Args:
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
      num_classes: number of predicted classes.
      is_training: whether is training or not.
      dropout_keep_prob: float, the fraction to keep before final layer.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
    Returns:
      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """
    end_points = {}

    with tf.variable_scope(scope, 'InceptionResnetV1', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):

                # # 79 x 79 x 32
                # net = _ghost_bottleneck_V(inputs, 32, 16, (3, 3), strides=2, ratio=2, squeeze=False, block_id=0, sub_block_id=0)
                # end_points['Conv2d_1a_3x3'] = net
                # # 77 x 77 x 32
                # net = _ghost_bottleneck_V(net, 32, 16, (3, 3), strides=1, ratio=2, squeeze=False, block_id=0,
                #                           sub_block_id=0)
                # end_points['Conv2d_2a_3x3'] = net
                # # 77 x 77 x 64
                # net = _ghost_bottleneck(net, 64, 16, (3, 3), strides=1, ratio=2, squeeze=False, block_id=0,
                #                           sub_block_id=0)
                # end_points['Conv2d_2b_3x3'] = net
                # # 38 x 38 x 64
                # net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                #                       scope='MaxPool_3a_3x3')
                # end_points['MaxPool_3a_3x3'] = net
                # # 38 x 38 x 80
                # net = _ghost_bottleneck_V(net, 80, 16, (3, 3), strides=1, ratio=2, squeeze=False, block_id=0,
                #                           sub_block_id=0)
                # end_points['Conv2d_3b_1x1'] = net
                # # 36 x 36 x 192
                # net = _ghost_bottleneck_V(net, 192, 16, (3, 3), strides=1, ratio=2, squeeze=False, block_id=0,
                #                           sub_block_id=0)
                # end_points['Conv2d_4a_3x3'] = net
                # # 17 x 17 x 256
                # net = _ghost_bottleneck_V(net, 32, 16, (3, 3), strides=2, ratio=2, squeeze=False, block_id=0,
                #                           sub_block_id=0)
                # end_points['Conv2d_4b_3x3'] = net




                # # 79 x 79 x 32
                # # net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID',
                # #                   scope='Conv2d_1a_3x3')
                # net = GBNeck_v(inputs, 3, 2, 16, 32, 2, False)
                # end_points['Conv2d_1a_3x3'] = net
                # # 77 x 77 x 32
                # # net = slim.conv2d(net, 32, 3, padding='VALID',
                # #                   scope='Conv2d_2a_3x3')
                # net = GBNeck_v(net, 3, 1, 16, 32, 2, False)
                # end_points['Conv2d_2a_3x3'] = net
                # # 77 x 77 x 64
                # # net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
                # net = GBNeck(net, 3, 1, 16, 64, 2, False)
                # end_points['Conv2d_2b_3x3'] = net
                # # 38 x 38 x 64
                # net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                #                       scope='MaxPool_3a_3x3')
                # end_points['MaxPool_3a_3x3'] = net
                # # 38x 38 x 80
                # # net = slim.conv2d(net, 80, 1, padding='VALID',
                # #                   scope='Conv2d_3b_1x1')
                # net = GBNeck_v(net, 3, 1, 16, 80, 2, False)
                # end_points['Conv2d_3b_1x1'] = net
                # # 36 x 36 x 192
                # # net = slim.conv2d(net, 192, 3, padding='VALID',
                # #                   scope='Conv2d_4a_3x3')
                # net = GBNeck_v(net, 3, 1, 16, 192, 2, False)
                # end_points['Conv2d_4a_3x3'] = net
                # # 17 x 17 x 256
                # # net = slim.conv2d(net, 256, 3, stride=2, padding='VALID',
                # #                   scope='Conv2d_4b_3x3')
                # net = GBNeck_v(net, 3, 2, 16, 256, 2, False)
                # end_points['Conv2d_4b_3x3'] = net

                # 79 x 79 x 32
                # net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID',
                #                   scope='Conv2d_1a_3x3')
                net = GhostModule_v(inputs, 32, 2, 3, 3, stride=2, padding=2)
                end_points['Conv2d_1a_3x3'] = net
                # 77 x 77 x 32
                # net = slim.conv2d(net, 32, 3, padding='VALID',
                #                   scope='Conv2d_2a_3x3')
                net = GhostModule_v(net, 77, 32, 3, 3, stride=1, padding=2)
                end_points['Conv2d_2a_3x3'] = net
                # 77 x 77 x 64
                # net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
                net = GhostModule_v(net, 64, 2, 3, 3, stride=1, padding=1)
                end_points['Conv2d_2b_3x3'] = net
                # 38 x 38 x 64
                net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                      scope='MaxPool_3a_3x3')
                end_points['MaxPool_3a_3x3'] = net
                # 38x 38 x 80
                # net = slim.conv2d(net, 80, 1, padding='VALID',
                #                   scope='Conv2d_3b_1x1')
                net = GhostModule_v(net, 80, 2, 1, 3, stride=1, padding=2)
                end_points['Conv2d_3b_1x1'] = net
                # 36 x 36 x 192
                # net = slim.conv2d(net, 192, 3, padding='VALID',
                #                   scope='Conv2d_4a_3x3')
                net = GhostModule_v(net, 192, 2, 3, 3, stride=1, padding=2)
                end_points['Conv2d_4a_3x3'] = net
                # 17 x 17 x 256
                # net = slim.conv2d(net, 256, 3, stride=2, padding='VALID',
                #                   scope='Conv2d_4b_3x3')
                net = GhostModule_v(net, 256, 2, 3, 3, stride=2, padding=2)
                end_points['Conv2d_4b_3x3'] = net

                #inception-A
                net = block_inception_a(net, 'Conv2d_4c_3x3')

                end_points['Conv2d_4c_3x3'] = net

                # 1 x Inception-resnet-A
                net = slim.repeat(net, 1, block35, scale=0.17)
                end_points['Mixed_5a'] = net

                # Reduction-A
                with tf.variable_scope('Mixed_6a'):
                    net = reduction_a(net, 192, 192, 256, 384)
                end_points['Mixed_6a'] = net

                # 2 x Inception-Resnet-B
                net = slim.repeat(net, 2, block17, scale=0.10)
                end_points['Mixed_6b'] = net

                # Reduction-B
                with tf.variable_scope('Mixed_7a'):
                    net = reduction_b(net)
                end_points['Mixed_7a'] = net

                # 1 x Inception-Resnet-C
                net = slim.repeat(net, 1, block8, scale=0.20)
                end_points['Mixed_8a'] = net

                net = block8(net, activation_fn=None)
                end_points['Mixed_8b'] = net

                with tf.variable_scope('Logits'):
                    end_points['PrePool'] = net
                    #pylint: disable=no-member
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                          scope='AvgPool_1a_8x8')
                    net = slim.flatten(net)

                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                       scope='Dropout')

                    end_points['PreLogitsFlatten'] = net

                net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None,
                        scope='Bottleneck', reuse=False)

    return net, end_points
