#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from keras import backend as K
from keras.layers import *
from keras.models import Model
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm


# In[ ]:


class MaxPoolingWithArgmax2D(Layer):

    def __init__(
            self,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
            **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
#         print("max pooling with argmax")
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == 'tensorflow':
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = tf.nn.max_pool_with_argmax(
                    inputs,
                    ksize=ksize,
                    strides=strides,
                    padding=padding)
        else:
            errmsg = '{} backend is not supported for layer {}'.format(
                    K.backend(), type(self).__name__)
            raise NotImplementedError(errmsg)
        argmax = tf.cast(argmax, K.floatx())
        return [output, argmax]
    
    def compute_output_shape(self, input_shape):
#         print("i guess its subsampling")
        ratio = (1, 2, 2, 1)
        output_shape = [
                dim//ratio[idx]
                if dim is not None else None
                for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]
    
class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        # one is pool and one is mask
        updates, mask = inputs[0], inputs[1]

        mask = tf.cast(mask, 'int32')
        input_shape = tf.shape(updates, out_type='int32')

        #  calculation new shape
        if output_shape is None:
            output_shape = (
                input_shape[0],
                input_shape[1]*self.size[0],
                input_shape[2]*self.size[1],
                input_shape[3])
        self.output_shape1 = output_shape

        # calculation indices for batch, height, width and feature maps
        one_like_mask = tf.ones_like(mask, dtype='int32')      #creates ones of the same shape as the mask
        batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], axis=0)
        batch_range = tf.reshape(tf.range(output_shape[0], dtype='int32'),shape=batch_shape)
        b = one_like_mask * batch_range

        y = mask // (output_shape[2] * output_shape[3])

        x = (mask // output_shape[3]) % output_shape[2]

        feature_range = tf.range(output_shape[3], dtype='int32')

        f = one_like_mask * feature_range

        # transpose indices & reshape update values to one dimension
        updates_size = tf.size(updates)       # Prints the number of elements in the updates
        indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
        values = tf.reshape(updates, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
                mask_shape[0],
                mask_shape[1]*self.size[0],
                mask_shape[2]*self.size[1],
                mask_shape[3]
                )


# In[ ]:


def usegnet_mod(n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, pool_size=(2, 2)):

    # encoder 
    
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)) 
    
    x = Conv2D(64, (3, 3), padding="same", kernel_initializer='he_normal', name="block1_conv1")(inputs) 
    x = BatchNormalization()(x) 
    x = Activation("relu")(x) 
    x = Conv2D(64, (3, 3), padding="same", kernel_initializer='he_normal', name="block1_conv2")(x) 
    x = BatchNormalization()(x) 
    skip = Activation("relu")(x)
    
    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size, name="block1_pool")(skip)
    
    x = Conv2D(128, (3, 3), padding="same" , kernel_initializer='he_normal', name="block2_conv1")(pool_1)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(128, (3, 3), padding="same", kernel_initializer='he_normal',  name="block2_conv2")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size, name="block2_pool")(x)
    
    x = Conv2D(256, (3, 3), padding="same" , kernel_initializer='he_normal', name="block3_conv1")(pool_2)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (3, 3), padding="same" , kernel_initializer='he_normal', name="block3_conv2")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (3, 3), padding="same" , kernel_initializer='he_normal', name="block3_conv3")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size, name="block3_pool")(x)
    
    #bottleneck
    x = Conv2D(256, (3, 3), padding="same", kernel_initializer='he_normal', name="block5_conv1")(pool_3)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (3, 3), padding="same" , kernel_initializer='he_normal',  name="block5_conv2")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (3, 3), padding="same" , kernel_initializer='he_normal',  name="block5_conv3")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
       
    #decoder
    unpool_1 = MaxUnpooling2D(pool_size)([x, mask_3])
    
    x = Conv2D(256, (3, 3), padding="same", kernel_initializer='he_normal')(unpool_1)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (3, 3), padding="same", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(128, (3, 3), padding="same", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    unpool_2 = MaxUnpooling2D(pool_size)([x, mask_2])
    
    x = Conv2D(128, (3, 3), padding="same", kernel_initializer='he_normal')(unpool_2)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(64, (3, 3), padding="same", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    unpool_3 = MaxUnpooling2D(pool_size)([x, mask_1])
    
    #concatenation (unpool_4, conv_2)
    merge = concatenate([unpool_3,skip], axis = -1)
    
    x = Conv2D(64, (1, 1), padding="same", kernel_initializer='he_normal')(merge)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(64, (3, 3), padding="same", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(n_classes, (1, 1), padding="same", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    outputs = Activation("softmax")(x)
         
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model


# In[ ]:


def get_model():
    return usegnet_mod(n_classes=8, IMG_HEIGHT=64, IMG_WIDTH=64, IMG_CHANNELS=3)

model = get_model()
from keras.optimizers import Adam
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', sm.metrics.iou_score])
model.summary()

