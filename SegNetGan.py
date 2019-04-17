import glob
from keras.layers import Input, Dense, Reshape, Flatten, concatenate
from keras.layers import BatchNormalization, Activation, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import LoadBatches
from keras import backend as K
import random
import cv2

#######
import tensorflow as tf
#######


from Models.utils import MaxUnpooling2D, MaxPoolingWithArgmax2D


class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 320
        self.img_cols = 320
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        # self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        base_generator = self.build_generator()
        base_discriminator = self.build_discriminator()
        ########
        self.generator = Model(
            inputs=base_generator.inputs,
            outputs=base_generator.outputs)

        self.discriminator = Model(
            inputs=base_discriminator.inputs,
            outputs=base_discriminator.outputs)
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        frozen_D = Model(
            inputs=base_discriminator.inputs,
            outputs=base_discriminator.outputs)
        frozen_D.trainable = False

        # combined
        z = Input(shape=(self.img_rows, self.img_cols, 3))

        img, seg = self.generator(z)
        valid = frozen_D([z, seg])
        self.combined = Model(inputs=z, outputs=[img, valid])
        self.combined.compile(optimizer=optimizer,
                              loss={'model_4': 'categorical_crossentropy', 'model_6': 'binary_crossentropy'})

    def build_generator(self):

        nClasses = 11
        input_height = 320
        input_width = 320
        img_input = Input(shape=(input_height, input_width, 3))

        # Block 1
        x = Conv2D(64, (3, 3),
                   activation='relu',
                   padding='same',
                   name='block1_conv1')(img_input)
        x = Conv2D(64, (3, 3),
                   activation='relu',
                   padding='same',
                   name='block1_conv2')(x)
        x, mask_1 = MaxPoolingWithArgmax2D(name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3),
                   activation='relu',
                   padding='same',
                   name='block2_conv1')(x)
        x = Conv2D(128, (3, 3),
                   activation='relu',
                   padding='same',
                   name='block2_conv2')(x)
        x, mask_2 = MaxPoolingWithArgmax2D(name='block2_pool')(x)

        # Block
        x = Conv2D(256, (3, 3),
                   activation='relu',
                   padding='same',
                   name='block3_conv1')(x)
        x = Conv2D(256, (3, 3),
                   activation='relu',
                   padding='same',
                   name='block3_conv2')(x)
        x = Conv2D(256, (3, 3),
                   activation='relu',
                   padding='same',
                   name='block3_conv3')(x)
        x, mask_3 = MaxPoolingWithArgmax2D(name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3),
                   activation='relu',
                   padding='same',
                   name='block4_conv1')(x)
        x = Conv2D(512, (3, 3),
                   activation='relu',
                   padding='same',
                   name='block4_conv2')(x)
        x = Conv2D(512, (3, 3),
                   activation='relu',
                   padding='same',
                   name='block4_conv3')(x)

        x, mask_4 = MaxPoolingWithArgmax2D(name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3),
                   activation='relu',
                   padding='same',
                   name='block5_conv1')(x)
        x = Conv2D(512, (3, 3),
                   activation='relu',
                   padding='same',
                   name='block5_conv2')(x)
        x = Conv2D(512, (3, 3),
                   activation='relu',
                   padding='same',
                   name='block5_conv3')(x)
        x, mask_5 = MaxPoolingWithArgmax2D(name='block5_pool')(x)

        Vgg_streamlined = Model(inputs=img_input, outputs=x)

        # 加载vgg16的预训练权重
        Vgg_streamlined.load_weights(
            r"./data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")

        # 解码层
        unpool_1 = MaxUnpooling2D()([x, mask_5])
        y = Conv2D(512, (3, 3), padding="same")(unpool_1)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)
        y = Conv2D(512, (3, 3), padding="same")(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)
        y = Conv2D(512, (3, 3), padding="same")(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)

        unpool_2 = MaxUnpooling2D()([y, mask_4])
        y = Conv2D(512, (3, 3), padding="same")(unpool_2)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)
        y = Conv2D(512, (3, 3), padding="same")(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)
        y = Conv2D(256, (3, 3), padding="same")(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)

        unpool_3 = MaxUnpooling2D()([y, mask_3])
        y = Conv2D(256, (3, 3), padding="same")(unpool_3)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)
        y = Conv2D(256, (3, 3), padding="same")(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)
        y = Conv2D(128, (3, 3), padding="same")(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)

        unpool_4 = MaxUnpooling2D()([y, mask_2])
        y = Conv2D(128, (3, 3), padding="same")(unpool_4)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)
        y = Conv2D(64, (3, 3), padding="same")(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)

        unpool_5 = MaxUnpooling2D()([y, mask_1])
        y = Conv2D(64, (3, 3), padding="same")(unpool_5)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)

        # 分类
        y = Conv2D(nClasses, (1, 1), padding="same")(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)

        y = Reshape((-1, nClasses))(y)
        y = Activation("softmax")(y)

        seg = Reshape((320, 320, 11))(y)
        model = Model(inputs=img_input, outputs=[y, seg])

        # model.summary()
        return model

    def build_discriminator(self):

        # 将原始图片进行处理
        branch1 = Input(shape=(320, 320, 3))
        b1 = Conv2D(16, kernel_size=3, padding="same")(branch1)
        b1 = BatchNormalization()(b1)
        b1 = Conv2D(64, kernel_size=3, padding="same")(b1)
        b1 = BatchNormalization()(b1)

        # 对生成的seg图进行处理
        branch2 = Input(shape=(320, 320, 11))
        b2 = Conv2D(64, kernel_size=3, padding="same")(branch2)
        b2 = BatchNormalization()(b2)

        # 将卷积后的原始图像与seg图像进行混合
        x = concatenate([b1, b2], axis=1)
        # 对混合后的张量进行处理
        x = Conv2D(128, kernel_size=3, padding="same")(x)
        x = MaxPooling2D()(x)
        x = Conv2D(256, kernel_size=3, padding="same")(x)
        x = MaxPooling2D()(x)
        x = Conv2D(512, kernel_size=3, padding="same")(x)

        x = Flatten()(x)
        x = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[branch1, branch2], outputs=x)
        return model

    def getceneroffset(self, shape, input_height, input_width):
        short_edge = min(shape[:2])
        xx = int((shape[0] - short_edge) / 2)
        yy = int((shape[1] - short_edge) / 2)
        return xx, yy

    def label2color(self, colors, n_classes, seg):
        seg_color = np.zeros((seg.shape[0], seg.shape[1], 3))
        for c in range(n_classes):
            seg_color[:, :, 0] += ((seg == c) *
                                   (colors[c][0])).astype('uint8')
            seg_color[:, :, 1] += ((seg == c) *
                                   (colors[c][1])).astype('uint8')
            seg_color[:, :, 2] += ((seg == c) *
                                   (colors[c][2])).astype('uint8')
        seg_color = seg_color.astype(np.uint8)
        return seg_color

    def predict(self):
        images_path = "data/dataset1/images_prepped_test/"
        segs_path = "data/dataset1/annotations_prepped_test/"

        images = sorted(glob.glob(images_path + "*.jpg") +
                        glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg"))
        segmentations = sorted(glob.glob(segs_path + "*.jpg") +
                               glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.jpeg"))

        m = Model(inputs=self.combined.inputs, outputs=self.combined.outputs)
        m.load_weights("output/my_model_weights.h5")

        PA = 0
        MPA = 0
        SMIOU = 0
        iter = 0
        for i, (imgName, segName) in enumerate(zip(images, segmentations)):
            print("%d/%d %s" % (i + 1, len(images), imgName))

            im = cv2.imread(imgName, 1)

            xx, yy = self.getceneroffset(im.shape, 320, 320)
            im = im[xx:xx + 320, yy:yy + 320, :]

            seg = cv2.imread(segName, 0)
            seg = seg[xx:xx + 320, yy:yy + 320]

            pr, _ = m.predict(np.expand_dims(LoadBatches.getImageArr(im), 0))
            pr = pr[0].reshape((320, 320, 11)).argmax(axis=2)
            pr = LoadBatches.getSegmentationArr(pr, 11, 320, 320)
            seg = LoadBatches.getSegmentationArr(seg, 11, 320, 320)

            # pixel acc
            count = 0.0
            for j in range(pr.shape[0]):
                if (pr[j] == seg[j]).all():
                    count += 1
            pa = count / 102400
            print(pa)
            PA += pa

            # mean pixel acc
            class_pixel_acc = []
            class_count = 0.0
            class_sum = 1.0
            for j in range(11):
                for k in range(pr.shape[0]):
                    if seg[k][j] == 1:
                        class_sum += 1
                    if (pr[k] == seg[k]).all() & (pr[k][j] == 1):
                        class_count += 1
                class_pixel_acc.append(class_count / (class_sum - 1))
            mpa = sum(class_pixel_acc) / len(class_pixel_acc)
            MPA += mpa

            # MIOU
            class_IOU = []
            class_count = 0.0
            class_sum = 0.0
            for j in range(11):
                for k in range(pr.shape[0]):
                    if seg[k][j] == 1:
                        class_sum += 1
                    if (pr[k] == seg[k]).all() & (pr[k][j] == 1):
                        class_count += 1
                class_IOU.append(class_count / (class_sum * 2 - class_count))
            MIOU = sum(class_IOU) / len(class_IOU)
            SMIOU += MIOU
            iter += 1
            print("PA:%.2f MPA:%.2f MIOU:%.2f" % (pa, mpa, MIOU))

        print("PA:%.2f MPA:%.2f MIOU:%.2f" % (PA / iter, MPA / iter, SMIOU / iter))


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.predict()


