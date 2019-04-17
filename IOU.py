import LoadBatches_1
from keras.models import load_model
from Models import FCN32, FCN8, SegNet, UNet
import glob
import cv2
import numpy as np
import random

################
import matplotlib.pyplot as plt
########################

n_classes = 11

key = "segnet"

method = {
    "fcn32": FCN32.FCN32,
    "fcn8": FCN8.FCN8,
    "segnet": SegNet.SegNet,
    'unet': UNet.UNet}

images_path = "data/dataset1/images_prepped_test/"
segs_path = "data/dataset1/annotations_prepped_test/"

input_height = 320
input_width = 320

colors = [
    (random.randint(
        0, 255), random.randint(
            0, 255), random.randint(
                0, 255)) for _ in range(n_classes)]

##########################################################################


def label2color(colors, n_classes, seg):
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

########################

def save_imgs(self, epoch, gen_imgs):
    r, c = 2, 4

    colors = [
        (random.randint(
            0, 255), random.randint(
            0, 255), random.randint(
            0, 255)) for _ in range(11)]

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            pr = gen_imgs[cnt]
            pr = pr.reshape((320, 320, 11)).argmax(axis=2)
            axs[i, j].imshow(self.label2color(colors, 11, pr))
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig("/home/zhangzhichao/CNN/keras-segmentation-master/images/%d.png" % epoch)
    plt.close()
#######################


def getcenteroffset(shape, input_height, input_width):
    short_edge = min(shape[:2])
    xx = int((shape[0] - short_edge) / 2)
    yy = int((shape[1] - short_edge) / 2)
    return xx, yy


images = sorted(
    glob.glob(
        images_path +
        "*.jpg") +
    glob.glob(
        images_path +
        "*.png") +
    glob.glob(
        images_path +
        "*.jpeg"))
segmentations = sorted(glob.glob(segs_path + "*.jpg") +
                       glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.jpeg"))


# m = load_model("output/%s_model.h5" % key)
m = method[key](11, 320, 320)  # 有自定义层时，不能直接加载模型
m.load_weights("output/%s_model.h5" % key)



PA = 0
MPA = 0
SMIOU = 0
iter = 0
for i, (imgName, segName) in enumerate(zip(images, segmentations)):

    print("%d/%d %s" % (i + 1, len(images), imgName))


    im = cv2.imread(imgName, 1)

    im = plt.imread(imgName, 1)

    # im=cv2.resize(im,(input_height,input_width))
    xx, yy = getcenteroffset(im.shape, input_height, input_width)
    im = im[xx:xx + input_height, yy:yy + input_width, :]

    seg = cv2.imread(segName, 0)
    # seg= cv2.resize(seg,interpolation=cv2.INTER_NEAREST)
    seg = seg[xx:xx + input_height, yy:yy + input_width]

    pr = m.predict(np.expand_dims(LoadBatches_1.getImageArr(im), 0))[0]
    pr = pr.reshape((input_height, input_width, n_classes)).argmax(axis=2)

    pr = LoadBatches_1.getSegmentationArr(pr, 11, 320, 320)
    seg = LoadBatches_1.getSegmentationArr(seg, 11, 320, 320)

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

