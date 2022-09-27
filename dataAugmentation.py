import numpy as np
import matplotlib.pyplot as plt
import cv2
import userFunctions as div
import scipy.ndimage as scipy_image
import tensorflow as tf

#Test
#from trainer import Trainer

class OnlineAugmentation(object):
    def __init__(self):
        self.batch = None
        self.general_batch = {'im': [], 'label': []}
        self.im, self.label = 0, 0
        self.seed = np.random.randint(1234)
        self.original_size = (224, 224)

    def add_im(self, img, label):
        '''
        Add a batch composed of (image, label, label_weights) into the general batch dict
        :param batch: batch
        :return: None
        '''
        self.general_batch['im'].append(img.astype(np.float32))
        self.general_batch['label'].append(label.astype(np.float32))

    def create_database(self, demo_depth_label):
        self.depth, self.label = [], []

        for im, label in demo_depth_label:
            im, label = im.astype(np.uint8), label.astype(np.uint8)
            im = tf.image.resize(im, (224, 224), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR).numpy()
            label = tf.image.resize(label, (224, 224), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR).numpy()
            self.x_lim, self.y_lim, _ = im.shape
            x, y, w, h = detect_tool(im[:, :, 0])
            square_im, square_lab = create_sub_square(im, label, x, y, w, h)
            self.depth.append(square_im)
            self.label.append(square_lab)

    def get_pair(self):
        i = np.random.randint(len(self.depth))
        square_im, square_lab = self.depth[i], self.label[i]
        x = square_im.shape[0]
        xc, yc, angle = int((np.random.random()-0.5) * (self.x_lim-x) + self.x_lim//2), int((np.random.random()-0.5) * (self.y_lim-x) + self.y_lim//2), int(360 * np.random.random())
        bool_add_noise = np.random.random() < 0.8
        if bool_add_noise:
            noisy_square_im = add_noise(np.copy(square_im))
        else:
            noisy_square_im = np.copy(square_im)
        rot_plain_img, rot_plain_label = rotation_insert(noisy_square_im, square_lab, angle, xc, yc, self.x_lim, self.y_lim)
        if bool_add_noise:
            rot_plain_img = add_noise(rot_plain_img)
        rot_plain_label = change_ang_value(rot_plain_label, angle)

        flip_chance = np.random.randint(2)
        if flip_chance == 0:
            return rot_plain_img, rot_plain_label
        elif flip_chance == 1:
            up_down_flip_label = flip_ang_value(np.copy(rot_plain_label), 1)
            return np.flip(rot_plain_img, 0), np.flip(up_down_flip_label, 0)

    def generate_batch(self, im, label, mini, augmentation_factor=2, viz=False):
        '''Generate new images and label from one image/demonstration
        :param im: input image (numpy (224x224x3))
        :param label: label (numpy (224x224x3))
        :param augmentation_factor: number of data at the end of augmentation (3xaugmentation_factor³)
        :return: Batch of the augmented DataSet
        '''
        im, label = im.astype(np.uint8), label.astype(np.uint8)
        x, y, w, h = detect_tool(im[:, :, 0])
        square_im, square_lab = create_sub_square(im, label, x, y, w, h)

        x_lim, y_lim, _ = im.shape
        for i in range(0, augmentation_factor**3):
            xc, yc, angle, zoom = int(np.random.random()*x_lim), int(np.random.random()*y_lim), int(360*np.random.random()), (np.random.random()-0.5)/5+1
            bool_add_noise = np.random.random() < 0.8
            if bool_add_noise:
                noisy_square_im = add_noise(np.copy(square_im))
            else:
                noisy_square_im = np.copy(square_im)

            rot_plain_img, rot_plain_label = rotation_insert(noisy_square_im, square_lab, angle, xc, yc, x_lim, y_lim)
            if bool_add_noise:
                rot_plain_img = add_noise(rot_plain_img)
            rot_plain_label = change_ang_value(rot_plain_label, angle)
            self.add_im(rot_plain_img, rot_plain_label)

            up_down_flip_label = flip_ang_value(np.copy(rot_plain_label), 1)
            self.add_im(np.flip(rot_plain_img, 0), np.flip(up_down_flip_label, 0))

            left_right_flip_label = flip_ang_value(np.copy(rot_plain_label), 2)
            self.add_im(np.flip(rot_plain_img, 1), np.flip(left_right_flip_label, 1))
            # plt.imshow(np.flip(left_right_flip_label, 1))
            # plt.show()
            both_flip_label = flip_ang_value(np.copy(rot_plain_label), 3)
            self.add_im(np.flip(np.flip(rot_plain_img, 1), 0), np.flip(np.flip(both_flip_label, 1), 0))

            # PETIT TRAVAIL A FAIRE SI ON VEUT RAJOUTER L'ANGLE
            if viz:
                plt.subplot(2, 2, 1)
                plt.imshow(rot_plain_img)
                plt.subplot(2, 2, 2)
                plt.imshow(rot_plain_label)

                plt.subplot(2, 2, 3)
                plt.imshow(np.flip(np.flip(rot_plain_img, 1), 0))
                plt.subplot(2, 2, 4)
                plt.imshow(np.flip(np.flip(rot_plain_label, 1), 0))
                plt.show()
        return self.general_batch

def add_noise(im):
    for i in range(5):
        x, y = int(np.random.random() * im.shape[0]), int(np.random.random() * im.shape[1])
        im[max(0, x - int(np.random.random()*10)):min(im.shape[0], x + int(np.random.random()*10)),
        max(0, y - int(np.random.random()*10)):min(im.shape[0], y + int(np.random.random()*10))] = 0
    return im

def detect_tool(im):
    contours, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    height, width = im.shape
    (x, y, w, h) = cv2.boundingRect(contours[np.argmax([len(i) for i in contours])])  # Sélectionne le contour le plus long
    x, y, w, h = div.evenisation(x - 10), div.evenisation(y - 10), div.evenisation(w + 10), div.evenisation(h + 10)
    viz = False
    if viz:
        plt.subplot(1, 2, 1)
        plt.imshow(im)
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        plt.subplot(1, 2, 2)
        plt.imshow(im)
        plt.show()
    return x, y, w, h

def create_sub_square(im, label, x, y, w, h):
    # Zoom des parties intéressantes à augmenter
    zoom_im, zoom_lab = im[y:y+h, x:x+w, :], label[y:y+h, x:x+w, :]
    # Mise dans un carré permettant des rotations sans aucun troncage sqrt(2)*c
    great_length = div.evenisation(int(np.sqrt(2)*np.max(zoom_im.shape)) + 1)

    square_im, square_lab = np.zeros((great_length, great_length, 3)),\
                            np.zeros((great_length, great_length, 3), dtype=np.int)


    square_im[great_length//2-h//2:great_length//2+h//2, great_length//2-w//2:great_length//2+w//2, :] = zoom_im[:, :, :]

    square_lab[great_length//2-h//2:great_length//2+h//2, great_length//2-w//2:great_length//2+w//2, :] = zoom_lab[:, :, :]
    viz = False
    if viz:
        plt.subplot(1, 2, 1)
        plt.imshow(square_im)
        plt.subplot(1, 2, 2)
        plt.imshow(square_lab)
        plt.show()
    return square_im, square_lab

def insert_demo(img, xc, yc, x_lim, y_lim):
    '''
    :param img: XX.XX.3 numpy array
    :return: img inserted into another bigger frame
    '''
    plain_img = np.zeros((x_lim, y_lim, 3))
    great_length, c = img.shape[0], img.shape[0] // 2
    plain_img[max(0, xc - c):min(x_lim, xc + c), max(0, yc - c):min(y_lim, yc + c), :] = img[max(0, c - xc):min(
        x_lim + c - xc, great_length), max(0, c - yc):min(y_lim + c - yc, great_length), :]
    return plain_img

def rotation_insert(img, label, angle, xc, yc, x_lim, y_lim):
    label_pos, label_neg = np.copy(label[:, :, 0]), np.copy(label[:, :, 0])

    label_pos[label_pos == 255] = 0
    label_pos[label_pos != 0] = 255
    label_neg[label_neg != 255] = 0

    # label_pos, label_neg = label_pos.astype(np.uint32), label_neg.astype(np.uint32)
    label_pos = scipy_image.rotate(label_pos, angle, reshape=False)
    label_neg = scipy_image.rotate(label_neg, angle, reshape=False)

    rot_img, rot_label = scipy_image.rotate(img.astype(np.uint8), angle, reshape=False), scipy_image.rotate(label, angle,
                                                                                                            reshape=False)
    rot_label[:, :, 0] = np.zeros(rot_label[:, :, 0].shape)
    label_neg, label_pos = label_neg.astype(np.int32), label_pos.astype(np.int32)
    label_pos[label_pos > 0] = 1
    label_pos[label_pos < 0] = 1
    label_neg[label_neg != 0] = 255

    rot_label = rot_label.astype(np.int32)
    rot_label[:, :, 0] = label_pos + label_neg
    rot_img, rot_label = insert_demo(rot_img, xc, yc, x_lim, y_lim), insert_demo(rot_label, xc, yc, x_lim, y_lim)

    return rot_img.astype(np.int32), rot_label.astype(np.int32)

def change_ang_value(label, ang):

    square_lab_ang = label[:, :, 1]

    square_lab_ang[np.where(square_lab_ang != 0)] += ang
    label[:, :, 1] = square_lab_ang%180
    return label

def flip_ang_value(label, thetype):
    '''
    :param label:
    :param thetype: 1 : up and down
                    2 : left to right
                    3 : both
    '''
    if thetype == 1:
        label[:, :, 1] = -label[:, :, 1]

    if thetype == 2:
        label[:, :, 1] = 180 - label[:, :, 1]
    if thetype == 3:
        label[:, :, 1] = 180 + label[:, :, 1]
    label[:, :, 1] = label[:, :, 1] % 180
    return label

