from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Merge, multiply, Embedding
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
import tensorflow as tf
from keras.backend import tensorflow_backend

from scipy import misc
import os
import cv2
import numpy as np

np.random.seed(0)
np.random.RandomState(0)
tf.set_random_seed(0)

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

root_dir = "multiclass"


class CGAN():
    def __init__(self, input_width, input_height, channel, output_width, output_height, z_dim, classes):
        self.input_width = input_width
        self.input_height = input_height
        self.channel = channel
        self.output_width = output_width
        self.output_height = output_height
        self.z_dim = z_dim
        self.classes = classes

        self.image_shape = (self.input_width, self.input_width, channel)
        self.noise_shape = (self.z_dim, )
        self.label_shape = (self.classes, )

        optimizer = Adam(lr=0.0002, beta_1=0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.generator = self.build_generator()

        z = Input(shape=(self.z_dim,))
        label = Input(shape=(self.classes,))
        gen_img = self.generator([z, label])

        self.discriminator.trainable = False

        valid = self.discriminator([gen_img, label])

        self.combined = Model([z, label], valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(512 * self.output_width/8 * self.output_height/8, input_shape=self.noise_shape))
        model.add(Reshape((self.output_width/8, self.output_height/8, 512)))
        model.add(Activation("relu"))

        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(Activation("relu"))

        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))

        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))

        model.add(Conv2D(self.channel, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=self.noise_shape)
        label = Input(shape=self.label_shape)
        labels = Dense(self.z_dim, activation='relu')(label)
        input_noise_label = multiply([noise, labels])
        validity = model(input_noise_label)

        return Model([noise, label], validity)

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=self.image_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.image_shape)
        label = Input(shape=self.label_shape)
        labels = Dense(self.input_width*self.input_height*self.channel, activation='relu')(label)
        labels = Reshape((self.input_width, self.input_height, self.channel))(labels)
        input_img_label = multiply([img, labels])
        validity = model(input_img_label)

        return Model([img, label], validity)

    def build_combined(self):
        self.discriminator.trainable = False
        model = Sequential([self.generator, self.discriminator])

        return model

    def train(self, real_images, labels, iterations, batch_size=128, save_interval=50, check_noise=None, n=None):

        half_batch = int(batch_size/2)
        real_images = (real_images.astype(np.float32) - 127.5) / 127.5

        for iteration in range(iterations):

            # ------------------
            # Training Discriminator
            # -----------------
            idx = np.random.randint(0, real_images.shape[0], half_batch)

            train_real_images = real_images[idx]
            train_real_label = labels[idx]

            noise = np.random.uniform(-1, 1, (half_batch, self.z_dim))
            noise_label = np.random.randint(0, self.classes, (half_batch, 1))
            noise_label = self.one_hot_encode(noise_label, self.classes)

            gen_imgs = self.generator.predict([noise, noise_label])

            d_loss_real = self.discriminator.train_on_batch([train_real_images, train_real_label], np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, noise_label], np.zeros((half_batch, 1)))

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # -----------------
            # Training Generator
            # -----------------

            noise = np.random.uniform(-1, 1, (batch_size, self.z_dim))
            noise_label = np.random.randint(0, self.classes, (batch_size, 1))
            noise_label = self.one_hot_encode(noise_label, self.classes)

            g_loss = self.combined.train_on_batch([noise, noise_label], np.ones((batch_size, 1)))

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (iteration, d_loss[0], 100 * d_loss[1], g_loss))

            if iteration % save_interval == 0:
                self.save_imgs(iteration, check_noise, n, self.classes)

    def load_imgs(self, file_path, num_type):
        images = []
        labels = []
        for i in range(num_type):
            img_path = file_path + '/' + str(i)
            imgs = os.listdir(img_path)
            for img in imgs:
                img_names = os.path.join(img_path, img)
                img = cv2.imread(img_names)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.asarray(img, dtype='float32')
                img = cv2.resize(img, (64, 64))
                images.append(img)
                labels.append(i)
        images = np.array(images)
        labels = np.array(labels)
        return images, labels

    def one_hot_encode(self, labels, classes):
        l = labels.shape[0]
        one_hot_y = np.zeros((l, classes))
        for i in range(l):
            one_hot_y[i, labels[i]] = 1
        return one_hot_y

    def save_imgs(self, iteration, check_noise, n, classes):
        test_labels = np.random.randint(0, classes, (n, 1))
        test_labels = self.one_hot_encode(test_labels, classes)
        gen_imgs = self.generator.predict([check_noise, test_labels])
        for i in range(n):
            misc.imsave('ctrain/' + str(iteration) + str(i) + str(test_labels[i]) + '.png', gen_imgs[i])
        return

    def test(self, number, classes):
        test_noise = np.random.uniform(-1, 1, (number, 100))
        test_label = np.random.randint(0, classes, (number, 1))
        test_label = self.one_hot_encode(test_label)
        gen_imgs = self.generator.predict([test_noise, test_label])
        for i in range(number):
            misc.imsave('ctest/' + str(i) + str(test_label[i]) + '.png', gen_imgs[i])
        return


if __name__ == '__main__':
    root_dir = "flower_images"  # file path of training dataset
    input_height = 64  # the height of the input image
    input_width = 64  # the width of the input image
    input_channel = 3  # the channel of the input image
    output_height = 64  # the height of the output image
    output_width = 64  # the height of the output image
    z_dim = 100  # the dimension of the noise z
    cgan = CGAN(input_height, input_width, input_channel, output_height, output_width, z_dim)
    if not os.path.exists('ctrain'):
        os.makedirs('ctrain')
    if not os.path.exists('ctest'):
        os.makedirs('ctest')
    n = 16
    check_noise = np.random.uniform(-1, 1, (n, 100))
    real_images, real_labels = cgan.load_imgs(root_dir, 2)
    real_labels = cgan.one_hot_encode(real_labels, 2)
    cgan.train(real_images, real_labels, iterations=5000, batch_size=64, save_interval=10, check_noise=check_noise, n=n)
    cgan.test(100, 2)