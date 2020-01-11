# Generative adversarial network project. Attachment for Master's thesis.
# Petro Mohyla Black Sea State University, Mykolaiv 2020.

# імпортуємо модулі та пакети
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from keras import initializers

# повідомляємо Keras, що ми використовуємо TensorFlow, як backend-рушій
os.environ["KERAS_BACKEND"] = "tensorflow"

# щоб переконатися, що ми зможемо відтворити експеримент та отримати 
# такі ж результати
np.random.seed(10)

# розмірність вектора випадкового шуму
random_dim = 100

def load_minst_data():
    # завантаження даних
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # нормалізуємо вхідні дані в діапазоні [-1, 1]
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    # перетворюємо x_train з формою (60000, 28, 28) у (60000, 784),
    # отримуємо 784 стовпчиків на рядок
    x_train = x_train.reshape(60000, 784)
    return (x_train, y_train, x_test, y_test)

# далі буде використано оптимізатор ADAM
def get_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)

def get_generator(optimizer):
    generator = Sequential()
    generator.add(Dense(256, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(784, activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return generator

def get_discriminator(optimizer):
    discriminator = Sequential()
    discriminator.add(Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator

def get_gan_network(discriminator, random_dim, generator, optimizer):
    # спочатку ми встановили trainable, як False, оскільки ми хочемо
    # тренувати тільки генератор або дискримінатор за раз
    discriminator.trainable = False
    # вхід (шум) GAN буде 100-мірними векторами
    gan_input = Input(shape=(random_dim,))
    # output генератора (зображення)
    x = generator(gan_input)
    # output дискримінатора (ймовірність чи зображення є реальним)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    return gan

# створюємо панель згенерованих MNIST зображень
def plot_generated_images(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image_epoch_%d.png' % epoch)

def train(epochs=1, batch_size=128):
    # отримуємо навчальні та тестові дані
    x_train, y_train, x_test, y_test = load_minst_data()
    # поділяємо навчальні дані на групи, розмірність 128
    batch_count = x_train.shape[0] // batch_size

    # конструюємо генеративно-змагальну мережу
    adam = get_optimizer()
    generator = get_generator(adam)
    discriminator = get_discriminator(adam)
    gan = get_gan_network(discriminator, random_dim, generator, adam)

    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(batch_count)):
            # отримуємо випадковий набір вхідного шуму та зображень
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            # генеруємо фейкові MNIST-зображення
            generated_images = generator.predict(noise)
            X = np.concatenate([image_batch, generated_images])

            # мітки для згенерованих та реальних даних
            y_dis = np.zeros(2*batch_size)
            # одностороннє згладжування міток
            y_dis[:batch_size] = 0.9

            # навчання дискримінатора
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            # навчання генератора
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)

        if e == 1 or e % 20 == 0:
            plot_generated_images(e, generator)

if __name__ == '__main__':
    train(500, 128)
