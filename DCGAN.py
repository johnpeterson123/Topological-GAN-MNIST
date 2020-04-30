import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose, Conv2D, Dropout, Flatten
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import numpy as numpy
import matplotlib.pyplot as plt 
import time
import os

#Load data - prepare training data and normalize images to -1, 1
(train_images, train_labels), (temp_images, temp_labels) = mnist.load_data()
Train_images = np.append(train_images, temp_images, axis=0)
Train_labels = np.append(train_labels, temp_labels, axis=0)
Train_images = Train_images.reshape(Train_images.shape[0], 28, 28, 1).astype('float32')
Train_images = (Train_images - 127.5) / 127.5
Buffer_Size = 700000
Batch_Size = 256
Train_Data_Set = tf.data.Dataset.from_tensor_slices(Train_images).shuffle(Buffer_Size).batch(Batch_Size)

def Build_Generator_Model():
    '''
    Generator model:
    Input: random seed
    Upsamples to produce an image 
    Output: 28x28x1 image
    '''
    model = Sequential()
    model.add(Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)

    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model 

def Build_Discriminator_Model():
    '''
    ConvNet Image Classifier
    '''
    model = Sequential()
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))

    return model 


cross_entropy = BinaryCrossentropy(from_logits=True)

def Generator_Loss(fake_output):
    '''
    Define loss function for Generator Model 
    Discriminator output for each image is compared to an array of 1's 
    '''
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def Discriminator_Loss(real_output, fake_output):
    '''
    Defines loss function for Discriminator Model
    1 is assigned to Real Images
    0 is assigned to Fake Images
    Output: Sum Total Loss
    '''
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


Generator_Optimizer = Adam(1e-4)
Discriminator_Optimizer = Adam(1e-4)

EPOCHS = 50
NOISE_DIM = 100
Num_Images_to_Generate = 10
seed = tf.random.normal([Num_Images_to_Generate, NOISE_DIM])

@tf.function
def Train_Step(images):
    noise = tf.random.normal([Batch_Size, NOISE_DIM])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        Generated_Images = Generator(noise, training=True)
        real_output = Discriminator(images, training=True)
        fake_output = Discriminator(Generated_Images, training=True)
        Gen_Loss = Generator_Loss(fake_output)
        Disc_Loss = Discriminator_Loss(Generated_Images, training=True)

    Generator_Grads = gen_tape.gradient(Gen_Loss, Generator.trainable_variables)
    Discriminator_Grads = disc_tape.gradient(Disc_Loss, Discriminator.trainable_variables)
    Generator_Optimizer.apply_gradients(zip(Generator_Grads, Generator.trainable_variables))
    Discriminator_Optimizer.apply_gradients(zip(Discriminator_Grads, Discriminator.trainable_variables))


def Generate_and_Save_Images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, : , 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def Train(Dataset, epochs):
    '''
    Train the DCGAN on a Dataset for a given number of epochs
    '''
    for epoch in range(epochs):
        start = time.time()

        for image_batch in Dataset:
            Train_Step(image_batch)
        
        Generate_and_Save_Images(Generator, epoch + 1, seed)

        print(f'Time for Epoch: {epoch + 1} is {time.time() - start}')
    Generate_and_Save_Images(Generator, epochs, seed)

Generator = Build_Generator_Model()
Discriminator = Build_Discriminator_Model()
Train(Train_Data_Set, EPOCHS)