#!/usr/bin/env python3
# coding=utf-8

import argparse
import glob
import logging
import os
import os.path
import pickle

import cv2

import matplotlib.pyplot as plt
import numpy as np
from imutils import build_montages
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

from convautoencoder import ConvAutoencoder

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Supress matplotlib warnings
plt.rcParams.update({'figure.max_open_warning': 0})


def get_features(filelist, X):
    '''
    Get features from images
    '''
    for i, imagepath in enumerate(filelist):
        img = image.load_img(imagepath, target_size=(256, 256))
        img = image.img_to_array(img)
        img_arr = np.expand_dims(img, axis=-1)
        X.append(img_arr)
    return X


def train_and_plot_model(train_x, val_x):
    '''
    Train the conv-autoencoder model and plot training loss and accuracy curve.
    '''
    no_epochs = 20
    init_lr = 1e-3
    batch_size = 64

    # construct our convolutional autoencoder
    logger.info("Starting building of Conv-Autoencoder")
    autoencoder = ConvAutoencoder.build(256, 256, 3)
    autoencoder.compile(loss="mse", optimizer=Adam(lr=init_lr))

    # train the convolutional autoencoder
    H = autoencoder.fit(
        train_x, train_x,
        validation_data=(val_x, val_x),
        epochs=no_epochs,
        batch_size=batch_size)

    # plotting and saving the training history
    N = np.arange(0, no_epochs)
    plt.style.use('fivethirtyeight')
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot.png")

    # serialize the autoencoder model to disk
    logger.info("Saving model in .h5 format")
    autoencoder.save("autoenc2.h5", save_format="h5")


def construct_feature_vectors(train_x):
    '''
    Generate index of feature vectors
    '''
    # load our autoencoder from disk
    print("[INFO] loading autoencoder model...")
    autoencoder = load_model("autoenc2.h5")

    # create the encoder model which consists of *just* the encoder
    # portion of the autoencoder
    encoder = Model(inputs=autoencoder.input,
                    outputs=autoencoder.get_layer("encoded").output)

    # quantify the contents of our input images using the encoder
    print("[INFO] encoding images...")
    features = encoder.predict(train_x)

    # construct a dictionary that maps the index of the MNIST training
    # image to its corresponding latent-space representation
    indexes = list(range(0, train_x.shape[0]))
    data = {"indexes": indexes, "features": features}

    # write the data dictionary to disk
    print("[INFO] saving index...")
    f = open("features2.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()


def get_similar(neigh_fit, train_x, image, n_neighbors=5):
    autoencoder = load_model("autoenc2.h5")

    encoder = Model(inputs=autoencoder.input,
                    outputs=autoencoder.get_layer("encoded").output)

    code = encoder.predict(image[None])

    (distances,), (idx,) = neigh_fit.kneighbors(code, n_neighbors=n_neighbors)

    return distances, train_x[idx]


def show_similar(image, neigh_fit, train_x):

    distances, neighbors = get_similar(
        neigh_fit, train_x, image, n_neighbors=3)

    plt.figure(figsize=[8, 7])
    plt.subplot(4, 4, 1)
    plt.imshow(np.clip(image + 0.5, 0, 1))
    plt.title("Original image")

    for i in range(3):
        plt.subplot(4, 4, i + 2)
        plt.imshow(np.clip(neighbors[i] + 0.5, 0, 1))
        plt.title("Dist=%.3f" % distances[i])
    plt.show()


def nearest_neighbors(train_x):

    # load our autoencoder from disk
    logger.info("Loading autoencoder model...")
    autoencoder = load_model("autoenc2.h5")

    # create the encoder model which consists of *just* the encoder
    # portion of the autoencoder
    encoder = Model(inputs=autoencoder.input,
                    outputs=autoencoder.get_layer("encoded").output)

    # quantify the contents of our input images using the encoder
    logger.info("Encoding testing images")
    features = encoder.predict(train_x)

    nei_clf = NearestNeighbors(metric="euclidean", algorithm='kd_tree')
    nei_clf.fit(features)

    return nei_clf


def euclidean(src, dist):
    # compute and return the euclidean distance between two vectors
    return np.linalg.norm(src - dist)


def perform_search(query_features, index, max_results=100):
    # initialize our list of results
    results = []

    # loop over our index
    for i in range(0, len(index["features"])):
        # compute the euclidean distance between our query features
        # and the features for the current image in our index, then
        # update our results list with a 2-tuple consisting of the
        # computed distance and the index of the image
        d = euclidean(query_features, index["features"][i])
        results.append((d, i))

    # sort the results and grab the top ones
    results = sorted(results)[:max_results]

    # return the list of results
    return results


def img_search_retrieval(train_x, test_x, N):
    '''
    Retrieve the top N similar images
    '''

    # load the autoencoder model and index from disk
    logger.info("Loading autoencoder and index")

    autoencoder = load_model("autoenc2.h5")
    index = pickle.loads(open("features2.pickle", "rb").read())

    # create the encoder model which consists of *just* the encoder
    # portion of the autoencoder
    encoder = Model(inputs=autoencoder.input,
                    outputs=autoencoder.get_layer("encoded").output)

    # quantify the contents of our input testing images using the encoder
    logger.info("Encoding testing images")
    features = encoder.predict(test_x)

    # randomly sample a set of testing query image indexes
    query_index = list(range(0, test_x.shape[0]))
    query_index = np.random.choice(query_index, size=10,
                                   replace=False)
    # loop over the testing indexes
    for i in query_index:
        # take the features for the current image, find all similar
        # images in our dataset, and then initialize our list of result
        # images
        query_features = features[i]
        results = perform_search(query_features, index, max_results=225)
        images = []

        # loop over the results
        for (_, j) in results:
            # grab the result image, convert it back to the range
            # [0, 255], and then update the images list
            image = (train_x[j] * 255).astype("uint8")
            image = np.dstack([image])
            images.append(image)

        # display the query image
        query = (test_x[i] * 255).astype("uint8")
        cv2.imshow("Query", query)

        # build a montage from the results and display it
        if N % 2 == 0:
            montage = build_montages(images, (64, 64), (N // 2, N // 2))[0]
        else:
            montage = build_montages(images, (64, 64), (N // 2, N // 2 + 1))[0]
        cv2.imshow("Results", montage)
        cv2.waitKey(0)


def main():

    directory = args.classpath
    N = args.similar
    # DIR containing images
    train_dir = f'{directory}/train'
    val_dir = f'{directory}/val'
    test_dir = f'{directory}/test'

    logger.info("Loading dataset")

    # Loop over files and get features
    train_filelist = glob.glob(os.path.join(train_dir, '*.jpg'))
    val_filelist = glob.glob(os.path.join(val_dir, '*.jpg'))
    test_filelist = glob.glob(os.path.join(test_dir, '*.jpg'))
    train_filelist.sort()
    val_filelist.sort()
    test_filelist.sort()

    # To store the features
    train_x = []
    val_x = []
    test_x = []

    train_x = get_features(train_filelist, train_x)
    val_x = get_features(val_filelist, val_x)
    test_x = get_features(test_filelist, test_x)

    # Convert to array
    train_x = np.array(train_x)
    val_x = np.array(val_x)
    test_x = np.array(test_x)

    # Normalizing the pixel values between [0,1]
    train_x = train_x.astype("float32") / 255.0
    val_x = val_x.astype("float32") / 255.0
    test_x = test_x.astype("float32") / 255.0

    train_x = train_x.squeeze()
    val_x = val_x.squeeze()
    test_x = test_x.squeeze()

    logger.info("Done converting to arrays")
    logger.debug(
        f"Shape of train_x: {train_x.shape}, val_x: {val_x.shape} and test_x: {val_x.shape}")

    # Training the autoencoder and saving the model
    train_and_plot_model(train_x, val_x)

    # Constructing features vectors
    construct_feature_vectors(train_x)

    '''
    [TO-DO] Nearest Neighbors
    '''
    # neighbors_fit = nearest_neighbors(train_x)
    # show_similar(test_x, neighbors_fit, train_x)

    # Retrieving N similar images
    img_search_retrieval(train_x, test_x, N)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Path to the dataset containing train, val and test folders.")
    parser.add_argument(
        '-c',
        '--classpath',
        type=str,
        help='directory with list of classes',
        required=True)
    parser.add_argument(
        '-N',
        '--similar',
        type=int,
        default=10,
        help='Number of similar images to retrieve.')

    args = parser.parse_args()

    main()
