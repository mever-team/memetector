from vit import (
    vit,
    vita,
    Patches,
    PatchEncoder,
    PositionalEmbedding,
)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model as load_keras_ckpt
from tensorflow.keras.callbacks import ModelCheckpoint
from split import split
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
from tensorflow import keras
from PIL import Image
import numpy as np
import cv2
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class LearningRateSchedule(keras.callbacks.Callback):
    def __init__(self, warmup_batches, init_lr, decay, verbose=0):
        super(LearningRateSchedule, self).__init__()
        self.warmup_batches = warmup_batches
        self.init_lr = init_lr
        self.decay = decay
        self.verbose = verbose
        self.batch_count = 0
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.batch_count = self.batch_count + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        if self.batch_count <= self.warmup_batches:
            lr = self.batch_count * self.init_lr / self.warmup_batches
            K.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print("\nBatch %05d, learning rate %s." % (self.batch_count + 1, lr))
        else:
            lr = self.init_lr / (
                1 + self.decay * self.batch_count * 1.001**self.batch_count
            )
            K.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print("\nBatch %05d, learning rate %s." % (self.batch_count + 1, lr))


class DataGenerator(Sequence):
    def __init__(self, files, isize, bsize, model_name, mean, var):
        self.files = (
            files  # list of tuples: [(filepath0, label0), (filepath1, label1), ...]
        )
        self.isize = isize  # target image size
        self.bsize = bsize  # batch size
        self.model_name = model_name
        self.mean = mean  # mean
        self.var = var  # var

    def __len__(self):
        return (np.ceil(len(self.files) / float(self.bsize))).astype(np.int)

    def __getitem__(self, idx):
        batch = self.files[idx * self.bsize : (idx + 1) * self.bsize]
        if self.model_name == "ViT":
            x = np.stack(
                [
                    cv2.resize(
                        (np.array(Image.open(x[0])) / 255 - self.mean)
                        / np.sqrt(self.var),
                        dsize=(self.isize, self.isize),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    for x in batch
                ]
            )
        elif self.model_name == "resnet":
            x = np.stack(
                [img_to_array(load_img(x[0], target_size=(224, 224))) for x in batch]
            )
        elif self.model_name == "vgg":
            x = np.stack(
                [img_to_array(load_img(x[0], target_size=(224, 224))) for x in batch]
            )
        elif self.model_name == "efficientnet":
            x = np.stack(
                [img_to_array(load_img(x[0], target_size=(456, 456))) for x in batch]
            )
        y = np.array([x[1] for x in batch])
        return x, y


def load_vgg():
    vgg = keras.applications.VGG16(
        include_top=False, pooling="avg", input_shape=(224, 224, 3)
    )
    for layer in vgg.layers:
        layer.trainable = False
    inputs = keras.layers.Input(shape=(224, 224, 3))
    preprocessed_inputs = keras.applications.vgg16.preprocess_input(inputs)
    embeddings = vgg(preprocessed_inputs)
    outputs = keras.layers.Dense(1, activation="sigmoid")(embeddings)
    vgg_new = keras.models.Model(inputs=inputs, outputs=outputs)
    return vgg_new


def load_resnet():
    resnet = keras.applications.resnet50.ResNet50(
        include_top=False, pooling="avg", input_shape=(224, 224, 3)
    )
    for layer in resnet.layers:
        layer.trainable = False
    inputs = keras.layers.Input(shape=(224, 224, 3))
    preprocessed_inputs = keras.applications.resnet50.preprocess_input(inputs)
    embeddings = resnet(preprocessed_inputs)
    outputs = keras.layers.Dense(1, activation="sigmoid")(embeddings)
    resnet_new = keras.models.Model(inputs=inputs, outputs=outputs)
    return resnet_new


def load_efficientnet():
    efficientnet = keras.applications.EfficientNetB5(
        include_top=False, pooling="avg", input_shape=(456, 456, 3)
    )
    for layer in efficientnet.layers:
        layer.trainable = False
    inputs = keras.layers.Input(shape=(456, 456, 3))
    preprocessed_inputs = keras.applications.efficientnet.preprocess_input(inputs)
    embeddings = efficientnet(preprocessed_inputs)
    outputs = keras.layers.Dense(1, activation="sigmoid")(embeddings)
    efficientnet_new = keras.models.Model(inputs=inputs, outputs=outputs)
    return efficientnet_new


stats = {
    "ConcepCap 0.00 TXT 0.00": {
        "mean": np.array([0.508, 0.474, 0.451]),
        "var": np.array([0.104, 0.101, 0.104]),
    },
    "ConcepCap 0.33 TXT 0.00": {
        "mean": np.array([0.502, 0.475, 0.448]),
        "var": np.array([0.099, 0.096, 0.101]),
    },
    "ConcepCap 0.33 TXT 0.33": {
        "mean": np.array([0.504, 0.475, 0.449]),
        "var": np.array([0.101, 0.098, 0.103]),
    },
    "ConcepCap 0.33 TXT 0.67": {
        "mean": np.array([0.509, 0.479, 0.453]),
        "var": np.array([0.103, 0.099, 0.104]),
    },
    "ConcepCap 0.33 TXT 1.00": {
        "mean": np.array([0.512, 0.482, 0.457]),
        "var": np.array([0.106, 0.102, 0.106]),
    },
    "ConcepCap 0.67 TXT 0.00": {
        "mean": np.array([0.500, 0.478, 0.450]),
        "var": np.array([0.095, 0.092, 0.099]),
    },
    "ConcepCap 0.67 TXT 0.33": {
        "mean": np.array([0.504, 0.480, 0.453]),
        "var": np.array([0.099, 0.095, 0.101]),
    },
    "ConcepCap 0.67 TXT 0.67": {
        "mean": np.array([0.508, 0.481, 0.454]),
        "var": np.array([0.103, 0.099, 0.105]),
    },
    "ConcepCap 0.67 TXT 1.00": {
        "mean": np.array([0.513, 0.485, 0.461]),
        "var": np.array([0.107, 0.103, 0.108]),
    },
    "ConcepCap 1.00 TXT 0.00": {
        "mean": np.array([0.496, 0.475, 0.443]),
        "var": np.array([0.092, 0.089, 0.098]),
    },
    "ConcepCap 1.00 TXT 0.33": {
        "mean": np.array([0.504, 0.482, 0.454]),
        "var": np.array([0.097, 0.094, 0.101]),
    },
    "ConcepCap 1.00 TXT 0.67": {
        "mean": np.array([0.508, 0.483, 0.457]),
        "var": np.array([0.103, 0.099, 0.105]),
    },
    "ConcepCap 1.00 TXT 1.00": {
        "mean": np.array([0.514, 0.487, 0.463]),
        "var": np.array([0.107, 0.103, 0.108]),
    },
}

img_size = 250
input_shape = (img_size, img_size, 3)
patch_size = 25
num_patches = (img_size // patch_size) ** 2

projection_dim = 64
transformer_layers = 8
transformer_units = [projection_dim * 2, projection_dim]
num_heads = 4
mlp_head_units = [2048, 1024]

batch_size = 64
learning_rate = 1e-3
weight_decay = 1e-3
warmup_batches = 532  # 532 batches = 10% of all batches
epochs = 20
decay = learning_rate / epochs

TRAIN, VAL, TEST = split()
scenarios = list(TRAIN.keys())

accuracy = np.zeros((len(scenarios), len(scenarios)))
print("ViT")
for i, scenario_i in enumerate(scenarios):
    print(f"Train: {scenario_i}")
    train = DataGenerator(
        TRAIN[scenario_i],
        img_size,
        batch_size,
        "ViT",
        stats[scenario_i]["mean"],
        stats[scenario_i]["var"],
    )
    val = DataGenerator(
        VAL[scenario_i],
        img_size,
        batch_size,
        "ViT",
        stats[scenario_i]["mean"],
        stats[scenario_i]["var"],
    )

    model = vit(
        input_shape,
        patch_size,
        projection_dim,
        num_patches,
        transformer_layers,
        num_heads,
        transformer_units,
        mlp_head_units,
    )
    ckpt = f"./ckpt/{scenario_i} ViT.h5"
    opt = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    checkpoint = ModelCheckpoint(
        ckpt, monitor="val_accuracy", verbose=0, save_best_only=True, mode="max"
    )
    lr_schedule = LearningRateSchedule(
        warmup_batches=warmup_batches, init_lr=learning_rate, decay=decay, verbose=0
    )
    h = model.fit(
        train,
        steps_per_epoch=train.__len__(),
        epochs=epochs,
        callbacks=[checkpoint, lr_schedule],
        validation_data=val,
        verbose=1,
    )

    model = load_keras_ckpt(
        ckpt,
        custom_objects={
            "Patches": Patches,
            "PatchEncoder": PatchEncoder,
            "PositionalEmbedding": PositionalEmbedding,
        },
    )

    for j, scenario_j in enumerate(scenarios):
        print(f"Test: {scenario_j}")
        test = DataGenerator(
            TEST[scenario_j],
            img_size,
            batch_size,
            "ViT",
            stats[scenario_j]["mean"],
            stats[scenario_j]["var"],
        )
        _, acc = model.evaluate(test, steps=test.__len__(), verbose=0)
        accuracy[i, j] = acc

        with open("results/accuracy ViT.npy", "wb") as f:
            np.save(f, accuracy)

    print()

accuracy = np.zeros((len(scenarios), len(scenarios)))
print("ViTa")
for i, scenario_i in enumerate(scenarios):
    print(f"Train: {scenario_i}")
    train = DataGenerator(
        TRAIN[scenario_i],
        img_size,
        batch_size,
        "ViT",
        stats[scenario_i]["mean"],
        stats[scenario_i]["var"],
    )
    val = DataGenerator(
        VAL[scenario_i],
        img_size,
        batch_size,
        "ViT",
        stats[scenario_i]["mean"],
        stats[scenario_i]["var"],
    )

    model = vita(
        input_shape,
        patch_size,
        projection_dim,
        num_patches,
        transformer_layers,
        num_heads,
        transformer_units,
        mlp_head_units,
    )
    ckpt = f"./ckpt/{scenario_i} ViTa.h5"
    opt = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    checkpoint = ModelCheckpoint(
        ckpt, monitor="val_accuracy", verbose=0, save_best_only=True, mode="max"
    )
    lr_schedule = LearningRateSchedule(
        warmup_batches=warmup_batches, init_lr=learning_rate, decay=decay, verbose=0
    )
    h = model.fit(
        train,
        steps_per_epoch=train.__len__(),
        epochs=epochs,
        callbacks=[checkpoint, lr_schedule],
        validation_data=val,
        verbose=1,
    )
    model = load_keras_ckpt(
        ckpt,
        custom_objects={
            "Patches": Patches,
            "PatchEncoder": PatchEncoder,
            "PositionalEmbedding": PositionalEmbedding,
        },
    )

    for j, scenario_j in enumerate(scenarios):
        print(f"Test: {scenario_j}")
        test = DataGenerator(
            TEST[scenario_j],
            img_size,
            batch_size,
            "ViT",
            stats[scenario_j]["mean"],
            stats[scenario_j]["var"],
        )
        _, acc = model.evaluate(test, steps=test.__len__(), verbose=0)
        accuracy[i, j] = acc

        with open("results/accuracy ViTa.npy", "wb") as f:
            np.save(f, accuracy)

    print()

accuracy = np.zeros((len(scenarios), len(scenarios)))
print("ResNet50")
for i, scenario_i in enumerate(scenarios):
    print(f"Train: {scenario_i}")
    train = DataGenerator(
        TRAIN[scenario_i],
        img_size,
        batch_size,
        "resnet",
        stats[scenario_i]["mean"],
        stats[scenario_i]["var"],
    )
    val = DataGenerator(
        VAL[scenario_i],
        img_size,
        batch_size,
        "resnet",
        stats[scenario_i]["mean"],
        stats[scenario_i]["var"],
    )

    model = load_resnet()
    ckpt = f"./ckpt/{scenario_i} ResNet50.h5"
    opt = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    checkpoint = ModelCheckpoint(
        ckpt, monitor="val_accuracy", verbose=0, save_best_only=True, mode="max"
    )
    lr_schedule = LearningRateSchedule(
        warmup_batches=warmup_batches, init_lr=learning_rate, decay=decay, verbose=0
    )
    h = model.fit(
        train,
        steps_per_epoch=train.__len__(),
        epochs=epochs,
        callbacks=[checkpoint, lr_schedule],
        validation_data=val,
        verbose=1,
    )
    model = load_keras_ckpt(ckpt)

    for j, scenario_j in enumerate(scenarios):
        print(f"Test: {scenario_j}")
        test = DataGenerator(
            TEST[scenario_j],
            img_size,
            batch_size,
            "resnet",
            stats[scenario_j]["mean"],
            stats[scenario_j]["var"],
        )
        _, acc = model.evaluate(test, steps=test.__len__(), verbose=0)
        accuracy[i, j] = acc

        with open("results/accuracy ResNet50.npy", "wb") as f:
            np.save(f, accuracy)

    print()

accuracy = np.zeros((len(scenarios), len(scenarios)))
print("EfficientNetB5")
for i, scenario_i in enumerate(scenarios):
    print(f"Train: {scenario_i}")
    train = DataGenerator(
        TRAIN[scenario_i],
        img_size,
        batch_size,
        "efficientnet",
        stats[scenario_i]["mean"],
        stats[scenario_i]["var"],
    )
    val = DataGenerator(
        VAL[scenario_i],
        img_size,
        batch_size,
        "efficientnet",
        stats[scenario_i]["mean"],
        stats[scenario_i]["var"],
    )

    model = load_efficientnet()
    ckpt = f"./ckpt/{scenario_i} EfficientNetB5.h5"
    opt = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    checkpoint = ModelCheckpoint(
        ckpt, monitor="val_accuracy", verbose=0, save_best_only=True, mode="max"
    )
    lr_schedule = LearningRateSchedule(
        warmup_batches=warmup_batches, init_lr=learning_rate, decay=decay, verbose=0
    )
    h = model.fit(
        train,
        steps_per_epoch=train.__len__(),
        epochs=epochs,
        callbacks=[checkpoint, lr_schedule],
        validation_data=val,
        verbose=1,
    )
    model = load_keras_ckpt(ckpt)

    for j, scenario_j in enumerate(scenarios):
        print(f"Test: {scenario_j}")
        test = DataGenerator(
            TEST[scenario_j],
            img_size,
            batch_size,
            "efficientnet",
            stats[scenario_j]["mean"],
            stats[scenario_j]["var"],
        )
        _, acc = model.evaluate(test, steps=test.__len__(), verbose=0)
        accuracy[i, j] = acc

        with open("results/accuracy EfficientNetB5.npy", "wb") as f:
            np.save(f, accuracy)

    print()

accuracy = np.zeros((len(scenarios), len(scenarios)))
print("VGG16")
for i, scenario_i in enumerate(scenarios):
    print(f"Train: {scenario_i}")
    train = DataGenerator(
        TRAIN[scenario_i],
        img_size,
        batch_size,
        "vgg",
        stats[scenario_i]["mean"],
        stats[scenario_i]["var"],
    )
    val = DataGenerator(
        VAL[scenario_i],
        img_size,
        batch_size,
        "vgg",
        stats[scenario_i]["mean"],
        stats[scenario_i]["var"],
    )

    model = load_vgg()
    ckpt = f"./ckpt/{scenario_i} VGG16.h5"
    opt = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    checkpoint = ModelCheckpoint(
        ckpt, monitor="val_accuracy", verbose=0, save_best_only=True, mode="max"
    )
    lr_schedule = LearningRateSchedule(
        warmup_batches=warmup_batches, init_lr=learning_rate, decay=decay, verbose=0
    )
    h = model.fit(
        train,
        steps_per_epoch=train.__len__(),
        epochs=epochs,
        callbacks=[checkpoint, lr_schedule],
        validation_data=val,
        verbose=1,
    )
    model = load_keras_ckpt(ckpt)

    for j, scenario_j in enumerate(scenarios):
        print(f"Test: {scenario_j}")
        test = DataGenerator(
            TEST[scenario_j],
            img_size,
            batch_size,
            "vgg",
            stats[scenario_j]["mean"],
            stats[scenario_j]["var"],
        )
        _, acc = model.evaluate(test, steps=test.__len__(), verbose=0)
        accuracy[i, j] = acc

        with open("results/accuracy VGG16.npy", "wb") as f:
            np.save(f, accuracy)

    print()
