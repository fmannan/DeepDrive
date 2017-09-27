import numpy as np
from keras.models import Model, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Input, Conv2D, MaxPool2D, ELU, Dropout, concatenate, \
    BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint, TensorBoard

import csv
import os
from scipy.misc import imread

"""
First load the csv file and split it into train and test

generator will return batched samples fromo the train and validation set

"""


def load_data_all(dataset, data_src):
    """
    Each training input is 3 x height x width x channels  and output is a vector 1 x 4
    """
    #print('load_data_all data_src', data_src)
    X_left = []
    X_center = []
    X_right = []

    y = []
    lst_X = [X_center, X_left, X_right]
    for entry in dataset:
        for idx, val in enumerate(entry[:3]):
            val = val.replace('\\', '/')
            filename = val.strip().split('/')[-2:]
            #print(filename)
            full_filename = os.path.join(data_src, filename[0], filename[1])
            #print(full_filename)
            img = imread(full_filename).astype(np.float32)
            lst_X[idx].append(img)
        y.append(np.array([float(v) for v in entry[3:]]))

    return {'label': np.stack(y, axis=0), 'left': np.stack(X_left, axis=0),
            'center': np.stack(X_center, axis=0), 'right': np.stack(X_right, axis=0)}


def load_data_center_label(dataset, data_src):
    data = load_data_all(dataset, data_src)
    return data['center'], data['label']


def load_data_center_and_steering(dataset, data_src):
    data = load_data_all(dataset, data_src)
    return data['center'], np.squeeze(data['label'][:, 0])


def load_data_center_and_steering_augmented(dataset, data_src, steering_offset):
    data = load_data_all(dataset, data_src)
    X_center = data['center']
    X_left = data['left']
    X_right = data['right']

    y_center = np.squeeze(data['label'][:, 0])

    # add left and right camera images and steering offset
    X = np.concatenate((X_center, X_left, X_right), axis=0)
    y = np.concatenate((y_center, y_center + steering_offset, y_center - steering_offset))

    # fliplr and steering angle
    X_mirror = X[:, :, ::-1, :]

    X = np.concatenate((X, X_mirror), axis=0)
    y = np.concatenate((y, -y))

    return X, y


def load_driving_data_from_csv(filename):
    """
    Loads all the data from a csv file and creates a data dictionary. To save memory this doesn't load the images but
    stores the full path to the images.
    :param filename: full path to the csv file
    :param data_src: full path to the directory containing the IMG directory
    :return: A list with each entry containing the left, right, and center image fullpath and steering angles.
    """
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        dataset_raw = list(reader)

    col_headers = dataset_raw[0]

    for idx, val in enumerate(['center', 'left', 'right']):
        assert col_headers[idx].lower() == val

    return dataset_raw[1:]


class DataGenerator(object):
    """
    A generic class that first loads some data dictionary using an implementation specified by the caller and then
    splits it into training and validation and provides generators for them.
    """

    def __init__(self, dataset, data_src, train_val_split, fn_data_loader):
        self.data_src = data_src
        self.dataset = dataset
        print('DataGenerator::__init__.data_src', self.data_src)
        dataset_size = len(self.dataset)
        trainset_size = int(np.round(dataset_size * train_val_split))

        self.trainset = self.dataset[:trainset_size]
        self.valset = self.dataset[trainset_size:]

        self.trainset_size = len(self.trainset)
        self.valset_size = len(self.valset)

        self.fn_data_loader = fn_data_loader

    @property
    def num_training_samples(self):
        return self.trainset_size

    @property
    def num_validation_samples(self):
        return self.valset_size

    def __batch_generator(self, dataset, dataset_size, batch_size, shuffle):
        while True:
            if shuffle:
                np.random.shuffle(dataset)

            n_batches = int(np.ceil(dataset_size / batch_size))
            #print(n_batches)
            for idx in range(n_batches):
                start_idx = idx * batch_size
                #print(start_idx, dataset_size)
                end_idx = min((idx + 1) * batch_size, dataset_size)
                # load the actual data
                X, y = self.fn_data_loader(dataset[start_idx:end_idx], self.data_src)
                subset_size = y.shape[0]
                if subset_size > batch_size:
                    indices = np.arange(y.shape[0])
                    # do finer partitioning
                    if shuffle:
                        np.random.shuffle(indices)
                    n_sub_batches = int(np.ceil(subset_size / batch_size))
                    for sub_idx in range(n_sub_batches):
                        start_sub_idx = sub_idx * batch_size
                        end_sub_idx = min((sub_idx + 1) * batch_size, subset_size)
                        yield X[indices[start_sub_idx:end_sub_idx]], y[indices[start_sub_idx:end_sub_idx]]
                else:
                    yield X, y

    def train_generator(self, batch_size, shuffle=True):
        return self.__batch_generator(dataset=self.trainset, dataset_size=self.trainset_size,
                                     batch_size=batch_size, shuffle=shuffle)

    def validation_generator(self, batch_size, shuffle=False):
        return self.__batch_generator(dataset=self.valset, dataset_size=self.valset_size,
                                         batch_size=batch_size, shuffle=shuffle)


def preprocess(x, crop_rows):
    crop = Cropping2D(cropping=(crop_rows, (0, 0)))(x)
    norm = Lambda(lambda x: (x / 255.0) - 0.5)(crop)
    return norm

def comma_ai_preprocess(x, crop_rows):
    crop = Cropping2D(cropping=(crop_rows, (0, 0)))(x)
    norm = Lambda(lambda x: (x / 127.5) - 1.0)(crop)
    return norm

def basic_network(x):
    fc = Flatten()(x)
    return Dense(1)(fc)

def LeNet(x):
    layer1 = Conv2D(6, (5, 5), padding='same', activation='relu')(x)
    layer1_maxpool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(layer1)
    layer2 = Conv2D(16, (5, 5), padding='same', activation='relu')(layer1_maxpool)
    layer2_maxpool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(layer2)
    fc1 = Dense(120)(Flatten()(layer2_maxpool))
    fc2 = Dense(84)(fc1)
    logits = Dense(1)(fc2)

    return logits


def DriveNet_v0(x):
    layer1_1 = Conv2D(8, (1, 1), padding='same', activation='relu')(x)
    layer1_2 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)

    layer1 = concatenate([layer1_1, layer1_2], axis=-1)

    layer2 = Conv2D(16, (5, 5), strides=(2, 2), padding='same', activation='relu')(layer1)
    layer3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(layer2)
    layer4 = ELU()(Dropout(0.5)(Dense(256)(Flatten()(layer3))))
    layer5 = ELU()(Dropout(0.25)(Dense(128)(layer4)))

    logits = Dense(1)(layer5)
    return logits


def DriveNet_v1(x):
    layer1_1 = Conv2D(8, (1, 1), padding='same', activation='relu')(x)
    layer1_2 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)

    layer1 = concatenate([layer1_1, layer1_2], axis=-1)

    layer2 = Conv2D(16, (5, 5), strides=(2, 2), padding='same', activation='relu')(layer1)
    layer3 = Conv2D(32, (5, 5), strides=(2, 2), padding='same', activation='relu')(layer2)
    layer4 = Dropout(0.5)(Dense(256)(Flatten()(layer3)))
    layer5 = Dropout(0.5)(Dense(64)(layer4))

    logits = Dense(1)(layer5)
    return logits


def DriveNet_v2(x):
    layer1_1 = Conv2D(8, (1, 1), padding='same', activation='relu')(x)
    layer1_2 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)

    layer1 = concatenate([layer1_1, layer1_2], axis=-1)

    layer2 = Conv2D(16, (5, 5), strides=(2, 2), padding='same', activation='relu')(layer1)
    layer3 = Conv2D(32, (5, 5), strides=(2, 2), padding='same', activation='relu')(layer2)
    layer4 = ELU()(Dropout(0.5)(Dense(256)(Flatten()(layer3))))
    layer5 = ELU()(Dropout(0.5)(Dense(64)(layer4)))

    logits = Dense(1)(layer5)
    return logits


def BN_RELU(x):
    x = BatchNormalization()(x)
    return Activation('relu')(x)


def DriveNet_v3(x):
    layer1_0 = Conv2D(8, (3, 3), padding='same')(x)
    layer1 = concatenate([BN_RELU(x), BN_RELU(layer1_0)], axis=-1)

    layer2 = BN_RELU(Conv2D(16, (5, 5), strides=(2, 2), padding='same')(layer1))
    layer3 = BN_RELU(Conv2D(32, (5, 5), strides=(2, 2), padding='same')(layer2))
    layer4 = Dropout(0.5)(BN_RELU((Dense(256)(Flatten()(layer3)))))
    layer5 = ELU()(Dropout(0.5)(Dense(64)(layer4)))
    layer6 = Dropout(0.2)(Dense(10)(layer5))

    logits = Dense(1)(layer6)
    return logits

def DriveNet_v3_small(x):
    layer1_0 = Conv2D(8, (3, 3), padding='same')(x)
    layer1 = concatenate([BN_RELU(x), BN_RELU(layer1_0)], axis=-1)

    layer2 = BN_RELU(Conv2D(8, (5, 5), strides=(2, 2), padding='same')(layer1))
    layer3 = BN_RELU(Conv2D(16, (5, 5), strides=(2, 2), padding='same')(layer2))
    layer4 = Dropout(0.5)(BN_RELU((Dense(128)(Flatten()(layer3)))))
    layer5 = ELU()(Dropout(0.5)(Dense(32)(layer4)))
    layer6 = Dropout(0.2)(Dense(10)(layer5))

    logits = Dense(1)(layer6)
    return logits


def DriveNet_v4(x):
    layer1_0 = Conv2D(16, (7, 7), padding='same')(x)
    layer1 = concatenate([BN_RELU(x), BN_RELU(layer1_0)], axis=-1)

    layer2 = BN_RELU(Conv2D(8, (5, 5), strides=(2, 2), padding='same')(layer1))
    layer3 = BN_RELU(Conv2D(16, (5, 5), strides=(2, 2), padding='same')(layer2))

    layer4 = BN_RELU(Conv2D(32, (3, 3), padding='same')(layer3))
    layer5 = BN_RELU(Conv2D(48, (3, 3), padding='same')(layer4))

    layer6 = Dropout(0.5)(BN_RELU(Dense(100)(Flatten()(layer5))))
    layer7 = Dense(10)(layer6)

    logits = Dense(1)(layer7)
    return logits


def DriveNet_v5(x):
    layer1 = BN_RELU(Conv2D(16, (7, 7), strides=(3, 3), padding='same')(x))
    layer2 = BN_RELU(Conv2D(8, (5, 5), strides=(2, 2), padding='same')(layer1))
    layer3 = BN_RELU(Conv2D(16, (5, 5), strides=(2, 2), padding='same')(layer2))

    layer4 = BN_RELU(Conv2D(32, (3, 3), padding='same')(layer3))
    layer5 = BN_RELU(Conv2D(48, (3, 3), padding='same')(layer4))

    layer6 = Dropout(0.5)(BN_RELU(Dense(100)(Flatten()(layer5))))
    layer7 = Dense(10)(layer6)

    logits = Dense(1)(layer7)
    return logits


def Nvidia(x):
    x = Conv2D(3, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(24, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(36, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)

    x = Conv2D(48, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)

    x = Dense(100)(Flatten()(x))
    x = Dense(50)(x)
    x = Dense(10)(x)
    logits = Dense(1)(x)

    return logits

def Nvidia_small(x):
    x = Conv2D(3, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(24, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(36, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)

    x = Conv2D(48, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)

    x = Dense(100)(Flatten()(x))
    x = Dense(10)(x)
    logits = Dense(1)(x)

    return logits

def Comma(x):
    layer1 = Conv2D(16, (8, 8), strides=(4, 4), padding='same', activation='elu')(x)
    layer2 = Conv2D(32, (5, 5), strides=(2, 2), padding='same', activation='elu')(layer1)
    layer3 = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(layer2)

    fc1 = ELU()(Dropout(.2)(Flatten()(layer3)))
    fc2 = ELU()(Dropout(0.5)(Dense(512)(fc1)))

    logits = Dense(1)(fc2)

    return logits


def main():
    import sys
    import argparse
    import argcomplete

    Network_Table = {'basic': basic_network,
                     'lenet': LeNet,
                     'comma': Comma,
                     'nvidia': Nvidia,
                     'nvidia-small': Nvidia_small,
                     'drive-v0': DriveNet_v0,
                     'drive-v1': DriveNet_v1,
                     'drive-v2': DriveNet_v2,
                     'drive-v3': DriveNet_v3,
                     'drive-v3-small': DriveNet_v3_small,
                     'drive-v4': DriveNet_v4,
                     'drive-v5': DriveNet_v5,
                     }

    # model.py --data_dir=... --output=model.h5  --network=...
    parser = argparse.ArgumentParser(description='Train a driving model.\n Usage: ' + sys.argv[0] +
                                                 '--data_dir ./data/ --output out.h5 --network alexnet --epochs 10')
    parser.add_argument('--data_dir', type=str, help='Directory where the CSV file(s) and the IMG folder(s) are.' +
                        'If the directory contains multiple CSV files then they will be merged into one.', required=True)
    parser.add_argument('--output', type=str, help='Output filepath', required=True)
    parser.add_argument('--epochs', type=int, help='Number of epochs', required=True)
    parser.add_argument('--steering_offset_left', type=float, help='Steering offset for left image', default=0.1)
    parser.add_argument('--pretrained', type=str, help='Path to pretrained model')
    parser.add_argument('--train_frac', type=float, help='Fraction of training images', default=0.75)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=500)
    parser.add_argument('--network', type=str, help='Network to use for driving model.', required=True)

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    print(args)

    src_dir = args.data_dir
    filename = 'driving_log.csv'
    csv_fullpath = os.path.join(src_dir, filename)

    print(csv_fullpath)
    dataset = load_driving_data_from_csv(csv_fullpath)

    fn_data_loader = lambda x, y: load_data_center_and_steering_augmented(x, y, steering_offset=args.steering_offset_left)
    data_generator = DataGenerator(dataset=dataset, data_src=src_dir, train_val_split=args.train_frac,
                                   fn_data_loader=fn_data_loader)
    batch_size = args.batch_size
    train_generator = data_generator.train_generator(batch_size=batch_size, shuffle=True)
    validation_generator = data_generator.validation_generator(batch_size=batch_size, shuffle=True)

    crop_rows_amount = [60, 20]  #[70, 25]

    im_shape = (160, 320, 3)
    inp = Input(shape=im_shape)
    if args.pretrained is not None:
        print('loading pretrained model from {}'.format(args.pretrained))
        model = load_model(args.pretrained)
    else:
        if args.network == 'commanet':
            preproc_im = comma_ai_preprocess(inp, crop_rows=crop_rows_amount)
        else:
            preproc_im = preprocess(inp, crop_rows=crop_rows_amount)

        net_out = Network_Table[args.network](preproc_im)

        model = Model(inputs=inp, outputs=net_out)
        model.compile(loss='mse', optimizer='adam')

    epochs = args.epochs

    checkpoints_dir = './ckpt/'
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    filepath = checkpoints_dir + "/weights-" + args.network + "-" + args.output.split('.')[0] + \
                                                                              "-{epoch:02d}-{val_loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max')

    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
                              write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                              embeddings_metadata=None)

    callbacks_list = [tensorboard, checkpoint]

    hist = model.fit_generator(train_generator,
                               steps_per_epoch=9 * data_generator.num_training_samples // batch_size,
                               epochs=epochs,
                               validation_data=validation_generator,
                               validation_steps=9 * data_generator.num_validation_samples // batch_size,
                               callbacks=callbacks_list)

    model.save(args.output)


if __name__ == '__main__':
    main()

