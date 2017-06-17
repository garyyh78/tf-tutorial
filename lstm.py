import os
import collections

import tensorflow as tf

ptb_path = "./PTB/"

# prepare PTB data
train_path = os.path.join(ptb_path, "ptb.train.txt")
valid_path = os.path.join(ptb_path, "ptb.valid.txt")
test_path = os.path.join(ptb_path, "ptb.test.txt")


def get_words_from_file(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().decode("utf-8").replace("\n", "<eos>").split()


def file_to_word_ids(filename, word_to_id):
    dat = get_words_from_file(filename)
    return [word_to_id[word] for word in dat if word in word_to_id]


def build_vocab(filename):
    dat = get_words_from_file(filename)
    counter = collections.Counter(dat)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    # show top ranked works for sanity check
    print(len(dat), "\n")
    print(dat[0], "\n")
    print(len(count_pairs), "\n")
    print(count_pairs[0], "\n")
    print(count_pairs[1], "\n")

    zipped_pairs = zip(*count_pairs)
    words, _ = list(zipped_pairs)  # need to take first element
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id

# build vacab from train data, each ID is 10000 long tuple
word_id = build_vocab(train_path)
print(word_id['apple'], "\n")

# map text to a sequence of integers
train_data = file_to_word_ids(train_path, word_id)
valid_data = file_to_word_ids(valid_path, word_id)
test_data = file_to_word_ids(test_path, word_id)

# data slicing test


def ptb_produce(raw, bs, n_steps):
    with tf.name_scope(None, "ptb_produce", [raw, bs, n_steps]):
        raw = tf.convert_to_tensor(raw, name="raw", dtype=tf.int32)
        raw_len = tf.size(raw)
        batch_len = raw_len // bs
        data = tf.reshape(raw[0: bs * batch_len], [bs, batch_len])
        epoch_size = tf.identity((batch_len - 1) // n_steps, name="epoch_size")

        idx = tf.train.range_input_producer(epoch_size,
                                            shuffle=False).dequeue()

        x = tf.strided_slice(data, [0, idx * num_steps],
                             [batch_size, (idx + 1) * num_steps])
        x.set_shape([batch_size, num_steps])

        y = tf.strided_slice(data, [0, idx * num_steps + 1],
                             [batch_size, (idx + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])

    return x, y

# small test case
batch_size = 3
num_steps = 2
raw_data = [11, 12, 13, 14, 15,
            21, 22, 23, 24, 25,
            31, 32, 33, 34, 35]
x, y = ptb_produce(raw_data, batch_size, num_steps)

with tf.Session() as session:

    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(session, coord=coord)

    try:

        xval, yval = session.run([x, y])
        print(xval, "\n")
        print(yval, "\n")

        xval, yval = session.run([x, y])
        print(xval, "\n")
        print(yval, "\n")

    finally:

        coord.request_stop()
        coord.join()
