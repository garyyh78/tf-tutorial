import tensorflow as tf
import numpy as np
import os


from tensorflow.examples.tutorials.mnist import input_data

# random data init
np.random.seed(0)
tf.set_random_seed(0)

# download MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
n_samples = mnist.train.num_examples
print("total train samples = ", n_samples, "\n")

# network meta data
MNIST_DIM = 28
net_metadata = {'input_size': MNIST_DIM * MNIST_DIM,
                'D_layer1_size': 500,
                'G_layer1_size': 500,
                'out_layer_size': 1,
                'z_size': 200,
                'activate func': tf.nn.relu,
                'output func': tf.nn.sigmoid,
                'd_learning_rate': 0.0001,
                'g_learning_rate': 0.0001,
                'training_epoch': 100,
                'display_step': 10,
                'model_path': "./model.gan/model.ckpt",
                'batch_size': 100,
                'cost function version': "raw2"
                }

# global data place holder for TF nodes
weights = dict()

# Graph definition: input
x = tf.placeholder(tf.float32, [None, net_metadata['input_size']], "x_input")
z = tf.placeholder(tf.float32, [None, net_metadata['z_size']], "z_input")

# Graph definition: D net
weights['D_h1'] = tf.get_variable("D_h1",
                                  [net_metadata['input_size'],
                                   net_metadata['D_layer1_size']],
                                  tf.float32,
                                  tf.contrib.layers.xavier_initializer())
weights['D_h2'] = tf.get_variable("D_h2",
                                  [net_metadata['D_layer1_size'],
                                   net_metadata['out_layer_size']],
                                  tf.float32,
                                  tf.contrib.layers.xavier_initializer())
weights['D_b1'] = tf.get_variable("D_b1",
                                  [net_metadata['D_layer1_size']],
                                  tf.float32,
                                  tf.zeros_initializer())
weights['D_b2'] = tf.get_variable("D_b2",
                                  [net_metadata['out_layer_size']],
                                  tf.float32,
                                  tf.zeros_initializer())

# D_net has to be a func so it can connect two input;
# Also don't do sigmoid and return raw value


def D_Net(x0, weights):

    D_L1 = tf.nn.relu(tf.matmul(x0, weights['D_h1']) + weights['D_b1'])
    D_rawout = tf.matmul(D_L1, weights['D_h2']) + weights['D_b2']
    return D_rawout

# Graph definition: G net
weights['G_h1'] = tf.get_variable("G_h1",
                                  [net_metadata['z_size'],
                                   net_metadata['G_layer1_size']],
                                  tf.float32,
                                  tf.contrib.layers.xavier_initializer())
weights['G_h2'] = tf.get_variable("G_h2",
                                  [net_metadata['G_layer1_size'],
                                   net_metadata['input_size']],
                                  tf.float32,
                                  tf.contrib.layers.xavier_initializer())
weights['G_b1'] = tf.get_variable("G_b1",
                                  [net_metadata['G_layer1_size']],
                                  tf.float32,
                                  tf.zeros_initializer())
weights['G_b2'] = tf.get_variable("G_b2",
                                  [net_metadata['input_size']],
                                  tf.float32,
                                  tf.zeros_initializer())

# feed Z into G net
G_L1 = net_metadata['activate func'](tf.add(tf.matmul(z, weights['G_h1']),
                                     weights['G_b1']))
G_out = net_metadata['output func'](tf.add(tf.matmul(G_L1, weights['G_h2']),
                                    weights['G_b2']))

D_raw_real = D_Net(x, weights)
D_raw_fromZ = D_Net(G_out, weights)

D_real = net_metadata['output func'](D_raw_real)
D_fromZ = net_metadata['output func'](D_raw_fromZ)

# define cost function


def DNetCost(x, f):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=f(x)))


cost_version = net_metadata['cost function version']
if cost_version == "raw2":
    # D pushes real sample to 1s
    cost_real = DNetCost(D_raw_real, tf.ones_like)
    # D pushes z samples to 0s
    cost_fromZ = DNetCost(D_raw_fromZ, tf.zeros_like)
    dcost = cost_real + cost_fromZ
    # G pushes z samples to 1s
    gcost = DNetCost(D_raw_fromZ, tf.ones_like)
else:
    # default raw version
    dcost = -tf.reduce_mean(tf.log(D_real) + tf.log(1.0 - D_fromZ))
    gcost = -tf.reduce_mean(tf.log(D_fromZ))

# define optimizer and make sure to set up correct params update list
dWeights = [weights['D_h1'], weights['D_h2'], weights['D_b1'], weights['D_b2']]
gWeights = [weights['G_h1'], weights['G_h2'], weights['G_b1'], weights['G_b2']]
d_rate = net_metadata['d_learning_rate']
g_rate = net_metadata['g_learning_rate']
d_optimizer = tf.train.AdamOptimizer(d_rate).minimize(dcost, var_list=dWeights)
g_optimizer = tf.train.AdamOptimizer(g_rate).minimize(gcost, var_list=gWeights)

# Init & Train
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
saver = tf.train.Saver()

dirname = os.path.dirname(net_metadata['model_path'])
if os.path.isdir(dirname):
    print("continue with saved model .. \n")
    saver.restore(sess, net_metadata['model_path'])
else:
    os.mkdir(dirname)
    print("start init ...\n")
    sess.run(init)

for epoch in range(net_metadata['training_epoch']):
    avg_cost = 0.0
    bsz = net_metadata['batch_size']
    total_batch = int(n_samples / bsz)
    for i in range(total_batch):

        batch_x, _ = mnist.train.next_batch(bsz)
        batch_z1 = np.random.uniform(-1., 1.,
                                     size=[bsz, net_metadata['z_size']])
        batch_z2 = np.random.uniform(-1., 1.,
                                     size=[bsz, net_metadata['z_size']])
        _, dcost0 = sess.run([d_optimizer, dcost],
                             feed_dict={x: batch_x, z: batch_z1})
        _, gcost0 = sess.run([g_optimizer, gcost],
                             feed_dict={z: batch_z2})

    # Display logs per epoch step
    if epoch % net_metadata['display_step'] == 0:
        saver.save(sess, net_metadata['model_path'])
        print("Epoch:",
              '%04d' % (epoch+1),
              "dcost=", "{:.9f}".format(dcost0),
              "gcost=", "{:.9f}".format(gcost0))
sess.close()
