
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import time
import csv
from random import shuffle
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics
from math import sqrt
import os
import pandas as pd


# In[2]:


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=  0.2)


# In[3]:


# flags
tf.flags.DEFINE_float("epsilon", 0.1, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("learning_rate", 0.00001, "Learning rate")
tf.flags.DEFINE_float("max_grad_norm", 20.0, "Clip gradients to this norm.")
tf.flags.DEFINE_float("keep_prob", 0.3, "Keep probability for dropout")
tf.flags.DEFINE_integer("hidden_layer_num", 1, "The number of hidden layers (Integer)")
tf.flags.DEFINE_integer("hidden_size", 200, "The number of hidden nodes (Integer)")
tf.flags.DEFINE_integer("preprocess_size", 200, "The number of preprocess nodes after one-hot (Integer)")
tf.flags.DEFINE_integer("embedding_size", 300, "The number of nodes for word embedding (Integer)")
tf.flags.DEFINE_integer("evaluation_interval", 5, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size",330, "Batch size for training.")
tf.flags.DEFINE_integer("epochs", 5, "Number of epochs to train for.")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("train_data_path", 'train_v2.npy', "Path to the training dataset")
# tf.flags.DEFINE_string("test_data_path", 'data/2012_assist_test.csv', "Path to the testing dataset")

# In[4]:


# output_path = 'output.npy'
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
# log_file_path = FLAGS.train_data_path[5:-4] + 'l2'  + '.txt'
log_file_path = 'log.txt'
# hidden_state_path =FLAGS.train_data_path[5:-4] + str(FLAGS.hidden_layer_num) + '.npy'


# In[5]:


print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# In[6]:


class HyperParamsConfig(object):
    """Small config."""
    init_scale = 0.05
    num_steps = 0
    max_grad_norm = FLAGS.max_grad_norm
    max_max_epoch = FLAGS.epochs
    keep_prob = FLAGS.keep_prob
    num_skills = 0
    state_size = [200]
    beta = 1


# In[7]:


class KKBoxModel(object):

    def __init__(self, is_training, config):
        self.state_size = config.state_size
        self._batch_size = batch_size = FLAGS.batch_size
        self.input_dimensions = input_dimensions = config.input_dimensions
        self.hidden_layer_num = len(self.state_size)
        self.hidden_size = size = FLAGS.hidden_size
        self.num_steps = num_steps = config.num_steps
        self.is_training = is_training
        input_size = input_dimensions
#         preprocess_size = FLAGS.preprocess_size
#         embedding_size = FLAGS.embedding_size
        inputs = self._input_data = tf.placeholder(tf.float32, [batch_size, num_steps, input_dimensions])
        # print "inputs"
        # print inputs.shape
#         self._target_id = target_id = tf.placeholder(tf.int32, [None])
#         input_vectors = self._input_vector = tf.placeholder(tf.float32, [batch_size,num_steps , embedding_size ])
        self._target_correctness = target_correctness = tf.placeholder(tf.float32, [None])
        final_hidden_size = self.state_size[-1]

        hidden_layers = []
        # input_vectors = tf.reshape(input_vectors, [batch_size,num_steps ,preprocess_size ])
        for i in range(self.hidden_layer_num):
            
            hidden1 = tf.contrib.rnn.BasicLSTMCell(self.state_size[i], state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
            if is_training and config.keep_prob < 1:
                hidden1 = tf.contrib.rnn.DropoutWrapper(hidden1, output_keep_prob=FLAGS.keep_prob)
            hidden_layers.append(hidden1)
        
        cell = tf.contrib.rnn.MultiRNNCell(hidden_layers, state_is_tuple=True)

        #input_data: [batch_size*num_steps]
        input_data = tf.reshape(self._input_data, [-1])
        
        inputs = tf.reshape(input_data,[-1,num_steps,input_size])
        # print "inputs"
        # print inputs.shape
#         print input_data.shape
        #one-hot encoding
#         with tf.device("/gpu:0"):
#             #labels: [batch_size* num_steps, 1]
#             labels = tf.expand_dims(input_data, 1)
#             #indices: [batch_size*num_steps, 1]
#             indices = tf.expand_dims(tf.range(0, batch_size*num_steps, 1), 1)
#             #concated: [batch_size * num_steps, 2]
#             concated = tf.concat( [indices, labels],1)

#             # If sparse_indices is an n by d matrix, then for each i in [0, n)
#             # dense[sparse_indices[i][0], ..., sparse_indices[i][d-1]] = sparse_values[i]
#             # input_size: 2* num_skills
#             # inputs: [batch_size* num_steps * input_size]
#             inputs = tf.sparse_to_dense(concated, tf.stack([batch_size*num_steps, input_size]), 1.0, 0.0)
#             inputs.set_shape([batch_size*num_steps, input_size])

        # [batch_size, num_steps, input_size]
        
        inputs = tf.reshape(inputs, [-1, num_steps, input_size])
        x = inputs
        # x = tf.transpose(inputs, [1, 0, 2])
        # # Reshape to (n_steps*batch_size, n_input)
        # x = tf.reshape(x, [-1, input_size])
        # # Split to get a list of 'n_steps'
        # # tensors of shape (doc_num, n_input)
        # x = tf.split(0, num_steps, x)
        #inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, num_steps, inputs)]
        #outputs, state = tf.nn.rnn(hidden1, x, dtype=tf.float32)
        
        
        # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
        outputs, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

        final_outputs = outputs[:, -1, :]
        output = tf.reshape(tf.concat(final_outputs,1), [-1, final_hidden_size])
        # calculate the logits from last hidden layer to output layer


        # print "outputs"
        # print state.shape
        # print "output"
        # print output.shape

        
        sigmoid_w = tf.get_variable("sigmoid_w", [final_hidden_size, 1])
        sigmoid_b = tf.get_variable("sigmoid_b", [1])
        logits = tf.matmul(output, sigmoid_w) + sigmoid_b
        
        # print "logits"
        # print  logits.shape
         # from output nodes to pick up the right one we want
        logits = tf.reshape(logits, [-1])
#         self._last_logits = logits[(batch_size)*(num_steps-1):,:]
        selected_logits = logits
 
         #make prediction
        self._pred = self._pred_values = pred_values = tf.sigmoid(selected_logits)
 
         # loss function
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = selected_logits,labels= target_correctness))
 
        # loss = tf.reduce_sum(loss + config.beta * tf.norm(lstm_weights))
        # loss = tf.reduce_sum(loss + config.beta * tf.nn.l2_loss(sigmoid_w))
        # loss += 

        #self._cost = cost = tf.reduce_mean(loss)
        self._final_state = state
        self._cost = cost = loss

    @property
    def batch_size(self):
        return self._batch_size


    @property
    def input_data(self):
        return self._input_data

    @property
    def input_vector(self):
        return self._input_vector

    @property
    def auc(self):
        return self._auc

    @property
    def pred(self):
        return self._pred

    @property
    def target_id(self):
        return self._target_id

    @property
    def target_correctness(self):
        return self._target_correctness

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def pred_values(self):
        return self._pred_values

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state


# In[21]:


def run_epoch(session, m, users, labels, eval_op, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()

    index = 0
    pred_labels = []
    actual_labels = []
    # print labels.shape
    while(index+m.batch_size < len(users) +1):
        x = np.zeros((m.batch_size, m.num_steps, m.input_dimensions))
        # x=[]
        target_correctness = []
#         problem_context = np.zeros((m.batch_size, m.num_steps, FLAGS.embedding_size))
        count = 0
        for i in range(m.batch_size):
            user = users[index+i]
#             problem_ids = student[1]
#             skill_ids = student[2]
            num_step = user.shape[0]
            target_correctness.append(labels[index+i])
            # print target_correctness.shape
#             print user.shape
        
            # for j in range(num_step):
# #                 skill_id = int(skill_ids[j])
# #                 problem_id = int(problem_ids[j])
# #                 context = context_df.loc[context_df['problem_ids'] == problem_id, 'problem_vectors'].iloc[0]
#                 # print context
# #                 label_index = 0
#                 if(int(correctness[j]) == 0):
#                     label_index = skill_id
#                 else:
#                     label_index = skill_id + m.num_skills
            x[i,:num_step] = user
                
#                 problem_context[i,j] = context
#                 target_id.append(i*m.num_steps*m.num_skills+j*m.num_skills+int(skill_ids[j+1]))
#                 target_correctness.append(int(correctness[j+1]))
#                 actual_labels.append(int(correctness[j+1]))
                # problem_context.append(context)
        # x = np.array(x)
        # print x.shape
        # print len(target_correctness)
        actual_labels += target_correctness
        pred, _, final_state, cost = session.run([m.pred, eval_op, m.final_state, m.cost], feed_dict={
            m.input_data: x,
            m.target_correctness: target_correctness})
        
        index += m.batch_size
        


        for p in pred:
            pred_labels.append(p)

        

    

    rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
    fpr, tpr, thresholds = metrics.roc_curve(actual_labels, pred_labels, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    #calculate r^2
    accuracy = metrics.accuracy_score(actual_labels, np.array(pred_labels).round())
    if not m.is_training:
        np.save('predicted_test.npy',np.array(pred_labels).round())
    return rmse, auc, accuracy, final_state


# In[15]:


def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks [2].
    tf.op_scope(values, name, default_name) is deprecated, use tf.name_scope(name, default_name, values)
    """
    with tf.name_scope( name, "add_gradient_noise",[t, stddev]) as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)


# In[16]:


def main(unused_args):
    
    config = HyperParamsConfig()
    eval_config = HyperParamsConfig()
    timestamp = str(time.time())
    train_data_path = FLAGS.train_data_path
    #path to your test data set
#     test_data_path = FLAGS.test_data_path
    #the file to store your test results
    result_file_path = "run_logs_{}".format(timestamp)
    #your model name
    model_name = "Deep KKBOX"

    train_users = np.load('active_data_train.npy')
    train_label = pd.read_csv('active_user_label_train.csv').is_churn
    test_users = np.load('active_data_test.npy')
    test_label = pd.read_csv('active_user_label_test.csv').is_churn
    train_max_steps = 0
    print train_users.shape
    print test_users.shape
    for test_user in test_users:
        if len(test_user) > train_max_steps:
            # print len(train_user)
            train_max_steps = len(test_user)
    
    train_input_dimensions = train_users[0].shape[1]
    config.num_steps = train_max_steps
    
    config.input_dimensions = train_input_dimensions
#     test_students, test_max_num_problems, test_max_skill_num = read_data_from_csv_file(test_data_path)
#     eval_config.num_steps = test_max_num_problems
#     eval_config.num_skills = test_max_skill_num

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement,
                                      gpu_options=gpu_options)

        global_step = tf.Variable(0, name="global_step", trainable=False)
        # decay learning rate
        starter_learning_rate = FLAGS.learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 3000, 0.96, staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=FLAGS.epsilon)

        with tf.Session(config=session_conf) as session:

            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

            # training model
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = KKBoxModel(is_training=True, config=config)
            # testing model
            with tf.variable_scope("model", reuse=True, initializer=initializer):
                mtest = KKBoxModel(is_training=False, config=config)

            grads_and_vars = optimizer.compute_gradients(m.cost)
            grads_and_vars = [(tf.clip_by_norm(g, FLAGS.max_grad_norm), v)
                              for g, v in grads_and_vars if g is not None]
            grads_and_vars = [(add_gradient_noise(g), v) for g, v in grads_and_vars]
            train_op = optimizer.apply_gradients(grads_and_vars, name="train_op", global_step=global_step)
            session.run(tf.global_variables_initializer())
            # log hyperparameters to results file
            # with open(result_file_path, "a+") as f:
            #     print("Writing hyperparameters into file")
            #     f.write("Hidden layer size: %d \n" % (FLAGS.hidden_size))
            #     f.write("Dropout rate: %.3f \n" % (FLAGS.keep_prob))
            #     f.write("Batch size: %d \n" % (FLAGS.batch_size))
            #     f.write("Max grad norm: %d \n" % (FLAGS.max_grad_norm))
            # saver = tf.train.Saver(tf.all_variables())
            
            for i in range(config.max_max_epoch):
                rmse, auc, accuracy, final_state = run_epoch(session, m, train_users, train_label, train_op, verbose=True)
                print("Epoch: %d Train Metrics:\n rmse: %.3f \t auc: %.3f \t accuracy: %.3f \n" % (i + 1, rmse, auc, accuracy))
                with open(log_file_path, "a+") as f:
                    f.write("Epoch: %d Train Metrics:\n rmse: %.3f \t auc: %.3f \t accuracy: %.3f \n" % (i + 1, rmse, auc, accuracy))
                if((i+1) % FLAGS.evaluation_interval == 0):
                    print "Save variables to disk"
                    # save_path = saver.save(session, model_name)#
                    print("*"*10)
#                     print("Start to test model....")
#                     rmse, auc, r2, _ = run_epoch(session, mtest, test_students, tf.no_op())
#                     print("Epoch: %d Test Metrics:\n rmse: %.3f \t auc: %.3f \t r2: %.3f" % (i+1, rmse, auc, r2))
#                     with open(log_file_path, "a+") as f:
#                         f.write("Epoch: %d Test Metrics:\n rmse: %.3f \t auc: %.3f \t r2: %.3f" % ((i+1) , rmse, auc, r2))
#                         f.write("\n")

#                         print("*"*10)
            rmse, auc, accuracy, final_state = run_epoch(session, mtest, test_users, test_label, tf.no_op())
                # c, h = final_state
                # if len(cs) < 1:
                #     cs = c
                # else 
    


# In[20]:


if __name__ == "__main__":
    tf.app.run()

