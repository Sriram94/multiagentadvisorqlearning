import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import gym
np.random.seed(1)
tf.set_random_seed(1)



GAMMA = 0.9     
LR_A = 0.001    
LR_C = 0.01     


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "Advisorac_state")
        self.a = tf.placeholder(tf.int32, None, "Advisorac_act")
        self.td_error = tf.placeholder(tf.float32, None, "Advisorac_td_error")  

        with tf.variable_scope('Admiraldmacactorvalue'):
            self.name_scope = tf.get_variable_scope().name
            with tf.variable_scope('Advisorac_Actor'):
                l1 = tf.layers.dense(
                    inputs=self.s,
                    units=50,    
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),    
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='Advisorac_l1'
                )

                l2 = tf.layers.dense(
                    inputs=l1,
                    units=50,    
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),    
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='Advisorac_lh1'
                )
                
                self.acts_prob = tf.layers.dense(
                    inputs=l2,
                    units=n_actions,    
                    activation=tf.nn.softmax,   
                    kernel_initializer=tf.random_normal_initializer(0., .1),  
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='Advisorac_acts_prob'
                )

            with tf.variable_scope('Advisorac_exp_v'):
                log_prob = tf.log(self.acts_prob[0, self.a])
                self.exp_v = tf.reduce_mean(log_prob * self.td_error)  

            with tf.variable_scope('Advisorac_train'):
                self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   


    def save_model(self, s):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        save_path = saver.save(self.sess, s)
        print("Model saved in path: %s" % save_path)

    def restore_model(self, s):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        saver.restore(self.sess, s)
        print("Model restored")





class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features+7], "Advisorac_state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "Advisorac_v_next")
        self.r = tf.placeholder(tf.float32, None, 'Advisorac_r')

        with tf.variable_scope('AdmiraldmacCritic'):
            self.name_scope = tf.get_variable_scope().name
            with tf.variable_scope('Advisorac_Critic'):
                l1 = tf.layers.dense(
                    inputs=self.s,
                    units=50,  
                    activation=tf.nn.relu,  
                    kernel_initializer=tf.random_normal_initializer(0., .1),  
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='Advisorac_l1'
                )

                l2 = tf.layers.dense(
                    inputs=l1,
                    units=50,    
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),    
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='Advisorac_lh1'
                )
                
                self.v = tf.layers.dense(
                    inputs=l2,
                    units=1,  
                    activation=None,
                    kernel_initializer=tf.random_normal_initializer(0., .1),  
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='Advisorac_V'
                )

            with tf.variable_scope('squared_TD_error'):
                self.td_error = self.r + GAMMA * self.v_ - self.v
                self.loss = tf.square(self.td_error)    
            with tf.variable_scope('Advisorac_train'):
                self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, opp_a, r, s_, opp_a_):
        s = list(s)


        for i in range(len(opp_a)):
            new_a = opp_a[i]
            new_a = float(new_a)
            s.append(new_a)


        s = np.array(s)
        
        s_ = list(s_)

        for i in range(len(opp_a_)):
            new_a = opp_a_[i]
            new_a = float(new_a)
            s_.append(new_a)

        s_ = np.array(s_)
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        
        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error

    def save_model(self, s):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        save_path = saver.save(self.sess, s)
        print("Model saved in path: %s" % save_path)

    def restore_model(self, s):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        saver.restore(self.sess, s)
        print("Model restored")
