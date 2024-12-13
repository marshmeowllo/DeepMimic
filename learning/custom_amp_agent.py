import numpy as np
import tensorflow as tf
from learning.amp_agent import AMPAgent
from rich.console import Console
import warnings

console = Console()
warnings.filterwarnings("ignore", category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class CustomAMPAgent(AMPAgent):
    NAME = "CustomAMP"

    def __init__(self, id, world, json_data):
        super().__init__(id, world, json_data)
        console.print("CustomAMPAgent initialized with ID: " + str(id), style="bold blue")


    def _build_nets(self, json_data):
        super()._build_nets(json_data)

        assert self.DISC_NET_KEY in json_data

        console.print("build nets call", style="bold blue")
        disc_net_name = json_data[self.DISC_NET_KEY]
        disc_init_output_scale = 1 if (self.DISC_INIT_OUTPUT_SCALE_KEY not in json_data) else json_data[self.DISC_INIT_OUTPUT_SCALE_KEY]
        self._reward_scale = 1.0 if (self.REWARD_SCALE_KEY not in json_data) else json_data[self.REWARD_SCALE_KEY]
        
        amp_obs_size = self._get_amp_obs_size()

        self._amp_obs_expert_ph = tf.placeholder(tf.float32, shape=[None, amp_obs_size], name="amp_obs_expert")
        self._amp_obs_agent_ph = tf.placeholder(tf.float32, shape=[None, amp_obs_size], name="amp_obs_agent")
        
        self._disc_expert_inputs = self._get_disc_expert_inputs()
        self._disc_agent_inputs = self._get_disc_agent_inputs()

        with tf.variable_scope(self.MAIN_SCOPE):
            with tf.variable_scope(self.DISC_SCOPE):
                self._disc_logits_expert_tf = self._build_custom_disc_net(disc_net_name, self._disc_expert_inputs, disc_init_output_scale)
                self._disc_logits_agent_tf = self._build_custom_disc_net(disc_net_name, self._disc_agent_inputs, disc_init_output_scale, reuse=True)

        if self._disc_logits_expert_tf is not None:
            console.print("Built custom discriminator net: " + disc_net_name, style="bold blue")

        # Additional outputs for training
        self._disc_prob_agent_tf = tf.sigmoid(self._disc_logits_agent_tf)
        self._abs_logit_agent_tf = tf.reduce_mean(tf.abs(self._disc_logits_agent_tf))
        self._avg_prob_agent_tf = tf.reduce_mean(self._disc_prob_agent_tf)

    def _build_custom_disc_net(self, net_name, input_tfs, init_output_scale, reuse=False):
        out_size = 1

        with tf.variable_scope("custom_fc_cnn_net", reuse=reuse):
            # Reshape input for CNN
            # from VGG papers
            input_tfs_reshaped = tf.reshape(input_tfs, [-1, 64, 64, 3])  # Example input shape

            # VGG Convolutional Layers
            h = tf.layers.conv2d(input_tfs_reshaped, filters=64, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu)
            h = tf.layers.conv2d(h, filters=64, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu)
            h = tf.layers.max_pooling2d(h, pool_size=2, strides=2)

            h = tf.layers.conv2d(h, filters=128, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu)
            h = tf.layers.conv2d(h, filters=128, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu)
            h = tf.layers.max_pooling2d(h, pool_size=2, strides=2)

            h = tf.layers.conv2d(h, filters=256, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu)
            h = tf.layers.conv2d(h, filters=256, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu)
            h = tf.layers.max_pooling2d(h, pool_size=2, strides=2)

            h = tf.layers.flatten(h)

            fc1 = tf.layers.dense(h, units=1024, activation=None, name="fc1")
            gate1 = tf.layers.dense(h, units=1024, activation=tf.nn.sigmoid, name="gate1")
            gated_fc1 = tf.multiply(fc1, gate1)

            fc2 = tf.layers.dense(gated_fc1, units=1024, activation=None, name="fc2")
            gate2 = tf.layers.dense(gated_fc1, units=1024, activation=tf.nn.sigmoid, name="gate2")
            gated_fc2 = tf.multiply(fc2, gate2) 

            # Output layer for logits
            logits_tf = tf.layers.dense(gated_fc2, units=out_size, activation=None,
                                        kernel_initializer=tf.random_uniform_initializer(minval=-init_output_scale, maxval=init_output_scale),
                                        name=self.DISC_LOGIT_NAME)
        return logits_tf
