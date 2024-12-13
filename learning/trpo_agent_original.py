# File: trpo_agent.py
import numpy as np
import tensorflow as tf
from learning.pg_agent import PGAgent
import util.mpi_util as MPIUtil
from util.logger import Logger
import learning.nets.net_builder as NetBuilder
from learning.solvers.mpi_solver import MPISolver
from learning.tf_distribution_gaussian_diag import TFDistributionGaussianDiag
from env.env import Env
import learning.rl_util as RLUtil

import os
os.environ["OMP_NUM_THREADS"] = "4"

class TRPOAgent(PGAgent):
    NAME = "TRPO"

    EPOCHS_KEY = "Epochs"
    BATCH_SIZE_KEY = "BatchSize"
    TD_LAMBDA_KEY = "TDLambda"
    KL_DELTA_KEY = "KLDelta"
    CG_ITERS_KEY = "CGIters"
    LINE_SEARCH_STEPS = "LineSearchSteps"
    LINE_SEARCH_ACCEPT_RATIO = "LineSearchAcceptRatio"

    ADV_EPS = 1e-5

    def __init__(self, world, id, json_data):
        super().__init__(world, id, json_data)
    
    def _load_params(self, json_data):
        """
        Load hyperparameters from the provided JSON data.
        Args:
            json_data (dict): A dictionary containing configuration parameters.
        """

        super()._load_params(json_data)
        # Number of training epochs per update
        self.epochs = 1 if (self.EPOCHS_KEY not in json_data) else json_data[self.EPOCHS_KEY]
        # Batch size for updates
        self.batch_size = 1024 if (self.BATCH_SIZE_KEY not in json_data) else json_data[self.BATCH_SIZE_KEY]
        self.td_lambda = 0.95 if (self.TD_LAMBDA_KEY not in json_data) else json_data[self.TD_LAMBDA_KEY]
        self.delta = json_data.get(self.KL_DELTA_KEY, 0.01)  # KLConstraint
        self.cg_iters = json_data.get(self.CG_ITERS_KEY, 10)  # Conjugate gradient iterations
        self.backtrack_coeff = json_data.get("BacktrackCoeff", 0.8)
        self.backtrack_iters = json_data.get("BacktrackIters", 10)

         # Adjust batch size to account for multiple processes
        num_procs = MPIUtil.get_num_procs()
        self._local_batch_size = int(np.ceil(self.batch_size / num_procs))

        # Ensure replay buffer size is sufficient to avoid overflow
        min_replay_size = 2 * self._local_batch_size # needed to prevent buffer overflow
        assert(self.replay_buffer_size > min_replay_size)

        self.replay_buffer_size = np.maximum(min_replay_size, self.replay_buffer_size)
    
    def _build_nets(self, json_data):
        """
        Build the actor and critic neural networks.
        Args:
            json_data (dict): A dictionary containing network configuration parameters.
        """

        assert self.ACTOR_NET_KEY in json_data
        assert self.CRITIC_NET_KEY in json_data

        actor_net_name = json_data[self.ACTOR_NET_KEY]
        critic_net_name = json_data[self.CRITIC_NET_KEY]
        actor_init_output_scale = 1 if (self.ACTOR_INIT_OUTPUT_SCALE_KEY not in json_data) else json_data[self.ACTOR_INIT_OUTPUT_SCALE_KEY]

        s_size = self.get_state_size()
        g_size = self.get_goal_size()
        a_size = self.get_action_size()

        # Placeholders for inputs
        self._s_ph = tf.placeholder(tf.float32, shape=[None, s_size], name="states")
        self._g_ph = tf.placeholder(tf.float32, shape=([None, g_size] if self.has_goal() else None), name="g")
        self._a_ph = tf.placeholder(tf.float32, shape=[None, a_size], name="actions")
        self._adv_ph = tf.placeholder(tf.float32, shape=[None], name="advantages")
        self._tar_val_ph = tf.placeholder(tf.float32, shape=[None], name="tar_val")

        with tf.variable_scope('main'):
            self._norm_a_pd_tf = self._build_net_actor(actor_net_name, self._s_ph, actor_init_output_scale)
            self._critic_tf = self._build_net_critic(critic_net_name, self._s_ph)

            # Collect actor parameters
            self._actor_params = tf.trainable_variables(scope=tf.get_variable_scope().name + '/actor')
        
        # Logging network build completion
        if self._norm_a_pd_tf is not None:
            Logger.print("Built actor net: " + actor_net_name)

        if self._critic_tf is not None:
            Logger.print("Built critic net: " + critic_net_name)
        
        sample_norm_a_tf = self._norm_a_pd_tf.sample()
        self._sample_a_tf = self._a_norm.unnormalize_tf(sample_norm_a_tf)
        self._sample_a_logp_tf = self._norm_a_pd_tf.logp(sample_norm_a_tf)
        
        mode_norm_a_tf = self._norm_a_pd_tf.get_mode()
        self._mode_a_tf = self._a_norm.unnormalize_tf(mode_norm_a_tf)
        self._mode_a_logp_tf = self._norm_a_pd_tf.logp(mode_norm_a_tf)
        
        norm_tar_a_tf = self._a_norm.normalize_tf(self._a_ph)
        self._a_logp_tf = self._norm_a_pd_tf.logp(norm_tar_a_tf)

        Logger.print('Built actor net: ' + actor_net_name)
        Logger.print('Built critic net: ' + critic_net_name)

        return

                            

    def _build_losses(self, json_data):
        # Penalty weight for action bounds
        actor_bound_loss_weight = 10.0
        # Regularization for actor
        actor_weight_decay = 0 if (self.ACTOR_WEIGHT_DECAY_KEY not in json_data) else json_data[self.ACTOR_WEIGHT_DECAY_KEY]
        # Regularization for critic
        critic_weight_decay = 0 if (self.CRITIC_WEIGHT_DECAY_KEY not in json_data) else json_data[self.CRITIC_WEIGHT_DECAY_KEY]

        # Critic loss
        self._target_values = tf.placeholder(tf.float32, shape=[None], name="target_values")
        self._critic_loss_tf = 0.5 * tf.reduce_mean(tf.square(self._critic_tf - self._target_values))

        # Add weight decay to critic loss (L2 regularization)
        if critic_weight_decay > 0:
            self._critic_loss_tf += critic_weight_decay * self._weight_decay_loss(self.MAIN_SCOPE + '/critic')

        # Surrogate loss
        surrogate_adv_tf = self._a_logp_tf * self._adv_ph
        self._actor_loss_tf = -tf.reduce_mean(surrogate_adv_tf)

        # Add action bound loss
        if actor_bound_loss_weight > 0.0:
            self._actor_loss_tf += actor_bound_loss_weight * self._build_action_bound_loss(self._norm_a_pd_tf)

        # Add weight decay to actor loss (L2 regularization)
        if actor_weight_decay > 0:
            self._actor_loss_tf += actor_weight_decay * self._weight_decay_loss(self.MAIN_SCOPE + '/actor')

        return

    def _build_solvers(self, json_data):
        # Debug variables to ensure they are created
        self._debug_tf_vars()
        actor_stepsize = 0.001 if (self.ACTOR_STEPSIZE_KEY not in json_data) else json_data[self.ACTOR_STEPSIZE_KEY]
        actor_momentum = 0.9 if (self.ACTOR_MOMENTUM_KEY not in json_data) else json_data[self.ACTOR_MOMENTUM_KEY]
        critic_stepsize = 0.01 if (self.CRITIC_STEPSIZE_KEY not in json_data) else json_data[self.CRITIC_STEPSIZE_KEY]
        critic_momentum = 0.9 if (self.CRITIC_MOMENTUM_KEY not in json_data) else json_data[self.CRITIC_MOMENTUM_KEY]

        # Critic optimizer
        critic_vars = self._tf_vars(self.MAIN_SCOPE + '/critic')
        critic_opt = tf.train.MomentumOptimizer(learning_rate=critic_stepsize, momentum=critic_momentum)
        self._critic_grad_tf = tf.gradients(self._critic_loss_tf, critic_vars)
        self._critic_solver = MPISolver(self.sess, critic_opt, critic_vars)

        # Actor solver
        actor_vars = self._tf_vars(self.MAIN_SCOPE + '/actor')
        actor_opt = tf.train.MomentumOptimizer(learning_rate=actor_stepsize, momentum=actor_momentum)
        self._actor_grad_tf = tf.gradients(self._actor_loss_tf, actor_vars)
        self._actor_solver = MPISolver(self.sess, actor_opt, actor_vars)

        return


    def _debug_tf_vars(self):
        print("Trainable Variables in Graph:")
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            print(f"{var.name} - {var.shape}")


    def _compute_kl_divergence(self, states):
        """
        Compute KL divergence between old and new policies.
        Args:
            states (numpy array): Batch of states.
        Returns:
            tf.Tensor: KL divergence computation graph.
        """
        # Fetch old mean and log standard deviation
        old_mean, old_log_std = self.sess.run(
            [self._norm_a_pd_tf.get_mean(), self._norm_a_pd_tf.get_logstd()],
            feed_dict={self._s_ph: states}  # Provide states array
        )

        # New mean and log standard deviation
        new_mean = self._norm_a_pd_tf.get_mean()
        new_log_std = self._norm_a_pd_tf.get_logstd()

        # Compute KL divergence
        kl = tf.reduce_mean(
            tf.reduce_sum(
                new_log_std - old_log_std +
                (tf.exp(2 * old_log_std) + tf.square(old_mean - new_mean)) / (2 * tf.exp(2 * new_log_std)) - 0.5,
                axis=1
            )
        )
        return kl


    
    def fisher_vector_product(self, vector):
        """
        Compute the Fisher Information Matrix (FIM) vector product.
        Args:
            vector (numpy array): The vector to multiply with the Fisher Information Matrix.
        Returns:
            numpy array: The resulting vector after multiplication with FIM.
        """
        kl_divergence = self._compute_kl_divergence(self._current_states)
        kl_grads = tf.gradients(kl_divergence, self._actor_params)
        flat_kl_grads = tf.concat([tf.reshape(g, [-1]) for g in kl_grads], axis=0)

        # Compute directional derivative
        kl_grad_vector_product = tf.reduce_sum(flat_kl_grads * vector)

        # Compute FIM vector product
        fisher_vector = tf.gradients(kl_grad_vector_product, self._actor_params)
        fisher_vector = tf.concat([tf.reshape(v, [-1]) for v in fisher_vector], axis=0)

        # Evaluate the Fisher-vector product
        fisher_vector_value = self.sess.run(fisher_vector, feed_dict={self._s_ph: self._current_states})

        return fisher_vector_value


    def _line_search(self, old_params, full_step, states):
        """
        Perform line search to find the maximum step size that satisfies the KL divergence constraint.
        Args:
            old_params (numpy array): Current policy parameters.
            full_step (numpy array): Step direction and size for updating parameters.
            states (numpy array): States used to compute KL divergence.
        Returns:
            numpy array: Updated policy parameters after line search.
        """
        for step_frac in [self.backtrack_coeff**i for i in range(self.backtrack_iters)]:
            new_params = old_params + step_frac * full_step
            self._set_policy_params(new_params)

            # Compute KL divergence
            kl_tensor = self._compute_kl_divergence(states)  # Feed the states array
            kl = self.sess.run(kl_tensor, feed_dict={self._s_ph: states})  # Provide the feed_dict for `_s_ph`

            # Check if the KL divergence is within the constraint
            if kl < self.delta:
                return new_params

        # If no step satisfies the constraint, return the original parameters
        return old_params




    def conjugate_gradient(self, fisher_vector_product, gradient, max_iters=10, tol=1e-10):
        """
        Solve Hx = g using the conjugate gradient method, where A is the Fisher Information Matrix.
        Args:
            fisher_vector_product (callable): Function to compute FIM-vector product.
            gradient (numpy array): Gradient vector b.
            max_iters (int): Maximum iterations.
            tol (float): Tolerance for convergence.
        Returns:
            numpy array: Solution vector x.
        """
        gradient = np.asarray(gradient, dtype=np.float32)  # Ensure gradient is numpy array
        assert gradient.ndim == 1, "Gradient should be a 1D array."

        x = np.zeros_like(gradient)  # Initialize solution vector with the same shape as gradient
        r = gradient.copy()          # Residual vector, initially equal to the gradient
        p = gradient.copy()          # Search direction, initially equal to the gradient
        r_dot_r = np.dot(r, r)       # Residual squared norm

        for _ in range(max_iters):
            Ap = fisher_vector_product(p)  # FIM-vector product
            assert Ap.shape == p.shape, f"Shape mismatch: Ap shape {Ap.shape}, p shape {p.shape}"

            alpha = r_dot_r / (np.dot(p, Ap) + 1e-8)  # Step size scalar

            x += alpha * p  # Update solution
            r -= alpha * Ap  # Update residual

            new_r_dot_r = np.dot(r, r)  # New residual squared norm

            # Convergence check
            if np.sqrt(new_r_dot_r) < tol:
                break

            beta = new_r_dot_r / r_dot_r  # Update scalar for conjugate direction
            p = r + beta * p  # Update search direction
            r_dot_r = new_r_dot_r  # Update residual squared norm

        return x

    def _train_step(self):
        start_idx = self.replay_buffer.buffer_tail
        end_idx = self.replay_buffer.buffer_head
        assert(start_idx == 0)
        assert(self.replay_buffer.get_current_size() <= self.replay_buffer.buffer_size)
        assert(start_idx < end_idx)

        idx = np.array(list(range(start_idx, end_idx)))
        end_mask = self.replay_buffer.is_path_end(idx)
        end_mask = np.logical_not(end_mask)

        rewards = self._fetch_batch_rewards(start_idx, end_idx)
        vals = self._compute_batch_vals(start_idx, end_idx)
        new_vals = self._compute_batch_new_vals(start_idx, end_idx, rewards, vals)

        valid_idx = idx[end_mask]
        exp_idx = self.replay_buffer.get_idx_filtered(self.EXP_ACTION_FLAG).copy()
        num_valid_idx = valid_idx.shape[0]
        num_exp_idx = exp_idx.shape[0]
        exp_idx = np.column_stack([exp_idx, np.array(list(range(0, num_exp_idx)), dtype=np.int32)])

        local_sample_count = valid_idx.size
        global_sample_count = int(MPIUtil.reduce_sum(local_sample_count))
        mini_batches = int(np.ceil(global_sample_count / self._local_batch_size))

        adv = new_vals[exp_idx[:, 0]] - vals[exp_idx[:, 0]]
        new_vals = np.clip(new_vals, self.val_min, self.val_max)

        adv_mean = np.mean(adv)
        adv_std = np.std(adv)
        adv = (adv - adv_mean) / (adv_std + self.ADV_EPS)

        critic_loss = 0
        actor_loss = 0

        for epoch in range(self.epochs):
            np.random.shuffle(valid_idx)
            np.random.shuffle(exp_idx)

            for b in range(mini_batches):
                batch_idx_beg = b * self._local_batch_size
                batch_idx_end = batch_idx_beg + self._local_batch_size

                critic_batch = np.array(range(batch_idx_beg, batch_idx_end), dtype=np.int32)
                actor_batch = critic_batch.copy()
                critic_batch = np.mod(critic_batch, num_valid_idx)
                actor_batch = np.mod(actor_batch, num_exp_idx)

                critic_batch = valid_idx[critic_batch]
                actor_batch = exp_idx[actor_batch]

                critic_batch_vals = new_vals[critic_batch]
                actor_batch_adv = adv[actor_batch[:, 1]]

                critic_s = self.replay_buffer.get('states', critic_batch)
                critic_g = self.replay_buffer.get('goals', critic_batch) if self.has_goal() else None
                self._update_critic(critic_s, critic_g, critic_batch_vals)

                actor_s = self.replay_buffer.get('states', actor_batch[:, 0])
                actor_g = self.replay_buffer.get('goals', actor_batch[:, 0]) if self.has_goal() else None
                actor_a = self.replay_buffer.get('actions', actor_batch[:, 0])

                self._current_states = actor_s

                self._update_actor(actor_s, actor_g, actor_a, actor_batch_adv)

        critic_loss = MPIUtil.reduce_avg(critic_loss)
        actor_loss = MPIUtil.reduce_avg(actor_loss)

        self.logger.log_tabular('Critic_Loss', critic_loss)
        self.logger.log_tabular('Actor_Loss', actor_loss)
        self.logger.log_tabular('Adv_Mean', adv_mean)
        self.logger.log_tabular('Adv_Std', adv_std)

    def _update_critic(self, s, g, tar_vals):

        feed = {
            self._s_ph: s,
            self._g_ph: g,
            self._target_values: tar_vals
        }
        
        loss, grads = self.sess.run([self._critic_loss_tf, self._critic_grad_tf], feed_dict=feed)
        self._critic_solver.update(grads)
        return loss

    def _update_actor(self, s, g, a, adv):
        feed = {
            self._s_ph: s,
            self._g_ph: g,
            self._a_ph: a,
            self._adv_ph: adv
        }

        # Compute gradients of actor loss with respect to actor parameters
        gradients = tf.gradients(self._actor_loss_tf, self._actor_params)
        flat_gradients = tf.concat([tf.reshape(g, [-1]) for g in gradients], axis=0)
        policy_gradient = self.sess.run(flat_gradients, feed_dict=feed)

        def fisher_vector_product(p):
            return self.fisher_vector_product(p)

        # Ensure policy_gradient is 1D
        policy_gradient = np.asarray(policy_gradient, dtype=np.float32).flatten()

        full_step = self.conjugate_gradient(fisher_vector_product, policy_gradient, max_iters=self.cg_iters)
        step_size = np.sqrt(2 * self.delta / (np.dot(full_step, fisher_vector_product(full_step)) + 1e-8))

        old_params = self._get_policy_params()
        new_params = self._line_search(old_params, full_step * step_size, s)
        self._set_policy_params(new_params)

        return policy_gradient

    def _fetch_batch_rewards(self, start_idx, end_idx):
        """
        Fetch rewards from the replay buffer for the specified index range.
        Args:
            start_idx (int): Start index in the replay buffer.
            end_idx (int): End index in the replay buffer.
        Returns:
            numpy array of rewards for the specified index range.
        """
        rewards = self.replay_buffer.get('rewards', np.arange(start_idx, end_idx))
        return rewards

    def _compute_batch_vals(self, start_idx, end_idx):
        """
        Compute value estimates for a batch of states.
        Args:
            start_idx (int): Start index in the replay buffer.
            end_idx (int): End index in the replay buffer.
        Returns:
            numpy array: Value estimates for the batch.
        """

        # Fetch states and goals
        states = self.replay_buffer.get_all("states")[start_idx:end_idx]
        goals = self.replay_buffer.get_all("goals")[start_idx:end_idx] if self.has_goal() else None

        # Identify terminal flags
        idx = np.arange(start_idx, end_idx)
        is_end = self.replay_buffer.is_path_end(idx)
        is_fail = np.logical_and(is_end, self.replay_buffer.check_terminal_flag(idx, Env.Terminate.Fail))
        is_succ = np.logical_and(is_end, self.replay_buffer.check_terminal_flag(idx, Env.Terminate.Succ))

        # Evaluate critic for non-terminal states
        vals = self._eval_critic(states, goals)

        # Assign values for terminal states
        vals[is_fail] = self.val_fail
        vals[is_succ] = self.val_succ

        return vals


    def _compute_batch_new_vals(self, start_idx, end_idx, rewards, val_buffer):
        if self.discount == 0:
            new_vals = rewards.copy()
        else:
            new_vals = np.zeros_like(val_buffer)

            curr_idx = start_idx
            while curr_idx < end_idx:
                idx0 = curr_idx - start_idx
                idx1 = self.replay_buffer.get_path_end(curr_idx) - start_idx
                r = rewards[idx0:idx1]
                v = val_buffer[idx0:(idx1 + 1)]

                new_vals[idx0:idx1] = RLUtil.compute_return(r, self.discount, self.td_lambda, v)
                curr_idx = idx1 + start_idx + 1
        
        return new_vals
    
    def _get_policy_params(self):
        """
        Get the current parameters of the actor network.
        Returns:
            numpy array: Flattened array of actor network parameters.
        """
        actor_params = self.sess.run(self._actor_params)
        flat_params = np.concatenate([param.flatten() for param in actor_params])
        return flat_params


    def _set_policy_params(self, flat_params):
        """
        Set the parameters of the actor network.
        Args:
            flat_params (numpy array): Flattened array of new parameters to set.
        """
        start = 0
        assign_ops = []
        for var in self._actor_params:
            shape = var.shape.as_list()
            size = np.prod(shape)
            param_values = flat_params[start:start + size].reshape(shape)
            assign_ops.append(tf.assign(var, param_values))
            start += size

        self.sess.run(assign_ops)

    def _train(self):
        super()._train()
        self.replay_buffer.clear()
        return
    
    def _fetch_batch_rewards(self, start_idx, end_idx):
        rewards = self.replay_buffer.get_all("rewards")[start_idx:end_idx]
        return rewards

    def _get_iters_per_update(self):
        return 1

    def _valid_train_step(self):
        samples = self.replay_buffer.get_current_size()
        exp_samples = self.replay_buffer.count_filtered(self.EXP_ACTION_FLAG)
        return (samples >= self._local_batch_size) and (exp_samples >= self._local_mini_batch_size)