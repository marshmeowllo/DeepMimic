import numpy as np
import tensorflow as tf
from learning.ppo_agent import PPOAgent
from learning.pg_agent import PGAgent
from util.logger import Logger
import learning.solvers.mpi_solver as mpi_solver
import util.mpi_util as MPIUtil

'''
Trust Region Policy Optimization Agent
'''

class TRPOAgent(PPOAgent):
    NAME = "TRPO"

    KL_DELTA_KEY = "KLDelta"
    CG_ITERS_KEY = "CGIters"
    LINE_SEARCH_STEPS = "LineSearchSteps"
    LINE_SEARCH_ACCEPT_RATIO = "LineSearchAcceptRatio"

    def __init__(self, world, id: int, json_data: dict):
        """
        Initializes TRPOAgent with hyperparameters.

        Args:
            id (int): Agent ID.
            world: The environment/world object.
            json_data (dict): Configuration dictionary.
        """
        super().__init__(world, id, json_data)

    

    def _load_params(self, json_data: dict):
        """
        Loads TRPO-specific parameters from configuration.
        
        Args:
            json_data (dict): Configuration dictionary.
        """
        super()._load_params(json_data)

        # TRPO-specific hyperparameters
        self.delta = 0.01 if self.KL_DELTA_KEY not in json_data else json_data[self.KL_DELTA_KEY]
        self.cg_iters = 10 if self.CG_ITERS_KEY not in json_data else json_data[self.CG_ITERS_KEY]
        self.line_search_steps = 10 if self.LINE_SEARCH_STEPS not in json_data else json_data[self.LINE_SEARCH_STEPS]
        self.line_search_accept_ratio = 0.1 if self.LINE_SEARCH_ACCEPT_RATIO not in json_data else json_data[self.LINE_SEARCH_ACCEPT_RATIO]
        
        Logger.print(f"Initialized TRPO Agent with KLDelta: {self.delta}, CGIters: {self.cg_iters}")

    def _build_nets(self, json_data: dict):
        """
        Builds actor (policy) and critic (value function) networks.

        Args:
            json_data (dict): Configuration dictionary.
        """
        Logger.print("Building networks for TRPO...")
        super()._build_nets(json_data)  # Calls PPOAgent's _build_nets

        # Fetch the variables under the correct scope
        self.actor_net_vars = self._tf_vars(self.MAIN_SCOPE + '/actor')
        self.critic_net_vars = self._tf_vars(self.MAIN_SCOPE + '/critic')

        # Debugging: log actor and critic variables
        if not self.actor_net_vars:
            Logger.print(f"No actor variables found under scope: {self.MAIN_SCOPE}/actor")
        else:
            Logger.print(f"Actor variables: {[v.name for v in self.actor_net_vars]}")

        if not self.critic_net_vars:
            Logger.print(f"No critic variables found under scope: {self.MAIN_SCOPE}/critic")
        else:
            Logger.print(f"Critic variables: {[v.name for v in self.critic_net_vars]}")

    def _build_losses(self, json_data: dict):
        """
        Build loss functions specific to TRPO.
        """
        super()._build_losses(json_data)

        ratio_tf = tf.exp(self._a_logp_tf - self._old_logp_ph)
        self.surrogate_loss = tf.reduce_mean(ratio_tf * self._adv_ph)

        # KL Divergence for Gaussian policies
        mu_old, log_std_old = self._norm_a_pd_tf.get_mean(), self._norm_a_pd_tf.get_logstd()
        mu_new, log_std_new = self._norm_a_pd_tf.get_mean(), self._norm_a_pd_tf.get_logstd()

        std_old = tf.exp(log_std_old)
        std_new = tf.exp(log_std_new)

        kl = (
            log_std_new - log_std_old +
            (tf.square(std_old) + tf.square(mu_new - mu_old)) / (2.0 * tf.square(std_new)) - 0.5
        )

        self.kl_divergence = tf.reduce_mean(tf.reduce_sum(kl, axis=-1))
        Logger.print("Loss functions built for TRPO.")

    
    def _build_solvers(self, json_data):
        """
        Builds optimizers (solvers) for the actor and critic networks.
        """
        # Call the parent class's solver setup for critic
        super()._build_solvers(json_data)
        # Replace the actor solver with TRPO's conjugate gradient-based method
        self._actor_grad_tf = tf.gradients(self.surrogate_loss, self.actor_net_vars)
        Logger.print("TRPO uses conjugate gradient for actor updates.")

    def _conjugate_gradient(self, feed, grads, max_iters=10):
        """
        Conjugate Gradient implementation for TRPO.
        """
        b = np.concatenate([g.flatten() for g in grads], axis=0)
        x = np.zeros_like(b)
        r = b.copy()
        p = r.copy()
        rs_old = np.dot(r, r)

        for _ in range(max_iters):
            Ap = self._fisher_vector_product(feed, p)
            alpha = rs_old / (np.dot(p, Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            rs_new = np.dot(r, r)
            if np.sqrt(rs_new) < 1e-10:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x



    def _fisher_vector_product(self, feed, vector):
        """
        Compute the Fisher vector product (Hv).
        """
        grads = self.sess.run(
            tf.gradients(self.kl_divergence, self.actor_net_vars), feed_dict=feed
        )
        flat_grads = np.concatenate([np.reshape(g, [-1]) for g in grads if g is not None])
        fisher_vector = flat_grads * vector
        damping = 1e-5
        fisher_vector += damping * vector
        return fisher_vector


    def _line_search_and_update(self, step):
        """
        Perform line search to ensure KL constraint is satisfied.
        """
        old_params = self._get_flat_params()
        for alpha in np.linspace(1, 0, self.line_search_steps):
            new_params = old_params + alpha * step

            self._set_flat_params(new_params)

            kl_div = self.sess.run(self.kl_divergence)
            if kl_div <= self.delta:
                Logger.print(f"Line search succeeded with alpha={alpha}, KL={kl_div}")
                return
        Logger.print("Line search failed. Reverting parameters.")
        self._set_flat_params(old_params)


    # Additional methods like `_get_flat_params`, `_set_flat_params`, `_train_step`, and others are present above. 



    def _get_flat_params(self) -> np.ndarray:
        """Returns flattened parameters of the actor network."""
        flat_params = self.sess.run(tf.concat([tf.reshape(var, [-1]) for var in self.actor_net_vars], axis=0))
        return flat_params

    def _set_flat_params(self, flat_params: np.ndarray):
        """Sets actor network parameters from a flattened vector."""
        shapes = [var.get_shape().as_list() for var in self.actor_net_vars]
        offset = 0
        assigns = []

        for shape, var in zip(shapes, self.actor_net_vars):
            size = np.prod(shape)
            value = flat_params[offset:offset + size].reshape(shape)
            assigns.append(tf.compat.v1.assign(var, value))
            offset += size

        self.sess.run(assigns)

    def _update_actor(self, s, g, a, logp, adv):
        """
        Updates the actor network using simplified KL divergence.
        """
        feed = {
            self._s_ph: s,
            self._g_ph: g if self.has_goal() else None,
            self._a_ph: a,
            self._adv_ph: adv,
            self._old_logp_ph: logp
        }

        grads = self.sess.run(tf.gradients(self.surrogate_loss, self.actor_net_vars), feed_dict=feed)
        flat_grads = np.concatenate([g.flatten() for g in grads if g is not None])
        
        Logger.print(f"Gradient : {grads}")
        Logger.print(f"Flat Gradient : {flat_grads}")

        step_dir = flat_grads
        old_params = self._get_flat_params()

        for alpha in np.linspace(1.0, 0.0, 10):  # Line search
            new_params = old_params + alpha * step_dir
            self._set_flat_params(new_params)

            # Compute KL divergence
            kl_div = self.sess.run(self.kl_divergence, feed_dict=feed)
            if kl_div <= self.delta:
                Logger.print(f"Line search succeeded with alpha={alpha}, KL={kl_div}")
                break
        else:
            Logger.print("Line search failed. Reverting parameters.")
            self._set_flat_params(old_params)

        # Log surrogate loss after update
        surrogate_loss = self.sess.run(self.surrogate_loss, feed_dict=feed)
        Logger.print(f"Surrogate Loss after update: {surrogate_loss}")

        return surrogate_loss, 1  # clip_frac is not used in TRPO



    def _compute_advantages(self, rewards, values, gamma=0.99, lambda_=0.95):
        """
        Computes raw advantages without normalization.
        """
        deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
        advantages = np.zeros_like(deltas)
        running_add = 0

        for t in reversed(range(len(deltas))):
            running_add = deltas[t] + gamma * lambda_ * running_add
            advantages[t] = running_add

        Logger.print(f"Raw Rewards: {rewards}")
        Logger.print(f"Computed Advantages: {advantages}")

        return advantages



    
    def _compute_target_values(self, rewards, gamma=0.99):
        """
        Compute discounted returns as target values for critic.
        """
        target_values = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = rewards[t] + gamma * running_add
            target_values[t] = running_add
        return target_values



    def _train_step(self):
        """
        Executes a single training step for TRPO.
        This includes:
        - Data preparation (states, actions, rewards, advantages).
        - Updating the critic (value function).
        - Updating the actor (policy network) with trust region constraint.
        - Logging training metrics.
        """
        # Retrieve data from the replay buffer
        start_idx = self.replay_buffer.buffer_tail
        end_idx = self.replay_buffer.buffer_head
        assert start_idx < end_idx, "Replay buffer must contain valid data for training."
        if start_idx >= end_idx:
            Logger.print("Replay buffer contains insufficient data for training.")
            return


        idx = np.array(list(range(start_idx, end_idx)))        
        end_mask = self.replay_buffer.is_path_end(idx)
        end_mask = np.logical_not(end_mask) 
        
        rewards = self._fetch_batch_rewards(start_idx, end_idx)
        Logger.print(f"reward {rewards}")
        vals = self._compute_batch_vals(start_idx, end_idx)
        new_vals = self._compute_batch_new_vals(start_idx, end_idx, rewards, vals)

        valid_idx = idx[end_mask]
        exp_idx = self.replay_buffer.get_idx_filtered(self.EXP_ACTION_FLAG).copy()
        num_valid_idx = valid_idx.shape[0]
        num_exp_idx = exp_idx.shape[0]
        exp_idx = np.column_stack([exp_idx, np.array(list(range(0, num_exp_idx)), dtype=np.int32)])
        
        local_sample_count = valid_idx.size
        global_sample_count = int(mpi_solver.MPIUtil.reduce_sum(local_sample_count))
        mini_batches = int(np.ceil(global_sample_count / self.mini_batch_size))
        
        adv = new_vals[exp_idx[:,0]] - vals[exp_idx[:,0]]
        new_vals = np.clip(new_vals, self.val_min, self.val_max)

        adv_mean = np.mean(adv)
        adv_std = np.std(adv)
        adv = (adv - adv_mean) / (adv_std + self.ADV_EPS)
        adv = np.clip(adv, -self.norm_adv_clip, self.norm_adv_clip)

        critic_loss = 0
        actor_loss = 0
        actor_clip_frac = 0

        # Iterate over epochs
        for epoch in range(self.epochs):

            np.random.shuffle(exp_idx)
            
            for batch in range(mini_batches):

                batch_idx_beg = batch * self._local_mini_batch_size
                batch_idx_end = batch_idx_beg + self._local_mini_batch_size
                critic_batch = np.array(range(batch_idx_beg, batch_idx_end), dtype=np.int32)
                actor_batch = critic_batch.copy()
                critic_batch = np.mod(critic_batch, num_valid_idx)
                actor_batch = np.mod(actor_batch, num_exp_idx)
                shuffle_actor = (actor_batch[-1] < actor_batch[0]) or (actor_batch[-1] == num_exp_idx - 1)

                critic_batch = valid_idx[critic_batch]
                actor_batch = exp_idx[actor_batch]
                critic_batch_vals = new_vals[critic_batch]
                actor_batch_adv = adv[actor_batch[:,1]]

                critic_s = self.replay_buffer.get('states', critic_batch)
                critic_g = self.replay_buffer.get('goals', critic_batch) if self.has_goal() else None
                curr_critic_loss = self._update_critic(critic_s, critic_g, critic_batch_vals)

                actor_s = self.replay_buffer.get("states", actor_batch[:,0])
                actor_g = self.replay_buffer.get("goals", actor_batch[:,0]) if self.has_goal() else None
                actor_a = self.replay_buffer.get("actions", actor_batch[:,0])
                actor_logp = self.replay_buffer.get("logps", actor_batch[:,0])
                curr_actor_loss, curr_actor_clip_frac = self._update_actor(actor_s, actor_g, actor_a, actor_logp, actor_batch_adv)
                
                critic_loss += curr_critic_loss
                actor_loss += np.abs(curr_actor_loss)
                actor_clip_frac += curr_actor_clip_frac

                if (shuffle_actor):
                    np.random.shuffle(exp_idx)

            total_batches = mini_batches * self.epochs
            critic_loss /= total_batches
            actor_loss /= total_batches
            actor_clip_frac /= total_batches

            critic_loss = MPIUtil.reduce_avg(critic_loss)
            actor_loss = MPIUtil.reduce_avg(actor_loss)
            actor_clip_frac = MPIUtil.reduce_avg(actor_clip_frac)

            critic_stepsize = self._critic_solver.get_stepsize()
            actor_stepsize = self._actor_solver.get_stepsize()

            self.logger.log_tabular('Critic_Loss', critic_loss)
            self.logger.log_tabular('Critic_Stepsize', critic_stepsize)
            self.logger.log_tabular('Actor_Loss', actor_loss) 
            self.logger.log_tabular('Actor_Stepsize', actor_stepsize)
            self.logger.log_tabular('Clip_Frac', actor_clip_frac)
            self.logger.log_tabular('Adv_Mean', adv_mean)
            self.logger.log_tabular('Adv_Std', adv_std)


                # Log metrics after each mini-batch update
                # Logger.print(f"Epoch {epoch+1}, Mini-batch {i//self.mini_batch_size + 1}: Surrogate Loss {surrogate_loss}, Clip Fraction {clip_frac}")

            # Optionally log the metrics at the end of each epoch
            Logger.print(f"End of Epoch {epoch+1}: Critic Loss {critic_loss}")
        
        # 5. Log finish message when the step is completed
        Logger.print("Finish")
