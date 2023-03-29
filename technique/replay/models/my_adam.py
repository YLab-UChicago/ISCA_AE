import tensorflow as tf

class MyAdam(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.99, epsilon=1e-8, name="MyAdam", **kwargs):
        super(MyAdam, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("beta1", beta1)
        self._set_hyper("beta2", beta2)
        self._set_hyper("epsilon", epsilon)
        self.it = 1

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "v")

    def _resource_apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        m_t = m.assign(self.beta1 * m + (1 - self.beta1) * grad)
        v_t = v.assign(self.beta2 * v + (1 - self.beta2) * tf.square(grad))

        m_t_hat = m_t / (1 - tf.pow(self.beta1, self.it))
        v_t_hat = v_t / (1 - tf.pow(self.beta2, self.it))
        self.it += 1

        do_update = -self.learning_rate * m_t_hat / (tf.sqrt(v_t_hat) + self.epsilon)
        update = do_update
        var_update = var.assign_add(update)

        return tf.group(*[var_update, m_t, v_t])

    def _resource_apply_sparse(self, grad, var, indices):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def reset(self, grads, var_list):
        self.it -= 1
        for var,grad in zip(var_list, grads):
            m = self.get_slot(var, "m")
            v = self.get_slot(var, "v")
            m_t_hat = m / (1 - tf.pow(self.beta1, self.it))
            v_t_hat = v / (1 - tf.pow(self.beta2, self.it))

            m.assign((m - (1-self.beta1) * grad) / self.beta1)
            v.assign((v - (1-self.beta2) * tf.square(grad)) / self.beta2)
            update = self.learning_rate * m_t_hat / (tf.sqrt(v_t_hat) + self.epsilon)
            var_update = var.assign_add(update)


    def get_config(self):
        config = super(MyAdam, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "beta1": self._serialize_hyperparameter("beta1"),
            "beta2": self._serialize_hyperparameter("beta2"),
            "epsilon": self._serialize_hyperparameter("epsilon")
        })
        return config

