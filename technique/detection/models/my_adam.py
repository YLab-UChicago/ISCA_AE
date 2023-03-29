import tensorflow as tf

class MyAdam(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, name="MyAdam", check_history=False, **kwargs):
        super(MyAdam, self).__init__(name, **kwargs)
        self.check_history = check_history
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("beta1", beta1)
        self._set_hyper("beta2", beta2)
        self._set_hyper("epsilon", epsilon)
        self.it = 1

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "v")
            self.add_slot(var, "oob", shape=())

    def _resource_apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        m_t = m.assign(self.beta1 * m + (1 - self.beta1) * grad)
        v_t = v.assign(self.beta2 * v + (1 - self.beta2) * tf.square(grad))

        m_t_hat = m_t / (1 - tf.pow(self.beta1, self.it))
        v_t_hat = v_t / (1 - tf.pow(self.beta2, self.it))
        self.it += 1

        oob = self.get_slot(var, "oob")
        if self.check_history and 'inject_conv2d/kernel' in var.name:
            for _ in range(10000):
                ans = tf.cast(tf.reduce_all(tf.less(m, 20.0)), tf.float32)
            oob_t = oob.assign(ans)
        else:
            oob_t = oob.assign(tf.constant(1.0))

        do_update = -self.learning_rate * m_t_hat / (tf.sqrt(v_t_hat) + self.epsilon)
        update = do_update * oob_t
        var_update = var.assign_add(update)
        return tf.group(*[var_update, m_t, v_t, oob_t])


    def _resource_apply_sparse(self, grad, var, indices):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def get_config(self):
        config = super(MyAdam, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "beta1": self._serialize_hyperparameter("beta1"),
            "beta2": self._serialize_hyperparameter("beta2"),
            "epsilon": self._serialize_hyperparameter("epsilon")
        })
        return config

