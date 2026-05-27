import tensorflow as tf

class NestedSamplerTF:
    def __init__(self, log_prob_theta, cov, mean=None, batch_size=4096, jit_compile=True):
        """
        log_prob_theta: function(theta) -> log-likelihood + log-prior
        cov: (dim, dim)
        mean: (dim,)
        """
        self.log_prob_theta = log_prob_theta
        self.cov = tf.convert_to_tensor(cov, tf.float32)

        self.dim = self.cov.shape[0]

        if mean is None:
            self.mean = tf.zeros(self.dim, dtype=tf.float32)
        else:
            self.mean = tf.convert_to_tensor(mean, tf.float32)

        self.chol = tf.linalg.cholesky(self.cov)
        self.log_det = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.chol)))

        self.batch_size = batch_size

    @tf.function(jit_compile=jit_compile)
    def z_to_theta(self, z):
        return self.mean + tf.linalg.matmul(z, self.chol, transpose_b=True)

    @tf.function(jit_compile=jit_compile)
    def log_prob_z(self, z):
        theta = self.z_to_theta(z)
        return self.log_prob_theta(theta) + self.log_det  # Jacobian included

    @tf.function(jit_compile=jit_compile)
    def find_worst(self, logl):
        idx = tf.argmin(logl)
        return idx, logl[idx]

    @tf.function(jit_compile=jit_compile)
    def sample_constrained(self, logl_min):
        def cond(zs, logls, n):
            return n < self.n_live

        def body(zs, logls, n):
            z = tf.random.normal((self.batch_size, self.dim))
            logl = self.log_prob_z(z)

            mask = logl > logl_min
            z_acc = tf.boolean_mask(z, mask)
            logl_acc = tf.boolean_mask(logl, mask)

            zs = tf.concat([zs, z_acc], axis=0)
            logls = tf.concat([logls, logl_acc], axis=0)
            n = tf.shape(zs)[0]

            return zs, logls, n

        zs = tf.zeros((0, self.dim), tf.float32)
        logls = tf.zeros((0,), tf.float32)
        n = tf.constant(0)

        zs, logls, _ = tf.while_loop(
            cond, body, [zs, logls, n],
            shape_invariants=[
                tf.TensorShape([None, self.dim]),
                tf.TensorShape([None]),
                tf.TensorShape([])
            ]
        )

        return zs[:self.n_live], logls[:self.n_live]

    def run(self, init_live_points, n_iter=1000):
        """
        init_live_points: (n_live, dim) in *z-space*
        """
        live_z = tf.convert_to_tensor(init_live_points, tf.float32)
        self.n_live = live_z.shape[0]

        live_logl = self.log_prob_z(live_z)

        logZ = tf.constant(-1e30, tf.float32)
        logX_prev = tf.constant(0.0, tf.float32)

        samples = []
        weights = []

        for i in range(n_iter):
            idx, worst_logl = self.find_worst(live_logl)

            logX = -tf.cast(i + 1, tf.float32) / self.n_live
            logWt = tf.math.reduce_logsumexp([logX_prev, logX]) + worst_logl
            logZ = tf.math.reduce_logsumexp([logZ, logWt])

            logX_prev = logX

            samples.append(live_z[idx])
            weights.append(logWt)

            new_z, new_logl = self.sample_constrained(worst_logl)

            live_z = tf.tensor_scatter_nd_update(live_z, [[idx]], [new_z[0]])
            live_logl = tf.tensor_scatter_nd_update(live_logl, [[idx]], [new_logl[0]])

            if i % 50 == 0:
                print(f"Iter {i} logZ ≈ {logZ.numpy():.3f}")

        return tf.stack(samples), tf.stack(weights), logZ
