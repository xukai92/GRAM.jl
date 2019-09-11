def ring(batchsize, num_cluster=8, scale=1, std=.01,**kwargs):
    pi_= tf.constant(np.pi)
    rand_indices = tf.random_uniform([batchsize], minval=0, maxval=num_cluster, dtype=tf.int32)
    base_angle = pi_ * 2 / num_cluster
    angle = (base_angle * tf.cast(rand_indices,dtype=float32)) - (pi_ / 2)
    mean_0 = tf.expand_dims(scale*tf.cos(angle),1)
    mean_1 = tf.expand_dims(scale*tf.sin(angle),1)
    mean = tf.concat([mean_0, mean_1], 1)
    return ds.Normal(mean, (std**2)*tf.ones_like(mean))
