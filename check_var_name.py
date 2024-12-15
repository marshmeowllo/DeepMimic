import tensorflow as tf

checkpoint_path = "output/agent0_model.ckpt"
checkpoint = tf.train.load_checkpoint(checkpoint_path)
variables = checkpoint.get_variable_to_shape_map()

for var_name in variables.keys():
    print(var_name)
