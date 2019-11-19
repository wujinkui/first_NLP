import tensorflow as tf
def poem_gen_model(input,output_data,vocab_size,hidden_size =300,
                   model = 'lstm',num_layers = 2, batch_size = 64, learning_rate = 0.001):

    end_points = {}
    cell_fun = tf.keras.layers.LSTMCell
    cell = cell_fun(hidden_size)
    cell = tf.keras.layers.StackedRNNCells([cell] * num_layers )
    initial_state = cell.zero.stat(batch_size,tf.float32)
  # embedding = tf.Variable(tf.random_normal_initializer((vocab_size +1,hidden_size))
    embedding = tf.keras.layers.Embedding(vocab_size,hidden_size)
    inputs  = tf.nn.embedding_lookup(embedding,input)
    outputs, last_state = tf.keras.layers.RNN(cell,inputs,initial_state)
    output = tf.reshape(outputs,[-1,hidden_size])

    weights = tf.Variable(tf.compat.v1.truncated_normal([hidden_size,vocab_size+1]))
    bias = tf.Variable(tf.zeros(shape = [vocab_size+1]))
    logits = tf.nn.bias_add(tf.matmul(output,weights)+bias)
    labels = tf.one_hot(tf.reshape(output_data,[-1]),depth = vocab_size+1)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels,logits)
    total_loss = tf.reduce_mean(loss)

    train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(total_loss)

    end_points['total_loss'] = total_loss
    return end_points



