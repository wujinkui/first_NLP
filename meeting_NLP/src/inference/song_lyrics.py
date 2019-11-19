
"""
we will implement a song writer AI who be able to generate lyrics.
"""
import collections
import os
import sys
import numpy as np
import tensorflow as tf
from models.model import rnn_model
from dataset.lyrics import process_lyrics, generate_batch
tf.compat.v1.app.flags.DEFINE_string('w','lyric','lyric')
tf.compat.v1.app.flags.DEFINE_bool('no-train',False,'write')
tf.compat.v1.app.flags.DEFINE_bool('train',True,'train')
tf.compat.v1.app.flags.DEFINE_integer('batch_size', 20, 'batch size.')
tf.compat.v1.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate.')

tf.compat.v1.app.flags.DEFINE_string('file_path', os.path.abspath('./dataset/data/周杰伦歌词大全.txt'), 'file path of lyrics.')
tf.compat.v1.app.flags.DEFINE_string('checkpoints_dir', os.path.abspath('./checkpoints/lyrics'), 'checkpoints save path.')
tf.compat.v1.app.flags.DEFINE_string('model_prefix', 'lyrics', 'model save prefix.')

tf.compat.v1.app.flags.DEFINE_integer('epochs', 500, 'train how many epochs.')

FLAGS = tf.compat.v1.app.flags.FLAGS

start_token = 'G'
end_token = 'E'


def run_training():
    if not os.path.exists(os.path.dirname(FLAGS.checkpoints_dir)):
        os.mkdir(os.path.dirname(FLAGS.checkpoints_dir))
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.mkdir(FLAGS.checkpoints_dir)

    poems_vector, word_to_int, vocabularies = process_lyrics(FLAGS.file_path)
    batches_inputs, batches_outputs = generate_batch(FLAGS.batch_size, poems_vector, word_to_int)
    tf.compat.v1.disable_eager_execution()
    input_data = tf.compat.v1.placeholder(tf.int32, [FLAGS.batch_size, None])
    output_targets = tf.compat.v1.placeholder(tf.int32, [FLAGS.batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=output_targets, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=FLAGS.batch_size, learning_rate=FLAGS.learning_rate)

    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
    init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
    with tf.compat.v1.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.run(init_op)

        start_epoch = 0
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoints_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("[INFO] restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        print('[INFO] start training...')
        try:
            for epoch in range(start_epoch, FLAGS.epochs):
                n = 0
                n_chunk = len(poems_vector) // FLAGS.batch_size
                for batch in range(n_chunk):
                    loss, _, _ = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op']
                    ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})
                    n += 1
                    print('[INFO] Epoch: %d , batch: %d , training loss: %.6f' % (epoch, batch, loss))
                if epoch % 20 == 0:
                    saver.save(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.model_prefix), global_step=epoch)
        except KeyboardInterrupt:
            print('[INFO] Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.model_prefix), global_step=epoch)
            print('[INFO] Last epoch were saved, next time will start from epoch {}.'.format(epoch))


def to_word(predict, vocabs):
    t = np.cumsum(predict)
    s = np.sum(predict)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    if sample > len(vocabs)-1:
        sample = len(vocabs) - 100
    return vocabs[sample]


def gen_lyric():
    batch_size = 1
    poems_vector, word_int_map, vocabularies = process_lyrics(FLAGS.file_path)
    tf.compat.v1.disable_eager_execution()
    input_data = tf.compat.v1.placeholder(tf.int32, [batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=FLAGS.learning_rate)

    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
    init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)

        checkpoint = tf.compat.v1.train.latest_checkpoint(FLAGS.checkpoints_dir)
        saver.restore(sess, checkpoint)

        x = np.array([list(map(word_int_map.get, start_token))])

        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                         feed_dict={input_data: x})

        word = to_word(predict, vocabularies)
        print(word)
        lyric = ''
        while word != end_token:
            lyric += word
            x = np.zeros((1, 1))
            x[0, 0] = word_int_map[word]
            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                             feed_dict={input_data: x, end_points['initial_state']: last_state})
            word = to_word(predict, vocabularies)
        # word = words[np.argmax(probs_)]
        return lyric


def main(is_train):
    if is_train:
        print('[INFO] train song lyric...')
        run_training()
    else:
        print('[INFO] compose song lyric...')
        lyric = gen_lyric()
        lyric_sentences = lyric.split(' ')
        for l in lyric_sentences:
            print(l)
            # if 4 < len(l) < 20:
            #     print(l)

if __name__ == '__main__':
    tf.app.run()