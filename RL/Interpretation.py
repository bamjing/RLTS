import gensim
from collections import deque
import tensorflow as tf

class Interpreter(object):
    def __init__(self, sess, para_dict):

        self.EMBEDDING_MODEL_PATH = para_dict['EMBEDDING_MODEL_PATH']
        self.EMBEDDING_MODEL_NAME = para_dict['EMBEDDING_MODEL_NAME']

        self.CLASSIFICATION_MODEL_PATH = para_dict['CLASSIFICATION_MODEL_PATH']
        self.CLASSIFIER_MODEL_META_NAME = para_dict['CLASSIFIER_MODEL_META_NAME']

        self.sess = sess
        self.embeddings = gensim.models.Word2Vec.load(self.EMBEDDING_MODEL_PATH + self.EMBEDDING_MODEL_NAME)
        saver = tf.train.import_meta_graph(self.CLASSIFICATION_MODEL_PATH + self.CLASSIFIER_MODEL_META_NAME)
        saver.restore(self.sess, tf.train.latest_checkpoint(self.CLASSIFICATION_MODEL_PATH))
        graph = tf.get_default_graph()
        self.input = graph.get_tensor_by_name("CLASSIFICATION_INPUT:0")
        self.output = graph.get_tensor_by_name("CLASSIFICATION_PRED:0")

    def research(self, state):
        output = deque()
        index_list = self._get_index(state)

        for item in index_list:
            output.append(self._get_vercor(str(item[0])))

        state = list(output)
        return state

    def _get_index(self, data):
        return self.sess.run(self.output, {self.input: data})

    def _get_vercor(self, index_list):
        return self.embeddings.wv[index_list]