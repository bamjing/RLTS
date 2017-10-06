import gensim, logging, os
from collections import deque
from Module import Config
import pandas as pd

class Candle2Vec(object):
    def __init__(self, para_dict):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        LABELED_PATH = para_dict['LABELED_PATH']
        self.LABELED_NAME = para_dict['LABELED_NAME']

        self.EMBEDDING_FEATURE_SIZE = para_dict['EMBEDDING_FEATURE_SIZE']
        self.WINDOW = para_dict['WINDOW']
        self.WORKERS = para_dict['WORKERS']
        self.BATCH_WORDS = para_dict['BATCH_WORDS']
        self.MIN_COUNT = para_dict['MIN_COUNT']
        self.EPOCHS = para_dict['EPOCHS']
        self.SENTENCE_LENGTH = para_dict['SENTENCE_LENGTH']

        self.EMBEDDING_MODEL_PATH = para_dict['EMBEDDING_MODEL_PATH']
        csv_path = LABELED_PATH + self.LABELED_NAME
        df = pd.read_csv(csv_path, parse_dates=True)
        self.label_list = df['LABEL'].as_matrix()

    def train(self):
        sentences = self._prepare_sentence()
        model = gensim.models.Word2Vec(
            sentences=sentences,
            size=self.EMBEDDING_FEATURE_SIZE,
            window=self.WINDOW,
            min_count=self.MIN_COUNT,
            workers=self.WORKERS,
            batch_words=self.BATCH_WORDS)
        model.train(sentences, total_examples=len(sentences), epochs=self.EPOCHS)

        self._save_model(model)

    def _save_model(self, model):
        print "embedding finished!"
        duration = 1.5  # second
        freq = 440  # Hz
        os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))

        model_name = self.LABELED_NAME + '-EMBEDDING-' + str(self.WINDOW) + '-' + str(self.EPOCHS)
        path = self.EMBEDDING_MODEL_PATH + model_name
        model.save(path)

    def _prepare_sentence(self):
        string_sentence = []
        tmp = deque()
        for item in self.label_list:
            if len(tmp) < self.SENTENCE_LENGTH:
                tmp.append(str(item))
            else:
                string_sentence.append(list(tmp))
                tmp.popleft()
                tmp.append(str(item))
        return string_sentence

if __name__ == "__main__":
    config = Config.Configuration()
    para_dict = config.parameter_dict

    candle2Vec = Candle2Vec(para_dict)
    candle2Vec.train()