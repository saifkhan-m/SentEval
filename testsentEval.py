import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append(f'{os.getcwd()}/SentEval')

import tensorflow as tf

# Prevent TF from claiming all GPU memory so there is some left for pytorch.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Memory growth needs to be the same across GPUs.
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import tensorflow_hub as hub
import tensorflow_text
import senteval
import time

# https://huggingface.co/sentence-transformers/nli-bert-base
PATH_TO_DATA = f'data'
MODEL = 'https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1' #@param ['https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1', 'https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-large/1']
PARAMS = 'rapid prototyping' #@param ['slower, best performance', 'rapid prototyping']
TASK = 'CR' #@param ['CR','MR', 'MPQA', 'MRPC', 'SICKEntailment', 'SNLI', 'SST2', 'SUBJ', 'TREC']

params_prototyping = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_prototyping['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

params_best = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
params_best['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 16,
                                 'tenacity': 5, 'epoch_size': 6}

params = params_best if PARAMS == 'slower, best performance' else params_prototyping

preprocessor = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
encoder = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1")

inputs = tf.keras.Input(shape=tf.shape(''), dtype=tf.string)
outputs = encoder(preprocessor(inputs))

model = tf.keras.Model(inputs=inputs, outputs=outputs)

def prepare(params, samples):
    return

def batcher(_, batch):
    batch = [' '.join(sent) if sent else '.' for sent in batch]
    return model.predict(tf.constant(batch))["default"]


se = senteval.engine.SE(params, batcher, prepare)
print("Evaluating task %s with %s parameters" % (TASK, PARAMS))
start = time.time()
results = se.eval(TASK)
end = time.time()
print('Time took on task %s : %.1f. seconds' % (TASK, end - start))
print(results)
