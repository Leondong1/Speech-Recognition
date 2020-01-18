from acoustics_model.am_cnn import AcousticModel
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import os
import sys
import time
project_path = os.path.abspath('.')
print(project_path)
sys.path.insert(0, project_path)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
set_session(tf.Session(config=config))

modelpath = 'model_log'


if(not os.path.exists(modelpath)):  # see if the model path exists
    # if not exists, then create one to avoid error when training
    os.makedirs(modelpath)


# datapath = '/Volumes/扩展/data/'
datapath = '/Users/leon/Documents/Code/thises/data/'
pny_vocab_file = './Pny_2gram_files/pny_dict/pny_vocab_dict.txt'
ms = AcousticModel(datapath, pny_vocab_file)


ms.load_model('./model_log/am.model')
ms.train_model(datapath, epoch = 50, batch_size = 16, save_step = 256)
nowtime = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
ms.save_model('./model_log/am_1' + nowtime + '.model')
