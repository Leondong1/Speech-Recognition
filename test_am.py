from LM.HMM_LM1 import LanguageModel
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
ms.test_model(subset='train', data_count = 128, out_report = True)
# ms.test_model(subset='dev', data_count = 128, out_report = True)
# ms.test_model(subset='test', data_count = 128, out_report = True)

r = ms.recognize_sound_wave_from_file(
    '/Users/leon/Documents/Code/thises/data/data_thchs30/data/A2_0.wav' )
print('*[Info] ASR results：\n', r)


pny_vocab_dict_file = './Pny_2gram_files/pny_dict/pny_vocab_dict.txt'
one_gram_file = './Pny_2gram_files/1gram_count.txt'
two_gram_file = './Pny_2gram_files/2gram_count.txt'
two_gram_pny_file = './Pny_2gram_files/2gram_pinyin_count.txt'
ml = LanguageModel(
    pny_vocab_dict_file,
    one_gram_file,
    two_gram_file,
    two_gram_pny_file)

#str_pinyin = ['zhe4','zhen1','shi4','ji2', 'hao3','de5']
#str_pinyin = ['jin1', 'tian1', 'shi4', 'xing1', 'qi1', 'san1']
#str_pinyin = ['ni3', 'hao3','a1']
str_pinyin = r
#str_pinyin =  ['su1', 'bei3', 'jun1', 'de5', 'yi4','xie1', 'ai4', 'guo2', 'jiang4', 'shi4', 'ma3', 'zhan4', 'shan1', 'ming2', 'yi1', 'dong4', 'ta1', 'ju4', 'su1', 'bi3', 'ai4', 'dan4', 'tian2','mei2', 'bai3', 'ye3', 'fei1', 'qi3', 'kan4', 'zhan4']

t1 = time.time()
r = ml.pinyin2Text(str_pinyin)
# r = ml.decode(str_pinyin)
# print('语音转文字结果：\n', r)
# print(time.time() - t1)
