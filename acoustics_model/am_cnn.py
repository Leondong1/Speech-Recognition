import os, sys, time, random
project_path = os.path.abspath('.')
print(project_path)
sys.path.insert(0,project_path)
sys.path.append(os.path.join(project_path))
from util.wav_feature import read_wav_data, get_mfcc_feature, get_sound_spectrum_feature
from util.readdata import get_pny_list, GetData
from util.distance import GetEditDistance


import keras
import numpy as np

import tensorflow as tf
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from keras.models import Model
from keras.layers import Dense, Dropout, Input, Reshape,  Activation,Conv2D, MaxPooling2D, BatchNormalization, Lambda, TimeDistributed
from keras import backend as K
from keras.optimizers import SGD, Adadelta, Adam


class AcousticModel(): # acoustic model, self explained
    def __init__(self, datapath, pny_vocab_file,is_train=True):
        '''
        initialization
        currently pinyin vocab size is 1421 plus blank
        '''
        self.pny_vocab_file = pny_vocab_file
        self.pny_vocab_list = get_pny_list(pny_vocab_file)
        # PNY_vocab_size = 1422
        PNY_vocab_size = len(self.pny_vocab_list)
        self.PNY_vocab_size = PNY_vocab_size # size of the output of every slice, which is the pinyin vocab size
        #self.BATCH_SIZE = BATCH_SIZE # batch size, self explained
        self.label_max_string_length = 64
        self.AUDIO_MAXLENGTH = 1600 # max length of input, because we use CNN, we have to pad or chop to this size
        self.WAV_FEATURE_LENGTH = 200
        self.is_train = is_train
        self.base_model = self.cnn_base_model()
        self.a_model = self.acoustic_model(self.base_model)
        self.datapath = datapath


    def cnn_base_model(self):
        '''
        CNN base model, define x to c , which is how audio wave feature to the probability of every slice
        every slice contains 200 feature，max lenth of one input is 1600(about 16s)
        hidden layer contains cnn and dropout
        output layer: dense layer, output size is self.PNY_vocab_size? or it should be choped into some fixed size, activation function is softmax
        :return:
        '''
        input_data = Input(name='the_input', shape=(self.AUDIO_MAXLENGTH, self.WAV_FEATURE_LENGTH, 1))

        layer_h1 = Conv2D(32, (3, 3), use_bias=False, activation='relu', padding='same',
                          kernel_initializer='he_normal')(input_data)
        if self.is_train:
            layer_h1 = Dropout(0.05)(layer_h1)
        layer_h2 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h1)
        layer_h3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2)

        if self.is_train:
            layer_h3 = Dropout(0.05)(layer_h3)
        layer_h4 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h3)
        if self.is_train:
            layer_h4 = Dropout(0.1)(layer_h4)
        layer_h5 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h4)
        layer_h6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h5)
        if self.is_train:
            layer_h6 = Dropout(0.1)(layer_h6)
        layer_h7 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                          kernel_initializer='he_normal')(layer_h6)
        if self.is_train:
            layer_h7 = Dropout(0.15)(layer_h7)
        layer_h8 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                          kernel_initializer='he_normal')(layer_h7)
        layer_h9 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h8)
        if self.is_train:
            layer_h9 = Dropout(0.15)(layer_h9)
        layer_h10 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h9)
        if self.is_train:
            layer_h10 = Dropout(0.2)(layer_h10)
        layer_h11 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h10)
        layer_h12 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h11)
        if self.is_train:
            layer_h12 = Dropout(0.2)(layer_h12)
        layer_h13 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h12)
        if self.is_train:
            layer_h13 = Dropout(0.2)(layer_h13)
        layer_h14 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h13)
        layer_h15 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h14)

        layer_h16 = Reshape((200, 3200))(layer_h15)

        if self.is_train:
            layer_h16 = Dropout(0.3)(layer_h16)
        layer_h17 = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h16)
        if self.is_train:
            layer_h17 = Dropout(0.3)(layer_h17)
        layer_h18 = Dense(self.PNY_vocab_size, use_bias=True, kernel_initializer='he_normal')(layer_h17)

        y_pred = Activation('softmax', name='Activation0')(layer_h18)
        base_model = Model(inputs=input_data, outputs=y_pred)

        return base_model


    def acoustic_model(self, base_model):
        labels = Input(name='the_labels', shape=[self.label_max_string_length], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer

        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')(
            [base_model.output, labels, input_length, label_length])

        model = Model(inputs=[base_model.input, labels, input_length, label_length], outputs=loss_out)

        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, epsilon=10e-8)
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt)

        print('[*Info] Create Model Successful, Compiles Model Successful. ')
        return model



    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args

        y_pred = y_pred
        # y_pred = y_pred[:, :, :]
        #y_pred = y_pred[:, 2:, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)



    def train_model(self, datapath, epoch = 2, save_step = 1000, batch_size = 32):
        '''
        train model
        :parameter
            datapath: path of data
            epoch: how many rounds you want to train
            save_step:
            filename: model name, no need suffix
        '''
        data=GetData(path=datapath, subset='train', pny_vocab_file=self.pny_vocab_file)

        yielddatas = data.data_genetator(batch_size, self.AUDIO_MAXLENGTH)

        for epoch in range(epoch): #
            print('[running] train epoch %d .' % epoch)
            n_step = 0 #
            while True:
                try:
                    print('[message] epoch %d . Have train datas %d+'%(epoch, n_step*save_step))

                    self.a_model.fit_generator(yielddatas, save_step)
                    n_step += 1
                except StopIteration:
                    print('[error] generator error. please check data format.')
                    break


                self.test_model(subset='train', data_count = 4)
                self.test_model(subset='dev', data_count = 4)

    def load_model(self,filename = 'model_log/'+'am'+'.model'):
        '''
        load model
        '''
        self.a_model.load_weights(filename)
        self.base_model.load_weights(filename + '.base')

    def save_model(self,filename):
        '''
        save model
        '''
        self.a_model.save_weights(filename)
        self.base_model.save_weights(filename + '.base')


    def test_model(self, subset='dev', data_count = 32, out_report = False, show_ratio = True, io_step_print = 10, io_step_file = 10):
        '''
        test model

        io_step_print
            reduce io expense of printing by modify this parameter

        io_step_file
            reduce io expense of writing by modify this parameter

        '''
        data=GetData(self.datapath, subset=subset, pny_vocab_file=self.pny_vocab_file)
        num_data = data.data_count() # get data number
        if(data_count <= 0 or data_count > num_data):
            # use all data to test if the parameter is not valid
            data_count = num_data

        try:
            ran_num = random.randint(0,num_data - 1) # get a random number

            words_num = 0
            word_error_num = 0

            nowtime = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
            if(out_report == True):
                txt_obj = open('Test_Report_' + subset + '_' + nowtime + '.txt', 'w', encoding='UTF-8') # open file

            txt = 'test_report\n' + '\n\n'
            for i in range(data_count):
                data_input, data_labels = data.get_data((ran_num + i) % num_data)  # get some data, start at random number pos

                # process wave file which is too long
                # jump over if on wave file is longer than max input length(1600)
                num_bias = 0
                while(data_input.shape[0] > self.AUDIO_MAXLENGTH):
                    print('*[Error]','wave data lenghth of num',(ran_num + i) % num_data, 'is too long.','\n A Exception raise when test Speech Model.')
                    num_bias += 1
                    data_input, data_labels = data.get_data((ran_num + i + num_bias) % num_data)  # get some data, start at random number pos
                # end process

                pre = self.predict(data_input, data_input.shape[0] // 8)

                words_n = data_labels.shape[0] # get length of wave label
                words_num += words_n # sum over all label's length
                edit_distance = GetEditDistance(data_labels, pre) # get editdistance
                if(edit_distance <= words_n): # if editdistance is less than label's length
                    word_error_num += edit_distance # use editdistance as the error number
                else: # or it must have added a bunch load of weird character
                    word_error_num += words_n # use the label's length as error number

                if((i % io_step_print == 0 or i == data_count - 1) and show_ratio == True):

                    print('Test Count: ',i,'/',data_count)


                if(out_report == True):
                    if(i % io_step_file == 0 or i == data_count - 1):
                        txt_obj.write(txt)
                        txt = ''

                    txt += str(i) + '\n'
                    txt += 'True:\t' + str(data_labels) + '\n'
                    txt += 'Pred:\t' + str(pre) + '\n'
                    txt += '\n'



            print('*[Test Result] Speech Recognition ' + subset + ' set word error ratio: ', word_error_num / words_num * 100, '%')
            if(out_report == True):
                txt += '*[test result] ASR ' + subset + ' word error ratio： ' + str(word_error_num / words_num * 100) + ' %'
                txt_obj.write(txt)
                txt = ''
                txt_obj.close()

        except StopIteration:
            print('[Error] Model Test Error. please check data format.')

    def predict(self, data_input, input_len):
        '''
        predict
        :return predict result,which will be a list of pinyin words
        '''

        batch_size = 1
        in_len = np.zeros((batch_size),dtype = np.int32)

        in_len[0] = input_len

        x_in = np.zeros((batch_size, 1600, self.WAV_FEATURE_LENGTH, 1), dtype=np.float)

        for i in range(batch_size):
            x_in[i,0:len(data_input)] = data_input


        base_pred = self.base_model.predict(x = x_in)



        base_pred =base_pred[:, :, :]
        #base_pred =base_pred[:, 2:, :]

        r = K.ctc_decode(base_pred, in_len, greedy = True, beam_width=100, top_paths=1)




        r1 = K.get_value(r[0][0])
        #print('r1', r1)


        #r2 = K.get_value(r[1])
        #print(r2)

        r1=r1[0]

        return r1


    def recognize_sound_wave(self, wavsignal, fs):
        '''
        predict from an audio wave data

        '''


        # get wave feature
        #data_input = GetMfccFeature(wavsignal, fs)
        #t0=time.time()
        data_input = get_sound_spectrum_feature(wavsignal, fs)
        #t1=time.time()
        #print('time cost:',t1-t0)

        input_length = len(data_input)
        input_length = input_length // 8

        data_input = np.array(data_input, dtype = np.float)
        #print(data_input,data_input.shape)
        data_input = data_input.reshape(data_input.shape[0],data_input.shape[1],1)
        #t2=time.time()
        r1 = self.predict(data_input, input_length)
        #t3=time.time()
        #print('time cost:',t3-t2)
        print(r1)
        r_str=[]
        for i in r1:
            r_str.append(self.pny_vocab_list[i])

        return r_str
        pass

    def recognize_sound_wave_from_file(self, filename):
        '''
        predict from local file
        '''

        wavsignal,fs = read_wav_data(filename)

        r = self.recognize_sound_wave(wavsignal, fs)

        return r

        pass



    @property
    def model(self):
        '''
        :return keras model
        '''
        return self.a_model


if __name__=='__main__':
    modelpath =  'model_log'


    if(not os.path.exists(modelpath)): # see if the model path exists
        os.makedirs(modelpath) # if not exists, then create one to avoid error when training


    datapath = '/Users/leon/Documents/Code/thises/data/'

    pny_vocab_file = './../Pny_2gram_files/pny_dict/pny_vocab_dict.txt'
    ms = AcousticModel(datapath, pny_vocab_file)



    ms.load_model('./../model_log/am.model')
    ms.train_model(datapath, epoch = 50, batch_size = 16, save_step = 256)
    # ms.save_model('../model_log/am.model')

    # t1=time.time()
    ms.test_model(subset='train', data_count = 128, out_report = True)
    ms.test_model(subset='dev', data_count = 128, out_report = True)
    ms.test_model(subset='test', data_count = 128, out_report = True)
    # t2=time.time()
    # print('Test Model Time Cost:',t2-t1,'s')
    r = ms.recognize_sound_wave_from_file('/Volumes/扩展/data/data_aishell/wav/test/S0764/BAC009S0764W0121.wav')
    print('*[Info] ASR results：\n',r)

    # ms.base_model.summary()
    # ms.a_model.summary()



    # tf.summary.FileWriter('./logdir/', tf.get_default_graph())