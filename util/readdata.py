import os
import random
import numpy as np
from util.wav_feature import read_wav_data, get_mfcc_feature, get_sound_spectrum_feature


def get_wav_name(filename):
    '''
    read a wave info file, returns a dict,
    key would be the wave file name, value would be its relative path
    also returns a list, contains every wave file's name
    '''
    txt_obj = open(filename, 'r')
    wave_file_dict = {}  # wave file dict
    wave_file_list = []  # wave file list
    for i in txt_obj:
        if i != '':
            txt_l = i.split()
            wave_file_dict[txt_l[0]] = txt_l[1]
            wave_file_list.append(txt_l[0])
    txt_obj.close()
    return wave_file_dict, wave_file_list


def get_wav_pny(filename):
    '''
    read a wave info file, returns a list and a dict
    list contains every va file's name
    dict's key would be wave file name, value would be its label pinyins
    '''
    txt_obj = open(filename, 'r')
    wave_pny_dict = {}  #
    wave_file_list = []  #
    for i in txt_obj:
        if i != '':
            txt_l = i.split()
            wave_pny_dict[txt_l[0]] = txt_l[1:]
            wave_file_list.append(txt_l[0])
    txt_obj.close()
    return wave_pny_dict, wave_file_list


def get_pny_list(filename):
    '''
    load pinyin vocab as a list
    :return list of pinyin vocab plus 'blank' mark as '_'
    '''
    list_pny = []  #
    for i in open(filename, 'r', encoding='UTF-8'):
        if (i != ''):
            txt_l = i.split('\t')
            list_pny.append(txt_l[0])

    list_pny.append('_')
    return list_pny


class GetData():

    def __init__(self, path, pny_vocab_file, subset='train'):
        '''

        :parameter
            path: data path
            subset: train, dev, test, option of data subset
        '''

        self.datapath = path  # data path
        self.subset = subset

        self.wave_file_list = []
        self.wave_file_path_dict = {}
        self.pny_seq_list = []
        self.pny_seq_dict = {}

        self.pny_vocab_list = get_pny_list(pny_vocab_file)  # pinyin vocab list
        self.pny_vocab_size = len(self.pny_vocab_list)  # pinyin vocab size

        self.number_of_data = 0  # number of data
        self.load_data_list(subset=subset)  # load data list

        pass

    def load_data_list(self, subset='train'):
        '''
        load data list
        :parameter
            subset: str, data type
                train
                dev
                test
        '''

        if subset == 'train':
            wave_file_list = [
                os.path.join('thchs30', 'train.wav.txt'),
                # os.path.join('st-cmds', 'train.wav.txt'),
            ]
            pny_file_list = [
                os.path.join('thchs30', 'train.pny.txt'),
                # os.path.join('st-cmds', 'train.pny.txt'),
            ]
        elif subset == 'dev':
            wave_file_list = [
                os.path.join('thchs30', 'cv.wav.txt'),
                # os.path.join('st-cmds', 'dev.wav.txt'),
            ]
            pny_file_list = [
                os.path.join('thchs30', 'cv.pny.txt'),
                # os.path.join('st-cmds', 'dev.pny.txt'),
            ]
        elif subset == 'test':
            wave_file_list = [
                os.path.join('thchs30', 'test.wav.txt'),
                # os.path.join('st-cmds', 'test.wav.txt'),
            ]
            pny_file_list = [
                os.path.join('thchs30', 'test.pny.txt'),
                # os.path.join('st-cmds', 'test.pny.txt'),
            ]
        else:
            wave_file_list = []
            pny_file_list = []

        # get wave file path and corresponding pinyin sequence
        for wave_file in wave_file_list:
            wave_file_dict, wave_file_list = get_wav_name(
                os.path.join(self.datapath, wave_file))
            self.wave_file_list.extend(wave_file_list)
            for key in wave_file_dict.keys():
                self.wave_file_path_dict[key] = wave_file_dict[key]
        for pny_file in pny_file_list:
            wave_pny_dict, wave_pny_list = get_wav_pny(
                os.path.join(self.datapath, pny_file))
            self.pny_seq_list.extend(wave_pny_list)
            for key in wave_pny_dict.keys():
                self.pny_seq_dict[key] = wave_pny_dict[key]

        # print(self.wave_file_list)
        # print(self.wave_file_path_dict)
        # print(self.pny_seq_list)
        # print(self.pny_seq_dict)

        # self.number_of_data = len(self.wave_file_list)
        self.number_of_data = self.data_count()

    def data_count(self):
        '''
        count data
        :return number of samples, -1 when number of data path list not equal with number of pinyin list
        '''

        if len(self.pny_seq_list) == len(self.wave_file_list):
            number_of_data = len(self.pny_seq_list)
        else:
            number_of_data = -1

        return number_of_data

    def get_data(self, n_start, n_amount=1):
        '''
        read data, inputs and labels fit for training
        :parameter
            n_start
            n_amount
        :return
            inputs and labels fit for training
        '''

        filename = self.wave_file_path_dict[self.wave_file_list[(
            n_start - 1) % self.number_of_data]]
        pny_seq = self.pny_seq_dict[self.pny_seq_list[(
            n_start - 1) % self.number_of_data]]

        wavsignal, fs = read_wav_data(self.datapath + filename)

        # get feature

        feat_out = []
        # print("data no.",n_start,filename)
        for i in pny_seq:
            if ('' != i):
                n = self.pny2sparse(i)
                feat_out.append(n)
        # print('feat_out:',feat_out)

        # get feature
        data_input = get_sound_spectrum_feature(wavsignal, fs)
        # data_input = np.array(data_input)
        data_input = data_input.reshape(
            data_input.shape[0], data_input.shape[1], 1)
        # arr_zero = np.zeros((1, 39), dtype=np.int16) #

        # while(len(data_input)<1600): # padding to 1600
        #    data_input = np.row_stack((data_input,arr_zero))

        # data_input = data_input.T
        data_label = np.array(feat_out)
        return data_input, data_label

    def data_genetator(self, batch_size=32, audio_length=1600):
        '''
        data genetator,for Keras's generator_fit
        batch_size:

        '''

        labels = np.zeros((batch_size,), dtype=np.float)

        while True:
            X = np.zeros((batch_size, audio_length, 200, 1), dtype=np.float)
            y = np.zeros((batch_size, 64), dtype=np.int16)

            input_length = []
            label_length = []

            for i in range(batch_size):
                ran_num = random.randint(
                    0, self.number_of_data - 1)  # get a random number
                # gen a data base on the random number
                data_input, data_labels = self.get_data(ran_num)

                input_length.append(
                    data_input.shape[0] //
                    8 +
                    data_input.shape[0] %
                    8)

                X[i, 0:len(data_input)] = data_input
                y[i, 0:len(data_labels)] = data_labels
                label_length.append([len(data_labels)])

            label_length = np.matrix(label_length)
            input_length = np.array(input_length).T
            yield [X, y, input_length, label_length], labels
        pass

    def get_vocab_size(self):
        '''
        get pinyin vocab size
        '''
        return len(self.pny_vocab_list)

    def pny2sparse(self, pny):
        '''
        pinyin to sparse
        '''
        if (pny != ''):
            return self.pny_vocab_list.index(pny)
        return self.pny_vocab_size

    def sparse2one_hot(self, num):
        '''
        sparse to one-hot
        '''
        v_tmp = []
        for i in range(0, len(self.pny_vocab_list)):
            if (i == num):
                v_tmp.append(1)
            else:
                v_tmp.append(0)
        v = np.array(v_tmp)
        return v


if __name__ == '__main__':
    path = '/Users/leon/Documents/Code/thises/data'
    l = GetData(
        path=path,
        subset='train',
        pny_vocab_file='./../Pny_2gram_files/pny_dict/pny_vocab_dict.txt')
    l.load_data_list()

    # print(l.data_count())
    # print(l.get_data(0))
    # aa=l.data_genetator()
    # for i in aa:
    #     a,b=i
    # print(a,b)

    print(l.get_vocab_size())
    print(l.pny_vocab_list)
    pass
