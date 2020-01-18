import os, wave, math, time
import numpy as np
import matplotlib.pyplot as plt
from util.audio_func import save_wave_file
from pyaudio import PyAudio, paInt16
from python_speech_features import mfcc, delta

from scipy.fftpack import fft


def read_wav_data(filename):
    '''
    read a wave file, get the wave data and sample rate
    :param filename:
    :return:
    '''
    wav = wave.open(filename, 'rb')  #
    num_frame = wav.getnframes()  # get frames count
    num_channel = wav.getnchannels()  # get channel count
    framerate = wav.getframerate()  # get sample rate
    num_sample_width = wav.getsampwidth()  # gen sample width, i.e. how many byte one sample has
    str_data = wav.readframes(num_frame)  # get the whole frame data
    wav.close()  # close the file
    wave_data = np.frombuffer(str_data, dtype=np.short)  # transform the bytes type data to ndarray
    wave_data.shape = -1, num_channel  # reshape the data, base on channel
    wave_data = wave_data.T  # transpose the data
    return wave_data, framerate


def get_mfcc_feature(wavsignal, fs):
    '''
    get mfcc feature
    :param wavsignal:
    :param fs:
    :return:
    '''
    feat_mfcc = mfcc(wavsignal[0], fs)
    feat_mfcc_d = delta(feat_mfcc, 2)
    feat_mfcc_dd = delta(feat_mfcc_d, 2)
    # return mfcc feature,  first order difference and second order difference
    wav_feature = np.column_stack((feat_mfcc, feat_mfcc_d, feat_mfcc_dd))
    return wav_feature


x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))  # Hamming window weights


def get_sound_spectrum_feature(wavsignal, fs):
    # wav data split windows with width 25ms and step 10ms
    if (16000 != fs):
        raise ValueError(
            '[Error] ASRT currently only supports wav audio files with a sampling rate of 16000 Hz, but this audio is ' + str(
                fs) + ' Hz. ')

    # wav data split windows with width 25ms and step 10ms
    time_window = 25  # ms
    window_length = fs // 1000 * time_window  # how many numbers within one slice(window)

    time_offset = 10  # ms
    window_offset = fs // 1000 * time_offset  # how many numbers between two slice(window), within
    wav_arr = np.array(wavsignal)
    wav_length = wav_arr.shape[1]

    range0_end = int(len(wavsignal[0]) / fs * 1000 - time_window) // time_offset  # how many slice(window) we get
    data_input = np.zeros((range0_end, int(window_length / 2)), dtype=np.float)  #

    for i in range(0, range0_end):
        p_start = i * window_offset
        p_end = p_start + window_length
        # print(p_start,p_end)

        data_line = np.array(wav_arr[0, p_start:p_end])

        data_line = data_line * w  # multiply with window weights

        data_line = np.abs(fft(data_line)) / wav_length

        data_input[i] = data_line[0:window_length // 2]  # get half data because it's symmetric

    # print(data_input.shape)
    data_input = np.log(data_input + 1)
    return data_input


def wav_show(wave_data, fs):  # draw wave file's wave
    time = np.arange(0, len(wave_data)) * (1.0 / fs)  # count time duration base on data length and sample rate
    # plot wave
    plt.figure(figsize=(30, 6))

    print(wave_data.shape)
    # rectangle window
    plt.plot(time[80000:80000 + len(w)], wave_data[80000:80000 + len(w)], )
    plt.grid()

    plt.figure(figsize=(30, 6))
    # Hamming window
    plt.plot(time[80000:80000 + len(w)], w * wave_data[80000:80000 + len(w)], )
    plt.grid()

    # see how a clip wave file looks
    pa = PyAudio()
    stream = pa.open(format=paInt16, channels=1,
                     rate=16000, input=True,
                     frames_per_buffer=2000)
    my_buf = wave_data[80000:80000 + len(w)]
    save_wave_file('01.wav', my_buf)
    stream.close()

    # plot fft wave
    tmp = fft(wave_data[80000:80000 + len(w)])

    plt.figure(figsize=(30, 6))
    length = len(tmp) // 2
    plt.plot(np.arange(length), np.abs(tmp[:length]))

    # plt.plot(time, wave_data[1], c = "g")
    plt.show()


from io import BytesIO

if __name__ == '__main__':
    # wave_data, fs = read_wav_data("/Volumes/扩展/data/data_thchs30/data/A4_45.wav")

    # wav_show(wave_data[0], fs)
    # t0 = time.time()
    # freimg3 = GetFrequencyFeature(wave_data, fs)
    # freimg = getMfccFeature(wave_data, fs)
    # t1 = time.time()
    # print('time cost:', t1 - t0)
    #
    # # print(w)
    # # print(x)
    #
    # # print(freimg)
    # # print(freimg.shape)
    # print(freimg3)
    # print(freimg3.shape)
    #
    # print(wave_data)
    # print(wave_data.shape)
    #
    # # 绘制频谱图
    # plt.imshow(freimg3.T, origin='lower')
    # plt.show()

    filename = "/Users/leon/Documents/Code/thises/data/data_thchs30/data/A4_45.wav"
    file_content = open(filename, 'rb').read()
    wav = wave.open(filename, 'rb')  # 打开一个wav格式的声音文件流
    num_frame = wav.getnframes()  # 获取帧数
    num_channel = wav.getnchannels()  # 获取声道数
    framerate = wav.getframerate()  # 获取抽样率
    num_sample_width = wav.getsampwidth()  # 读取实例的比特宽度， 即每一帧的字节数
    str_data = wav.readframes(num_frame)  # 读取全部的帧数据
    wav.close()  # 关闭流
    wave_data = np.frombuffer(str_data, dtype=np.short)  # 将声音文件数据转换成数组矩阵形式
    wave_data.shape = -1, num_channel  # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
    wave_data = wave_data.T  # 转置

    print(str_data)

    file_content = open(filename, 'rb').read()
    wav = wave.open(BytesIO(file_content), 'rb')  # 打开一个wav格式的声音文件流
    num_frame = wav.getnframes()  # 获取帧数
    num_channel = wav.getnchannels()  # 获取声道数
    framerate = wav.getframerate()  # 获取抽样率
    num_sample_width = wav.getsampwidth()  # 读取实例的比特宽度， 即每一帧的字节数
    str_data1 = wav.readframes(num_frame)  # 读取全部的帧数据
    wav.close()  # 关闭流
    # wave_data = np.frombuffer(str_data, dtype=np.short)  # 将声音文件数据转换成数组矩阵形式
    # wave_data.shape = -1, num_channel  # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
    # wave_data = wave_data.T  # 转置

    print(str_data1)

    print(str_data == str_data1)
