import wave
from pyaudio import PyAudio, paInt16

framerate = 16000  # 抽样率
NUM_SAMPLES = 2000  # 底层的缓存的块的大小，底层的缓存由N个同样大小的块组成
channels = 1    # 通道数
sampwidth = 2   # 量化字节数
TIME = 10


def save_wave_file(filename, data):
    '''save the date to the wavfile'''
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)
    wf.writeframes(b"".join(data))
    wf.close()


def audio_record():
    pa = PyAudio()
    stream = pa.open(format=paInt16, channels=1,
                     rate=framerate, input=True,
                     frames_per_buffer=NUM_SAMPLES)
    my_buf = []
    count = 0
    while count < TIME * 10:  # 控制录音时间
        string_audio_data = stream.read(NUM_SAMPLES)
        my_buf.append(string_audio_data)
        count += 1
        print('.')
    save_wave_file('010.wav', my_buf)
    stream.close()


chunk = 2000  # 一次读入多少帧


def play():
    wf = wave.open(r"010.wav", 'rb')
    p = PyAudio()
    stream = p.open(
        format=p.get_format_from_width(
            wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True)
    while True:
        data = wf.readframes(chunk)
        if data == b"":
            break
        # print(data)
        # print(type(data))
        stream.write(data)
    stream.close()
    p.terminate()


if __name__ == '__main__':
    audio_record()
    print('Over!')
    play()
