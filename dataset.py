from torch.utils import data
import torch
import librosa
import numpy as np

class TranscriptionBaseDataset(data.Dataset):
    def __init__(self, wav_path, sr=22050, n_fft=2048, hop_length=512, win_length=None, n_mels=128, fmax=8000, onset_time_list=None, onset_src_list=None):
        super().__init__()
        self.wav_path = wav_path
        self.sr = sr
        self.y, _ = librosa.load(path=self.wav_path, sr=self.sr)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmax = fmax
        self.melspectrogram = librosa.feature.melspectrogram(y=self.y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, \
                                                             n_mels=self.n_mels, fmax=self.fmax)
        self.onset_time_list = onset_time_list
        self.onset_frame_list = librosa.time_to_frames(self.onset_time_list, sr=self.sr, hop_length=self.hop_length, n_fft=self.n_fft)
        self.onset_src_list = onset_src_list

    def __getitem__(self, index):
        return self.melspectrogram[:, index]

    def __len__(self):
        raise NotImplementedError

class MDB_Dataset(data.Dataset):
    def __init__(self, musicname, sr=22050, n_fft=2048, hop_length=512, win_length=None, n_mels=128, fmax=8000, input_frame_size=3):
        super().__init__()
        self.music_name = musicname
        self.annotations = open(file=f"./data/MDB_drum/annotations/class/{self.music_name}_class.txt", mode='r').readlines()
        self.ground_truth_time_list = np.array([float(line.split('\t')[0].strip()) for line in self.annotations])
        self.ground_truth_src_list = [line.split('\t')[1].split('\n')[0].strip() for line in self.annotations]
        self.y, self.sr = librosa.load(path=f"./data/MDB_drum/audio/drum_only/{self.music_name}_Drum.wav", sr=sr)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmax = fmax
        self.input_frame_size = input_frame_size
        self.ground_truth_frame_list = librosa.time_to_frames(self.ground_truth_time_list, sr=self.sr, hop_length=self.hop_length, n_fft=self.n_fft)
        self.melspectrogram = librosa.feature.melspectrogram(y=self.y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, \
                                                             n_mels=self.n_mels, fmax=self.fmax)
        self.melspectrogram_frame_num = np.shape(self.melspectrogram)[1]
        assert len(self.ground_truth_frame_list) == len(self.ground_truth_src_list) == len(self.ground_truth_time_list)
        negative_idx = np.where(self.ground_truth_frame_list < 0)
        self.ground_truth_src_list, self.ground_truth_frame_list, self.ground_truth_time_list = \
            np.delete(self.ground_truth_src_list, negative_idx), np.delete(self.ground_truth_frame_list, negative_idx), np.delete(self.ground_truth_time_list, negative_idx)
        
    def __getitem__(self, index):
        src_instrument = self.ground_truth_src_list[index]
        src_time = self.ground_truth_time_list[index]
        src_melspectrogram = self.melspectrogram[:, self.ground_truth_frame_list[index]:self.ground_truth_frame_list[index]+self.input_frame_size]
        return src_instrument, src_time, src_melspectrogram


if __name__ == '__main__':
    sampledata = MDB_Dataset('MusicDelta_80sRock', input_frame_size=3)
