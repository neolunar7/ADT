from torch.utils import data
import torch
import librosa
import numpy as np
import glob, os, random

"""
    TODO
    Permutations on onset. Onset lable 인접 frame 들에 대한 labeling 방식까지 Dataset 인자로 받을 수 있게 수정 필요.
"""

class MDB_Dataset(data.Dataset):
    def __init__(self, spectral_feature='melspectrogram', sr=22050, n_fft=2048, hop_length=512, win_length=None, n_mels=128, fmax=8000, input_frame_size=10):
        super().__init__()
        MDB_music_names_with_ext = glob.glob(os.path.join("./data/MDB_drum/audio/drum_only", "*.wav"))
        self.music_names = [os.path.basename(music_with_ext).replace("_Drum.wav", "") for music_with_ext in MDB_music_names_with_ext]
        self.music_name_and_frame_len_pair_list = []
        self.music_name_and_frame_and_src_pair_list = []

        self.spectral_feature = spectral_feature
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmax = fmax
        self.input_frame_size = input_frame_size

        self._init_music_name_and_frame_pair_list() # Initialized into [music_name, frame_num] pairs.

    def __getitem__(self, index):
        current_music_name, current_frame_idx = self.music_name_and_frame_len_pair_list[index][0], self.music_name_and_frame_len_pair_list[index][1]
        current_music_frame_len = self._frame_num_calculation_for_music_name(current_music_name, self.spectral_feature)
        start_idx, end_idx, padding_count = self._get_music_start_end_index(current_music_frame_len, current_frame_idx)

        model_input_feature, model_input_label = self._get_model_input(current_music_name, start_idx, end_idx, padding_count)
        return tuple((model_input_feature, model_input_label))

    def __len__(self):
        return len(self.music_name_and_frame_len_pair_list)

    def _init_music_name_and_frame_pair_list(self):
        for music_name in self.music_names:
            frame_num = self._frame_num_calculation_for_music_name(music_name=music_name, spectral_feature=self.spectral_feature)
            self.music_name_and_frame_len_pair_list += [[music_name, frame_idx] for frame_idx in range(frame_num)]

    def _frame_num_calculation_for_music_name(self, music_name, spectral_feature='melspectrogram'):
        y = np.load(f'./data/MDB_drum/raw_audio_npy/{music_name}.npy')
        if spectral_feature == 'melspectrogram':
            feature = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, \
                                                    n_mels=self.n_mels, fmax=self.fmax)
        elif spectral_feature == 'mfcc':
            feature = librosa.feature.mfcc(y=y, sr=self.sr)
        frame_num = np.shape(feature)[1]
        return frame_num

    def _get_ground_truth_src_per_frame(self, music_name, frame_num):
        # This part naturally removes the negative frame index, so no need to consider it anymore.
        # TODO - For a frame that two instruments occur simultaneously, frame_idx may be 2, and the later one will not be learned.
        annotations = open(file=f"./data/MDB_drum/annotations/class/{music_name}_class.txt", mode='r').readlines()
        ground_truth_time_list = np.array([float(line.split('\t')[0].strip()) for line in annotations])
        ground_truth_src_list = [line.split('\t')[1].split('\n')[0].strip() for line in annotations]
        ground_truth_frame_list = librosa.time_to_frames(ground_truth_time_list, sr=self.sr, hop_length=self.hop_length, n_fft=self.n_fft) 
        if frame_num in ground_truth_frame_list:
            frame_idx = np.where(ground_truth_frame_list == frame_num)[0]
            if len(frame_idx) > 1:
                ground_truth_src = ground_truth_src_list[random.choice(frame_idx)] # TODO - Need to change this part, because both case should be returned, not randomly.
            else:
                ground_truth_src = ground_truth_src_list[frame_idx.item()]
        else:
            ground_truth_src = 'NotOnset'
        return ground_truth_src

    def _get_music_start_end_index(self, current_music_frame_len, current_frame_idx):
        end_idx = current_frame_idx + 1
        if current_frame_idx >= self.input_frame_size - 1:
            start_idx = current_frame_idx - self.input_frame_size + 1
            padding_count = 0
        else:
            start_idx = 0
            padding_count = self.input_frame_size - current_frame_idx - 1
        return start_idx, end_idx, padding_count

    def _get_model_input(self, music_name, start_idx, end_idx, padding_count):
        raw_audio_npy_file = np.load(f'./data/MDB_drum/raw_audio_npy/{music_name}.npy')
        if self.spectral_feature == 'melspectrogram':
            feature = librosa.feature.melspectrogram(y=raw_audio_npy_file, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, \
                                                    n_mels=self.n_mels, fmax=self.fmax) 
        elif self.spectral_feature == 'mfcc':
            feature = librosa.feature.mfcc(y=y, sr=self.sr)
        model_input = feature[:, start_idx:end_idx]

        padding_list_for_input_feature = np.zeros((self.n_mels, padding_count))

        ground_truth_src_list = []
        for frame_idx in range(start_idx, end_idx):
            ground_truth_src_list.append(self._get_ground_truth_src_per_frame(music_name, frame_idx))
        ground_truth_src_list_to_onehot = []
        for src in ground_truth_src_list:
            if src == "KD":
                ground_truth_src_list_to_onehot.append([1,0,0,0])
            elif src == "SD":
                ground_truth_src_list_to_onehot.append([0,1,0,0])
            elif src == "HH":
                ground_truth_src_list_to_onehot.append([0,0,1,0])
            elif src == "NotOnset":
                ground_truth_src_list_to_onehot.append([0,0,0,1])
        for i in range(padding_count):
            ground_truth_src_list_to_onehot.append([0,0,0,1]) # Currently, padding is "also" considered as "NotOnset" -> This should be reconsidered later on.
        
        model_input = np.concatenate([model_input, padding_list_for_input_feature], axis=1).T

        # Model input shape : [Frame_size, Mel_num]
        # Model label shape : [Frame_size, 4] -> As there are only 4 lables to consider for now.
        return torch.Tensor(model_input), torch.Tensor(ground_truth_src_list_to_onehot)



if __name__ == '__main__':
    sampledata = MDB_Dataset()
    print(sampledata[150])