import os
import numpy as np
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch import nn
import torchaudio
import torchaudio.transforms as T
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, list_dir, list_files

import maad
import librosa


class Darksound(VisionDataset):
    """`Darksound <https://github.com/joachimpoutaraud/darksound>`_ Dataset.

    Args:
        root (str): Root directory of dataset where directory ``darksound`` exists.
        split (str, optional): If "train", creates dataset from the train set, if "val" 
            creates dataset from the validation set, if "test" creates from the test set.
        hpss (bool, optional): Whether to apply Harmonic Percussive Source Separation (HPSS)
            on the spectrogram. Defaults to True.
        tfr (str, optional): Type of time-frequency representation. Possible to choose
            between 'spec', 'cqt' or 'mel'. Defaults to 'mel'.
        remove_background: Whether to remove background from the spectrogram. 
            Defaults to True.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset zip files from the internet and
            puts it in root directory. If the zip files are already downloaded, they are not
            downloaded again.
    """

    folder = "darksound"   
    zips_md5 = {"train":"b95ebadd31bdc149adc8d07fe779eeeb", 
                "val":"846f5d7457525012b94e7e9b5c215705", 
                "test":"6dbd0cc8950fed84ff91d17da6d8c994"}

    def __init__(
        self,
        root: str = os.getcwd(),
        split: str = 'train',
        hpss: bool = True,
        tfr: str = 'mel',
        remove_background: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False) -> None:

        super().__init__(os.path.join(root, self.folder), transform=transform, target_transform=target_transform)
        self.split = split
        self.hpss = hpss
        self.tfr = tfr
        self.remove_background = remove_background

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self.target_folder = os.path.join(self.root, self.split)
        self._family = list_dir(self.target_folder)
        self._species: List[str] = sum(
            ([os.path.join(a, c) for c in list_dir(os.path.join(self.target_folder, a))] for a in self._family), []
        )
        self._species_files = [
            [(audio, idx) for audio in list_files(os.path.join(self.target_folder, species), ".wav")]
            for idx, species in enumerate(self._species)
        ]
        self._flat_species_files: List[Tuple[str, int]] = sum(self._species_files, [])

    def __len__(self) -> int:
        return len(self._flat_species_files)
    
    def __getlabel__(self) -> list:
        return [instance[1] for instance in self._flat_species_files]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where label is index of the label species class.
        """
        # Load audio file
        audio, label = self._flat_species_files[index]
        audio_path = os.path.join(self.target_folder, self._species[label], audio)
        y, sr = self._load_audio(audio_path, pad=True)

        # Compute spectrogram
        Y = torch.flip(self._get_spectrogram(y, sr, tfr=self.tfr), dims=[0, 1])

        # Remove background 
        if self.remove_background:
            Y, _, _ = maad.sound.remove_background(Y.numpy()[0])
            # convert array to tensor
            Y = torch.from_numpy(Y).unsqueeze_(dim=0) 
        # Compute HP Source Separation
        if self.hpss:
            Y = self._source_separation(Y.numpy()[0])
        else:
            Y = Y.repeat(3, 1, 1)

        if self.transform:
            Y = self.transform(Y)
        if self.target_transform:
            label = self.target_transform(label)
            
        return Y, label
    
    def _load_audio(self, path, pad=False, seconds=3):
        # Load audio file and normalize it using torch
        y, sr = torchaudio.load(path, normalize=True)
        # Fade in and out to avoid aliasing from window effects.
        fade_transform = T.Fade(fade_in_len=int(sr/10), fade_out_len=int(sr/10), fade_shape='half_sine')
        y = fade_transform(y)

        if pad: # Pad audio file to a fix length in seconds
            samples = sr*seconds
            if y.shape[1] >= (samples):
                y.resize_(1, samples)
            else:
                diff = (samples) - y.shape[1]
                pad = nn.ConstantPad1d((int(np.ceil(diff/2)), int(np.floor(diff/2))), 0)
                y = pad(y)
        return y, sr
    
    def _get_spectrogram(self, y, sr, tfr='mel', n_fft=1024, hop_length=512, n_mels=128):
        # FT spectrogram
        if tfr == 'spec':
            spectrogram = T.Spectrogram(
                n_fft=n_fft,
                hop_length=hop_length,
                center=True,
                pad_mode="reflect",
                power=2.0)
        
        # CQT spectrogram
        if tfr == 'cqt':
            cqt = np.abs(librosa.cqt(y.numpy(), sr=sr))
            return torch.Tensor(cqt)

        # Mel spectrogram
        if tfr == 'mel':
            spectrogram = T.MelSpectrogram(
                sample_rate=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                center=True,
                pad_mode="reflect",
                power=2.0,
                norm='slaney',
                n_mels=n_mels)
            
        return spectrogram(y)
    
    def _scale_minmax(self, X, min=0.0, max=1.0):
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        return X_scaled
    
    def _source_separation(self, Y, margin=(1.0,5.0)):
        # Compute HP Source Separation
        H, P = librosa.decompose.hpss(Y, margin=margin)
        # Convert amplitude spectrogram to dB-scaled 
        H = librosa.amplitude_to_db(H, ref=np.max)
        P = librosa.amplitude_to_db(P, ref=np.max)
        Y = librosa.amplitude_to_db(Y, ref=np.max)

        # Compute delta features
        D = librosa.feature.delta(Y)

        # Normalize spectrogram
        H = self._scale_minmax(H)
        P = self._scale_minmax(P)
        D = self._scale_minmax(D) 

        HPD = np.nan_to_num(np.transpose(np.asarray(np.dstack((H,P,D))), (2,0,1)))
        return torch.from_numpy(HPD)

    def _check_integrity(self) -> bool:
        zip_filename = self.split
        if not check_integrity(os.path.join(self.root, zip_filename + ".zip"), self.zips_md5[zip_filename]):
            return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        zip_filename = self.split + ".zip"
        
        if self.split == 'train':
            url = "https://drive.google.com/file/d/1GUgpDaf7AqzqIKwD9_Rb8sH643l1CrqQ/view?usp=share_link" 
        elif self.split == 'val':
            url = "https://drive.google.com/file/d/1JPKfPUrPsXVS8mKFoGVGcDjYUuQ804B3/view?usp=share_link"
        elif self.split == 'test':
            url = "https://drive.google.com/file/d/1DKXRf-Y2KbmkQ8Civ-mdiNa9mhykqqoS/view?usp=share_link"
        else:
            print("Dataset not found or corrupted. Enter either split='train, 'val' or 'test'.")
            
        download_and_extract_archive(url, self.root, filename=zip_filename, md5=self.zips_md5[self.split])
