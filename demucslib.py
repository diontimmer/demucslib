import torch as th

from demucs.apply import apply_model
from demucs.audio import save_audio
from demucs.separate import load_track
from demucs.pretrained import (
    get_model,
)

from pathlib import Path
import os


class DemucsProcessor:
    def __init__(self, device='auto', model_name="htdemucs"):
        if device == 'auto':
            if th.has_mps:
                self.device = "mps"
            elif th.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

            

        self.loaded_model_name = model_name
        self.loaded_model = self.load_model(self.loaded_model_name)

    def load_model(self, model_name):
        """
        Load model from pretrained models

        Args:
            model_name (str): name of the model to load

        Returns:
            model (torch.nn.Module): model loaded from pretrained models
        """
        model = get_model(name=model_name)
        model.to(self.device)
        model.eval()
        return model

    def prepare_audio(self, file_path):
        """
        Prepare audio for processing

        Args:
            file_path (str): path to audio file

        Returns:
            wav (torch.Tensor): audio tensor
            ref (torch.Tensor): reference tensor
        """
        wav = load_track(
            file_path, self.loaded_model.audio_channels, self.loaded_model.samplerate
        )
        ref = wav.mean(0)
        wav -= ref.mean()
        wav /= ref.std()
        return wav, ref

    def save_result_dict(
        self,
        result_dict,
        out_folder=None,
        create_folder=True,
        ext="wav",
        clip="rescale",
        bits_per_sample=16,
        as_float=False,
        mp3_preset=2,
        mp3_bitrate=320,
    ):
        """
        Save the result dictionary to the output folder.

        Args:
            result_dict (dict): dictionary containing the results
            out_folder (str): path to the output folder
            create_folder (bool): create a folder with the audio-file name in the desired out_folder for the files.
            ext (str): extension of the output file
            clip (str): clipping method
            bits_per_sample (int): bits per sample when saving as float (16, 24, 32)
            as_float (bool): save as float32
            mp3_preset (int): mp3 preset when saving as mp3 (1,2,3,4,5,6,7)
            mp3_bitrate (int): mp3 bitrate when saving as mp3

        """

        # if out folder is none, save it next to the original file
        for path, result in result_dict.items():
            path = Path(path)
            out = Path(out_folder) if out_folder else path.parent
            out.mkdir(parents=True, exist_ok=True)
            for stem, source in result.items():
                # filename is og file name + the stem extension
                filename = f"{path.stem}_{stem}"
                out_filename = str(out / f"{filename}.{ext}") if not create_folder else str(out / f"{path.stem}/{filename}.{ext}")
                os.makedirs(os.path.dirname(out_filename), exist_ok=True)
                save_audio(
                    source,
                    out_filename,
                    samplerate=self.loaded_model.samplerate,
                    clip=clip,
                    as_float=as_float,
                    bits_per_sample=bits_per_sample,
                    bitrate=mp3_bitrate,
                    preset=mp3_preset,
                )

    def process_tracks(
        self,
        paths,
        shifts=1,
        overlap=0.25,
        split=True,
        segment=None,
        jobs=8,
        stem=None,
    ):
        '''
        Process tracks for source separation

        Args:
            paths (str or list): path to the audio file or list of paths
            shifts (int): number of random shifts for equivariant stabilization.
            overlap (float): overlap between splits
            split (bool): split the input into chunks
            segment (int): segment size
            jobs (int): number of parallel jobs
            stem (str): stem to separate
        
        '''


        paths = [paths] if not isinstance(paths, list) else paths
        total_results = {}
        for path in paths:
            wav, ref = self.prepare_audio(path)

            sources = apply_model(
                self.loaded_model,
                wav[None],
                device=self.device,
                shifts=shifts,
                split=split,
                overlap=overlap,
                progress=True,
                num_workers=jobs,
                segment=segment,
            )[0]
            sources *= ref.std()
            sources += ref.mean()

            result = {}

            if stem is None:
                for source, name in zip(sources, self.loaded_model.sources):
                    result[name] = source
            else:
                sources = list(sources)
                result[stem] = sources.pop(self.loaded_model.sources.index(stem))
                other_stem = th.zeros_like(sources[0])
                for i in sources:
                    other_stem += i
                result["no_" + stem] = other_stem

            total_results[path] = result
        return total_results
