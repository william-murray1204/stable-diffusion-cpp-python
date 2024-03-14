from . import stable_diffusion_cpp
from .stable_diffusion_types import StableDiffusionResult, StableDiffusionSegment, StableDiffusionToken
from typing import List, Literal, Any
import ctypes


class StableDiffusion():
    def __init__(self, model_path, strategy = 0, n_threads = 1):
        self.context = stable_diffusion_cpp.whisper_init_from_file(model_path.encode('utf-8'))
        self.params  = stable_diffusion_cpp.whisper_full_default_params(strategy)
        self.params.n_threads = n_threads
        self.params.print_special = False
        self.params.print_progress = False
        self.params.print_realtime = False
        self.params.print_timestamps = False
    
    def generate(self):
        print()

    # def transcribe(self, file, prompt = None, response_format = 'json', temperature = 0.8, language = 'en') -> Any:
    #     data, sr = librosa.load(file, sr=Whisper.WHISPER_SR)
    #     self.params.language = language.encode('utf-8')
    #     if prompt:
    #         self.params.initial_prompt = prompt.encode('utf-8')
    #     self.params.temperature = temperature
    #     result = self._full(data)
    #     return self._parse_format(result, response_format)

    # def translate(self, file, prompt = None, response_format = 'json', temperature = 0.8) -> Any:
    #     data, sr = librosa.load(file, sr=Whisper.WHISPER_SR)
    #     self.params.translate = True
    #     self.params.initial_prompt = prompt.encode('utf-8')
    #     self.params.temperature = temperature
    #     result = self._full(data)
    #     return self._parse_format(result, response_format)

    # def _full(self, data) -> StableDiffusionResult:
    #     # run the inference
    #     r = stable_diffusion_cpp.whisper_full(ctypes.c_void_p(self.context), self.params, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(data))
    #     if r != 0:
    #         raise "Error: {}".format(result)

    #     result: StableDiffusionResult = {
    #         "task": "translate" if self.params.translate else "transcribe",
    #         "language": self.params.language,
    #         "duration": librosa.get_duration(y=data, sr=Whisper.WHISPER_SR),
    #     }

    #     segments: List[StableDiffusionSegment] = []
    #     all_text = ''
    #     n_segments = stable_diffusion_cpp.whisper_full_n_segments(ctypes.c_void_p(self.context))
    #     for i in range(n_segments):
    #         t0  = stable_diffusion_cpp.whisper_full_get_segment_t0(ctypes.c_void_p(self.context), i)/100.0
    #         t1  = stable_diffusion_cpp.whisper_full_get_segment_t1(ctypes.c_void_p(self.context), i)/100.0
    #         txt = stable_diffusion_cpp.whisper_full_get_segment_text(ctypes.c_void_p(self.context), i).decode('utf-8')
    #         all_text += txt
    #         n_tokens = stable_diffusion_cpp.whisper_full_n_tokens(ctypes.c_void_p(self.context), i)
    #         tokens: List[StableDiffusionToken] = []
    #         for j in range(n_tokens):
    #             token_data = stable_diffusion_cpp.whisper_full_get_token_data(ctypes.c_void_p(self.context), i, j)
    #             tokens.append({
    #                 "id": token_data.id,
    #                 "prob": token_data.p,
    #                 "logprob": token_data.plog,
    #                 "pt": token_data.pt,
    #                 "pt_sum": token_data.ptsum,
    #             })
    #         segments.append({
    #             "start": t0,
    #             "end": t1,
    #             "text": txt,
    #             "tokens": tokens,
    #         })

    #     result["segments"] = segments
    #     result["text"] = all_text.strip()
    #     return result

    # def _parse_format(self, result: WhisperResult, response_format: Literal["json", "text", "srt", "verbose_json", "vtt"]):
    #     return {
    #         "json": self._parse_format_json,
    #         "text": self._parse_format_text,
    #         "srt": self._parse_format_srt,
    #         "verbose_json": self._parse_format_verbose_json,
    #         "vtt": self._parse_format_vtt,
    #     }[response_format](result)

    # def _parse_format_verbose_json(self, result: WhisperResult):
    #     return {
    #         "task": result["task"],
    #         "language": result["language"],
    #         "duration": result["duration"],
    #         "text": result["text"],
    #         "segments": [{
    #             "id": i,
    #             "seek": s['start'],
    #             "start": s['start'],
    #             "end": s['end'],
    #             "text": s['text'],
    #             "tokens": [t["id"] for t in s["tokens"]],
    #             "temperature": self.params.temperature + self.params.temperature_inc * i,
    #             "avg_logprob": sum([t["logprob"] for t in s["tokens"]])/len(s["tokens"]),
    #             "compression_ratio": self.params.entropy_thold,
    #             "no_speech_prob": 0.0,
    #             "transient": False,
    #         } for i, s in enumerate(result["segments"])],
    #     }

    # def _parse_format_json(self, result: WhisperResult):
    #     return {
    #         "text": result["text"],
    #     }

    # def _parse_format_text(self, result: WhisperResult):
    #     return result["text"]

    # def _parse_format_srt(self, result: WhisperResult):
    #     return '\n'.join([f'{i + 1}\n{Whisper.format_time(s["start"])} --> {Whisper.format_time(s["end"])}\n{s["text"]}\n' for i, s in enumerate(result["segments"])])

    # def _parse_format_vtt(self, result: WhisperResult):
    #     return '\n'.join([f'{i + 1}\n{Whisper.format_time(s["start"])} --> {Whisper.format_time(s["end"])} align:middle\n{s["text"]}\n' for i, s in enumerate(result["segments"])])

    # def __dealloc__(self):
    #     # free the memory
    #     stable_diffusion_cpp.whisper_free(ctypes.c_void_p(self.context))

    # @staticmethod
    # def format_time(t: int):
    #     msec = t * 10
    #     hr = msec / (1000 * 60 * 60)
    #     msec = msec - hr * (1000 * 60 * 60)
    #     minu = msec / (1000 * 60)
    #     msec = msec - minu * (1000 * 60)
    #     sec = msec / 1000
    #     msec = msec - sec * 1000
    #     return f'{int(hr):02}:{int(minu):02}:{int(sec):02}.{int(msec):03}'

