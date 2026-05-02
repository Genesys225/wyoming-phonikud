import asyncio
import logging
from functools import partial

import numpy as np
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import Synthesize

_LOGGER = logging.getLogger(__name__)
_CHUNK_BYTES = 8192


class PhonikudHandler(AsyncEventHandler):
    def __init__(self, wyoming_info: Info, phonikud, piper, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._info = wyoming_info
        self._phonikud = phonikud
        self._piper = piper

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self._info.event())
            return True

        if Synthesize.is_type(event.type):
            synthesize = Synthesize.from_event(event)
            _LOGGER.debug("Synthesizing %d chars", len(synthesize.text))

            loop = asyncio.get_running_loop()
            audio_bytes, sample_rate = await loop.run_in_executor(
                None, partial(self._synthesize, synthesize.text)
            )

            await self.write_event(AudioStart(rate=sample_rate, width=2, channels=1).event())
            for i in range(0, len(audio_bytes), _CHUNK_BYTES):
                await self.write_event(
                    AudioChunk(
                        audio=audio_bytes[i : i + _CHUNK_BYTES],
                        rate=sample_rate,
                        width=2,
                        channels=1,
                    ).event()
                )
            await self.write_event(AudioStop().event())

        return True

    def _synthesize(self, text: str) -> tuple[bytes, int]:
        from phonikud import phonemize

        vocalized = self._phonikud.add_diacritics(text)
        phonemes = phonemize(vocalized)
        samples, sample_rate = self._piper.create(phonemes, is_phonemes=True)
        audio_bytes = (np.clip(samples, -1.0, 1.0) * 32767).astype("int16").tobytes()
        return audio_bytes, sample_rate
