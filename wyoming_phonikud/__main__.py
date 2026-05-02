import argparse
import asyncio
import logging
from functools import partial
from pathlib import Path

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer

from .handler import PhonikudHandler

_LOGGER = logging.getLogger(__name__)

_HF_G2P_REPO = "thewh1teagle/phonikud-onnx"
_HF_TTS_REPO = "thewh1teagle/phonikud-tts-checkpoints"
_G2P_FILE = "phonikud-1.0.int8.onnx"
_CONFIG_FILE = "model.config.json"


def _download_models(data_dir: Path, voice: str) -> tuple[Path, Path, Path]:
    from huggingface_hub import hf_hub_download

    data_dir.mkdir(parents=True, exist_ok=True)

    g2p_path = data_dir / _G2P_FILE
    if not g2p_path.exists():
        _LOGGER.info("Downloading G2P model...")
        hf_hub_download(repo_id=_HF_G2P_REPO, filename=_G2P_FILE, local_dir=str(data_dir))

    tts_onnx = data_dir / f"{voice}.onnx"
    if not tts_onnx.exists():
        _LOGGER.info("Downloading %s voice...", voice)
        hf_hub_download(
            repo_id=_HF_TTS_REPO, filename=f"{voice}.onnx", local_dir=str(data_dir)
        )

    tts_config = data_dir / _CONFIG_FILE
    if not tts_config.exists():
        _LOGGER.info("Downloading TTS config...")
        hf_hub_download(
            repo_id=_HF_TTS_REPO, filename=_CONFIG_FILE, local_dir=str(data_dir)
        )

    return g2p_path, tts_onnx, tts_config


def _make_info(voice: str) -> Info:
    attr = Attribution(
        name="thewh1teagle",
        url="https://github.com/thewh1teagle/phonikud-tts",
    )
    return Info(
        tts=[
            TtsProgram(
                name="phonikud",
                description="Hebrew TTS — Phonikud + Piper ONNX",
                attribution=attr,
                installed=True,
                version="1.0.0",
                voices=[
                    TtsVoice(
                        name=f"he_IL-{voice}-medium",
                        description=f"Hebrew {voice.capitalize()} (male)",
                        attribution=attr,
                        installed=True,
                        languages=["he-IL"],
                        version="1.0.0",
                    )
                ],
            )
        ]
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Wyoming server for Phonikud Hebrew TTS")
    parser.add_argument("--port", type=int, default=10201)
    parser.add_argument("--data-dir", default="/data")
    parser.add_argument("--voice", default="shaul", choices=["shaul", "michael"])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    data_dir = Path(args.data_dir)
    g2p_path, tts_onnx, tts_config = _download_models(data_dir, args.voice)

    _LOGGER.info("Loading models...")
    from phonikud_onnx import Phonikud
    from piper_onnx import Piper

    phonikud = Phonikud(str(g2p_path))
    piper = Piper(str(tts_onnx), str(tts_config))

    wyoming_info = _make_info(args.voice)
    server = AsyncServer.from_uri(f"tcp://0.0.0.0:{args.port}")
    _LOGGER.info("Listening on port %d (voice: %s)", args.port, args.voice)

    await server.run(partial(PhonikudHandler, wyoming_info, phonikud, piper))


asyncio.run(main())
