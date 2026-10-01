from pathlib import Path

WORKER_LIBRARIES = (Path(__file__).parent / "prompts/libraries.txt").read_text(encoding="utf-8")
