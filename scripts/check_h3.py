"""Generate and decode one H3 video through the authenticated GPUStack route."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import ProxyHandler, Request, build_opener
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--key-stdin", action="store_true", required=True)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument(
        "--prompt",
        default=(
            "The subject moves gently as the camera slowly moves closer. "
            "Soft ambient sounds accompany the movement."
        ),
    )
    args = parser.parse_args()
    key = sys.stdin.read().strip()
    if not key:
        parser.error("A nonempty inference key is required on stdin")
    reference = args.reference.read_bytes()
    if not reference.startswith(b"\x89PNG\r\n\x1a\n") or len(reference) > 10 * 1024**2:
        parser.error("Reference must be a PNG of at most 10 MiB")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    boundary = "h3-" + uuid.uuid4().hex
    body = bytearray()
    fields = {"prompt": args.prompt, "size": "832x480", "seconds": "5", "seed": str(args.seed)}
    for name, value in fields.items():
        body.extend(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    body.extend(
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="input_reference"; '
            'filename="reference.png"\r\nContent-Type: image/png\r\n\r\n'
        ).encode()
    )
    body.extend(reference)
    body.extend(f"\r\n--{boundary}--\r\n".encode())
    opener = build_opener(ProxyHandler({}))
    request = Request(
        args.base_url.rstrip("/") + "/v1/videos/sync",
        data=bytes(body),
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "multipart/form-data; boundary=" + boundary,
        },
    )
    start = time.monotonic()
    output = args.output_dir / "generated.mp4"
    with opener.open(request, timeout=7500) as response, output.open("wb") as stream:
        if "video/" not in response.headers.get("Content-Type", ""):
            raise RuntimeError("Expected a video response")
        while chunk := response.read(1024**2):
            stream.write(chunk)
    elapsed = time.monotonic() - start
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    media = json.loads(probe.stdout)
    video = next(s for s in media["streams"] if s["codec_type"] == "video")
    audio = next(s for s in media["streams"] if s["codec_type"] == "audio")
    if (video["width"], video["height"]) != (832, 480):
        raise RuntimeError("Unexpected video dimensions")
    if Fraction(video["avg_frame_rate"]) != 24 or int(video["nb_frames"]) != 124:
        raise RuntimeError("Expected 124 frames at 24 FPS")
    if int(audio["sample_rate"]) != 32000 or audio["channels"] != 2:
        raise RuntimeError("Expected 32 kHz stereo audio")
    if not 5 <= float(media["format"]["duration"]) <= 5.5:
        raise RuntimeError("Unexpected clip duration")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", str(output), "-f", "null", "-"], check=True
    )
    result = {
        "request": fields,
        "reference_sha256": hashlib.sha256(reference).hexdigest(),
        "elapsed_seconds": round(elapsed, 3),
        "media": media,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "scope": "One 480P reference-to-video sample with stereo audio; human quality review pending",
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "video": str(output),
                "elapsed_seconds": round(elapsed, 3),
                "media_decode": "passed",
                "quality_review": "pending",
            }
        )
    )


if __name__ == "__main__":
    main()
