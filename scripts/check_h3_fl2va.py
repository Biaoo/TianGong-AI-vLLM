"""Generate an H3 FL2VA or text-only T2VA clip and validate its media streams."""

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


def request_body(images, indices, *, prompt, seconds, size, seed, task="fl2va"):
    if task not in {"fl2va", "t2va"}:
        raise ValueError("Expected fl2va or t2va")
    if task == "t2va" and (images or indices):
        raise ValueError("T2VA takes text only, without reference images or frame indices")
    if task == "fl2va" and (indices not in ([0], [-1], [0, -1]) or len(images) != len(indices)):
        raise ValueError("Expected first, last, or ordered first/last inputs")
    boundary = "h3-fl2va-" + uuid.uuid4().hex
    fields = {
        "prompt": prompt,
        "seconds": str(seconds),
        "size": size,
        "seed": str(seed),
        "num_inference_steps": "50",
        "flow_shift": "12",
        "extra_params": json.dumps(
            {"task": task, "audio_flow_shift": 3.0}
            | ({"frame_indices": indices} if task == "fl2va" else {"aspect_ratio": "16:9"})
        ),
    }
    body = bytearray()
    for name, value in fields.items():
        body.extend(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    for index, image in enumerate(images):
        if not image.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("References must be PNG images")
        body.extend(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="input_references"; '
                f'filename="frame-{index}.png"\r\nContent-Type: image/png\r\n\r\n'
            ).encode()
        )
        body.extend(image)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return bytes(body), boundary, fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--first", type=Path)
    parser.add_argument("--last", type=Path)
    parser.add_argument("--task", choices=["fl2va", "t2va"], default="fl2va")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--key-stdin", action="store_true")
    parser.add_argument("--seconds", type=int, choices=range(4, 16), default=5)
    parser.add_argument("--size", choices=["832x480", "1344x768"], default="832x480")
    parser.add_argument("--seed", type=int, default=2102)
    parser.add_argument(
        "--prompt",
        default="The subjects move naturally through the scene with synchronized ambient sound.",
    )
    args = parser.parse_args()
    if args.task == "fl2va" and not (args.first or args.last):
        parser.error("At least one of --first or --last is required")
    if args.task == "t2va" and (args.first or args.last):
        parser.error("T2VA must not include --first or --last")
    refs = [(0, args.first), (-1, args.last)]
    refs = [(index, path) for index, path in refs if path is not None]
    images = [path.read_bytes() for _, path in refs]
    body, boundary, fields = request_body(
        images,
        [index for index, _ in refs],
        prompt=args.prompt,
        seconds=args.seconds,
        size=args.size,
        seed=args.seed,
        task=args.task,
    )
    headers = {"Content-Type": "multipart/form-data; boundary=" + boundary}
    if args.key_stdin:
        key = sys.stdin.read().strip()
        if not key:
            parser.error("Nonempty inference key required on stdin")
        headers["Authorization"] = "Bearer " + key
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "generated.mp4"
    partial = output.with_suffix(".mp4.part")
    opener = build_opener(ProxyHandler({}))
    request = Request(args.base_url.rstrip("/") + "/v1/videos/sync", data=body, headers=headers)
    start = time.monotonic()
    with opener.open(request, timeout=7500) as response, partial.open("wb") as stream:
        if response.headers.get_content_type() != "video/mp4":
            raise RuntimeError("Expected MP4 response")
        response_info = {
            "status": response.status,
            "request_id": response.headers.get("X-Request-Id"),
        }
        while chunk := response.read(1024**2):
            stream.write(chunk)
    partial.rename(output)
    elapsed = round(time.monotonic() - start, 3)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    media = json.loads(probe.stdout)
    video = next(s for s in media["streams"] if s["codec_type"] == "video")
    audio = next(s for s in media["streams"] if s["codec_type"] == "audio")
    if (video["width"], video["height"]) != tuple(map(int, args.size.split("x"))):
        raise RuntimeError("Unexpected output size")
    if Fraction(video["avg_frame_rate"]) != 24:
        raise RuntimeError("Expected 24 FPS")
    if abs(float(media["format"]["duration"]) - args.seconds) > 0.5:
        raise RuntimeError("Unexpected output duration")
    if int(audio["sample_rate"]) != 32000 or audio["channels"] != 2:
        raise RuntimeError("Expected 32 kHz stereo audio")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", str(output), "-f", "null", "-"], check=True
    )
    for label, frame in [("first", 0), ("last", int(video["nb_frames"]) - 1)]:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(output),
                "-vf",
                f"select=eq(n\\,{frame})",
                "-frames:v",
                "1",
                str(args.output_dir / (label + ".png")),
            ],
            check=True,
        )
    result = {
        "request": fields,
        "response": response_info,
        "elapsed_seconds": elapsed,
        "reference_sha256": [hashlib.sha256(image).hexdigest() for image in images],
        "media": media,
        "media_decode": "passed",
        "visual_review": "pending",
        "endpoint_visual_review": "pending" if args.task == "fl2va" else "not_applicable",
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "video": str(output),
                "elapsed_seconds": elapsed,
                "media_decode": "passed",
                "visual_review": result["visual_review"],
                "endpoint_visual_review": result["endpoint_visual_review"],
            }
        )
    )


if __name__ == "__main__":
    main()
