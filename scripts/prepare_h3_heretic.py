"""Package the pinned Heretic BF16 encoder with the existing FastH3 components."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct

REVISION = "e8967f6a39ea5b4939a1aff81be3e8706490c0e8"
SOURCE_FILE = "qwen3vl_32b_h3_ultra_uncensored_heretic_bf16.safetensors"
SOURCE_SHA256 = "bbcd92a732e911cfafd86960e0e26aacc6efe949e02f16a9641f201c62984860"
FAST_REVISION = "f01e6a23ac6066aa9eb622cf94b6259442e70ec2"
BASE_REVISION = "9bfb6693f2cf6de171db46d1aa586f67d773a1da"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_header(path):
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < size <= 16 * 1024**2:
            raise ValueError("Invalid safetensors header size")
        header = json.loads(stream.read(size))
    return header, 8 + size


def native_name(name):
    if name.startswith("model.layers.") or name == "model.embed_tokens.weight":
        return "model.language_model." + name.removeprefix("model.")
    if name.startswith("visual."):
        return "model." + name
    raise ValueError(f"Unexpected encoder tensor: {name}")


def converted_header(header, expected):
    converted = {"__metadata__": {"format": "pt"}}
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        target = native_name(name)
        if target in converted:
            raise ValueError("Duplicate mapped tensor")
        if tensor["dtype"] != "BF16" or tensor["shape"] != expected.get(target):
            raise ValueError(f"Tensor shape/dtype mismatch: {name}")
        start, end = tensor["data_offsets"]
        if start < 0 or end - start != math.prod(tensor["shape"]) * 2:
            raise ValueError(f"Invalid tensor byte range: {name}")
        converted[target] = tensor
    if set(converted) - {"__metadata__"} != set(expected):
        raise ValueError("Missing native encoder tensors")
    return converted


def check_file(path, entry):
    if path.stat().st_size != entry["size"]:
        raise ValueError(f"Checksum mismatch: {entry['path']}")
    if entry["sha256"]:
        valid = sha256(path) == entry["sha256"]
    else:
        digest = hashlib.sha1(f"blob {entry['size']}\0".encode())
        digest.update(path.read_bytes())
        valid = digest.hexdigest() == entry["git_oid"]
    if not valid:
        raise ValueError(f"Checksum mismatch: {entry['path']}")


def verify(root, expected_hash, allowed_root):
    manifest_path = root / "assembly-manifest.json"
    if sha256(manifest_path) != expected_hash:
        raise ValueError("Assembly manifest digest does not match the runtime release")
    manifest = json.loads(manifest_path.read_text())
    if manifest["encoder_source_sha256"] != SOURCE_SHA256:
        raise ValueError("Unexpected encoder source")
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.resolve().is_relative_to(allowed_root.resolve()):
            raise ValueError("Assembly file escapes the model mount")
        check_file(path, entry)
    print(json.dumps({"verified_files": len(manifest["files"]), "manifest_sha256": expected_hash}))


def prepare(source, fast, native, output):
    if source.parent.name != REVISION or fast.name != FAST_REVISION:
        raise ValueError("Expected pinned source and FastH3 revisions")
    if native.name != "FL2VA" or native.parent.name != BASE_REVISION:
        raise ValueError("Expected pinned native H3 metadata")
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("Heretic source checksum mismatch")
    if output.exists():
        raise ValueError("Output already exists; use a new assembly directory")
    source_header, data_start = read_header(source)
    index = json.loads((native / "text_encoder/model.safetensors.index.json").read_text())
    expected = {}
    for shard in sorted(set(index["weight_map"].values())):
        header, _ = read_header(native / "text_encoder" / shard)
        for name, tensor in header.items():
            if (
                name.startswith("model.visual.")
                or name == "model.language_model.embed_tokens.weight"
            ):
                expected[name] = tensor["shape"]
            elif name.startswith("model.language_model.layers.") and int(name.split(".")[3]) < 50:
                expected[name] = tensor["shape"]
    if len(expected) != 902:
        raise ValueError("Unexpected native H3 encoder structure")
    converted = converted_header(source_header, expected)
    output.mkdir(parents=True)
    encoder = output / "text_encoder"
    encoder.mkdir()
    payload = json.dumps(converted, separators=(",", ":")).encode()
    payload += b" " * (-len(payload) % 8)
    weight = encoder / "model.safetensors"
    with source.open("rb") as src, weight.with_suffix(".part").open("xb") as dst:
        dst.write(struct.pack("<Q", len(payload)))
        dst.write(payload)
        src.seek(data_start)
        shutil.copyfileobj(src, dst, 8 * 1024**2)
    weight.with_suffix(".part").replace(weight)
    # Only tensor names/header change: the entire BF16 tensor payload is copied verbatim.
    native_manifest = json.loads((native.parent / "snapshot-manifest.json").read_text())
    config_entry = next(
        e for e in native_manifest["files"] if e["path"] == "FL2VA/text_encoder/config.json"
    )
    check_file(native / "text_encoder/config.json", config_entry)
    shutil.copyfile(native / "text_encoder/config.json", encoder / "config.json")
    entries = []
    fast_manifest = json.loads((fast / "snapshot-manifest.json").read_text())
    for entry in fast_manifest["files"]:
        name = entry["path"]
        if name.startswith(("text_encoder/", "transformer/", ".")) or name == "README.md":
            continue
        check_file(fast / name, entry)
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(fast / name)
        entries.append(
            {"path": name, "size": entry["size"], "sha256": entry["sha256"] or sha256(fast / name)}
        )
    for path in (weight, encoder / "config.json"):
        entries.append(
            {
                "path": str(path.relative_to(output)),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "encoder_source_revision": REVISION,
        "encoder_source_sha256": SOURCE_SHA256,
        "base_revision": BASE_REVISION,
        "fast_revision": FAST_REVISION,
        "tensor_count": 902,
        "conversion": "BF16 payload unchanged; ComfyUI names mapped to native Qwen3-VL names",
        "files": sorted(entries, key=lambda e: e["path"]),
    }
    path = output / "assembly-manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"assembled": str(output), "manifest_sha256": sha256(path)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--fast", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--allowed-root", type=Path)
    args = parser.parse_args()
    if args.verify_only:
        if not args.expected_manifest_sha256 or not args.allowed_root:
            parser.error("Verification requires the release digest and allowed model mount")
        verify(args.output, args.expected_manifest_sha256, args.allowed_root)
    else:
        if not all((args.source, args.fast, args.native)):
            parser.error("Preparation requires source, fast and native paths")
        prepare(args.source, args.fast, args.native, args.output)


if __name__ == "__main__":
    main()
