"""Protect source identity and native H3 conditioning layout during conversion."""

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "prepare_heretic", Path(__file__).resolve().parents[1] / "scripts/prepare_h3_heretic.py"
)
package = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package)


def test_checks_git_blob_metadata_before_recording_sha256(tmp_path):
    path = tmp_path / "config.json"
    data = b'{"model_type":"qwen3_vl"}'
    path.write_bytes(data)
    entry = {
        "path": path.name,
        "size": len(data),
        "sha256": None,
        "git_oid": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
    }
    package.check_file(path, entry)
    path.write_bytes(data.replace(b"qwen3", b"qwen2"))
    with pytest.raises(ValueError, match="Checksum"):
        package.check_file(path, entry)


def test_native_names_preserve_tensor_payload_offsets():
    tensor = {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]}
    original = {"model.layers.31.self_attn.o_proj.weight": tensor}
    mapped = "model.language_model.layers.31.self_attn.o_proj.weight"
    assert package.converted_header(original, {mapped: [2, 3]})[mapped] == tensor
    assert package.native_name("visual.merger.norm.weight") == "model.visual.merger.norm.weight"
    assert (
        package.native_name("model.embed_tokens.weight")
        == "model.language_model.embed_tokens.weight"
    )


@pytest.mark.parametrize("name", ["lm_head.weight", "model.norm.weight", "arbitrary.weight"])
def test_rejects_generation_tail_and_unknown_tensors(name):
    with pytest.raises(ValueError, match="Unexpected"):
        package.native_name(name)


@pytest.mark.parametrize(
    "dtype,shape,offsets",
    [("I8", [2, 3], [0, 6]), ("BF16", [3, 2], [0, 12]), ("BF16", [2, 3], [0, 11])],
)
def test_rejects_wrong_precision_topology_or_offsets(dtype, shape, offsets):
    header = {
        "model.embed_tokens.weight": {"dtype": dtype, "shape": shape, "data_offsets": offsets}
    }
    with pytest.raises(ValueError):
        package.converted_header(header, {"model.language_model.embed_tokens.weight": [2, 3]})


def test_rejects_missing_tensor():
    with pytest.raises(ValueError, match="Missing"):
        package.converted_header({}, {"model.language_model.embed_tokens.weight": [2, 3]})


def test_verifier_rejects_tampering_and_symlink_escape(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    weight = root / "weight.bin"
    weight.write_bytes(b"original")
    entry = {"path": "weight.bin", "size": 8, "sha256": package.sha256(weight)}
    manifest = root / "assembly-manifest.json"
    manifest.write_text(
        json.dumps({"encoder_source_sha256": package.SOURCE_SHA256, "files": [entry]})
    )
    digest = package.sha256(manifest)
    package.verify(root, digest, root)
    weight.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Checksum"):
        package.verify(root, digest, root)
    weight.unlink()
    other = tmp_path / "elsewhere"
    other.write_bytes(b"original")
    weight.symlink_to(other)
    with pytest.raises(ValueError, match="escapes"):
        package.verify(root, digest, root)
    with pytest.raises(ValueError, match="digest"):
        package.verify(root, "0" * 64, tmp_path)


def test_native_assembly_replaces_only_encoder_and_keeps_both_tasks(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    native = tmp_path / package.BASE_REVISION / "FL2VA"
    native.mkdir(parents=True)
    encoder = existing / "text_encoder/model.safetensors"
    encoder.parent.mkdir()
    encoder.write_bytes(b"converted-heretic")
    original = native / "text_encoder/original.safetensors"
    original.parent.mkdir()
    original.write_bytes(b"original-encoder")
    index = native / "model_index.json"
    index.write_text(
        json.dumps({"_minimax_h3": {"partition": "fl2va", "tasks": ["fl2va", "t2va"]}})
    )

    def entry(path, root):
        return {
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "sha256": package.sha256(path),
        }

    manifest = existing / "assembly-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "encoder_source_sha256": package.SOURCE_SHA256,
                "tensor_count": 902,
                "files": [entry(encoder, existing)],
            }
        )
    )
    (native.parent / "snapshot-manifest.json").write_text(
        json.dumps({"files": [entry(index, native.parent), entry(original, native.parent)]})
    )
    output = tmp_path / "new/FL2VA"
    package.prepare_native(existing, package.sha256(manifest), native, output, tmp_path)
    assert (output / "text_encoder/model.safetensors").resolve() == encoder
    assert not (output / "text_encoder/original.safetensors").exists()
    assert (output / "model_index.json").resolve() == index
    package.verify(output, package.sha256(output / "assembly-manifest.json"), tmp_path)
