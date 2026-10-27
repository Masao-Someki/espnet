"""Fingerprint runner configuration and explicitly referenced input files."""

import hashlib
import os
from pathlib import Path

import cloudpickle
from omegaconf import OmegaConf


def resume_fingerprint(provider, runner_type, output_dir):
    """Identify outputs by provider settings, runner type and input file content.

    Providers accessing mutable external sources that are not represented by
    config values or file paths can implement ``resume_identity()``. Its return
    value must describe the dataset/model revision, not transient worker state.
    Legacy manifests without a fingerprint cannot be resumed safely.
    """
    if hasattr(provider, "resume_identity"):
        identity = provider.resume_identity()
    else:
        identity = provider.config if hasattr(provider, "config") else vars(provider)
    if OmegaConf.is_config(identity):
        identity = OmegaConf.to_container(identity, resolve=True)

    files = {}
    output_root = Path(output_dir).resolve()

    output_keys = {"exp_dir", "inference_dir", "stats_dir", "output_dir", "log_dir"}
    output_paths = {output_root}

    def collect_output_paths(value):
        if isinstance(value, dict):
            for name, child in value.items():
                if name in output_keys and isinstance(child, (str, Path)):
                    output_paths.add(Path(child).resolve())
                else:
                    collect_output_paths(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect_output_paths(child)

    collect_output_paths(identity)

    def directory_identity(root):
        excluded = output_paths
        entries = []
        seen = set()
        transient = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".stage_sync",
            ".config_history",
        }
        for current, directories, names in os.walk(root, followlinks=True):
            current = Path(current)
            stat = current.stat()
            inode = (stat.st_dev, stat.st_ino)
            if inode in seen:
                directories[:] = []
                continue
            seen.add(inode)
            directories[:] = [
                name
                for name in directories
                if name not in transient and (current / name).resolve() not in excluded
            ]
            for name in names:
                child = current / name
                if child.is_file() and not name.endswith((".lock", ".tmp", ".partial")):
                    stat = child.stat()
                    entries.append(
                        (
                            str(child.relative_to(root)),
                            stat.st_size,
                            stat.st_mtime_ns,
                            stat.st_ctime_ns,
                        )
                    )
        return sorted(entries)

    def visit(value, key=""):
        if isinstance(value, dict):
            for name, child in value.items():
                visit(child, str(name))
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child, key)
        elif isinstance(value, (str, Path)):
            # Names such as split="test" or mode="train" are not paths even
            # when the current checkout happens to contain a matching directory.
            path_key = any(
                part in key.lower()
                for part in (
                    "path",
                    "dir",
                    "file",
                    "root",
                    "checkpoint",
                    "manifest",
                )
            )
            if (
                not isinstance(value, Path)
                and not path_key
                and not any(character in value for character in ("/", "\\", "."))
            ):
                return
            try:
                path = Path(value).resolve()
                directory = path.is_dir()
                regular_file = path.is_file()
            except (OSError, ValueError):
                return
            if directory:
                if (
                    key in output_keys
                    or path == output_root
                    or output_root in path.parents
                ):
                    return
                files[str(path)] = directory_identity(path)
            elif regular_file:
                # Once a referenced input is identified, read errors must not
                # silently remove it from identity checking.
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                files[str(path)] = digest.hexdigest()

    visit(identity)
    descriptor = (
        f"{type(provider).__module__}.{type(provider).__qualname__}",
        f"{runner_type.__module__}.{runner_type.__qualname__}",
        identity,
        sorted(files.items()),
    )
    return hashlib.sha256(cloudpickle.dumps(descriptor)).hexdigest()
