"""Static-analysis fixtures only; never execute these example calls."""

import io
import os
import subprocess
import tempfile
from pathlib import Path
from subprocess import run as run_process
from tempfile import NamedTemporaryFile as named_file

import subprocess_utils as utils
from subprocess_utils import run_git_command, run_git_command_with_input as git_input, run_safe_command


def run_git_command_with_input(payload, argv, options) -> None:
    # Original failure: the text defaults were hidden in shared kwargs.
    kwargs = {"text": True, "encoding": "utf-8"}
    # ruleid: mcmc.python.git-stdin-binary-transport
    subprocess.run(argv, input=payload, **kwargs)
    # ruleid: mcmc.python.git-stdin-binary-transport
    subprocess.run(argv, input=payload, text=True)
    # ruleid: mcmc.python.git-stdin-binary-transport
    subprocess.run(argv, input=payload.encode(), text=False, encoding="utf-8")
    # ruleid: mcmc.python.git-stdin-binary-transport
    subprocess.run(argv, input=payload.encode(), text=False, errors="strict")
    # ruleid: mcmc.python.git-stdin-binary-transport
    subprocess.run(argv, input=payload.encode(), text=False, universal_newlines=True)
    # ruleid: mcmc.python.git-stdin-binary-transport
    run_process(argv, input=payload, text=True)
    # ruleid: mcmc.python.git-stdin-binary-transport
    subprocess.Popen(argv, stdin=subprocess.PIPE)
    # ruleid: mcmc.python.git-stdin-binary-transport
    subprocess.Popen(argv, stdin=subprocess.PIPE, text=False, encoding="utf-8")

    # ok: mcmc.python.git-stdin-binary-transport
    subprocess.run(argv, input=payload.encode("utf-8"), text=False, **options)
    # ok: mcmc.python.git-stdin-binary-transport
    run_process(argv, input=payload.encode(), text=False)
    # ok: mcmc.python.git-stdin-binary-transport
    subprocess.run(argv, input=payload.encode(), text=False, encoding=None, errors=None, universal_newlines=False)
    # ok: mcmc.python.git-stdin-binary-transport
    subprocess.Popen(argv, stdin=subprocess.PIPE, text=False)


def git_input_routing(payload, argv) -> None:
    # ruleid: mcmc.python.git-stdin-use-shared-helper
    run_git_command(argv, input=payload)
    # ruleid: mcmc.python.git-stdin-use-shared-helper
    utils.run_git_command(argv, input=payload)
    # ruleid: mcmc.python.git-stdin-use-shared-helper
    run_safe_command("git", argv, input=payload)
    # ruleid: mcmc.python.git-stdin-use-shared-helper
    utils.run_safe_command("git", argv, input=payload)
    # ruleid: mcmc.python.git-stdin-use-shared-helper
    subprocess.run(["git", "hash-object", "--stdin"], input=payload, text=True)
    # ruleid: mcmc.python.git-stdin-use-shared-helper
    run_process(["git", "hash-object", "--stdin"], input=payload, text=True)

    # ok: mcmc.python.git-stdin-use-shared-helper
    git_input(argv, payload)
    # ok: mcmc.python.git-stdin-use-shared-helper
    utils.run_git_command_with_input(argv, input_data=payload)
    # ok: mcmc.python.git-stdin-use-shared-helper
    run_git_command(argv)
    # ok: mcmc.python.git-stdin-use-shared-helper
    run_git_command(argv, input=None)
    # Ordinary text-mode subprocesses outside the Git-input helper are allowed.
    # ok: mcmc.python.git-stdin-use-shared-helper, mcmc.python.git-stdin-binary-transport
    run_safe_command("ruff", ["check", "-"], input=payload)
    # ok: mcmc.python.git-stdin-use-shared-helper, mcmc.python.git-stdin-binary-transport
    subprocess.run(["formatter"], input=payload, text=True, encoding="utf-8")


def path_text_writes(path: Path, contents: str, newline_policy: str) -> None:
    # The real extracted-notebook writer, before and after explicit LF output.
    # ruleid: mcmc.python.text-writes-explicit-policy
    path.write_text(contents, encoding="utf-8")
    # ok: mcmc.python.text-writes-explicit-policy
    path.write_text(contents, encoding="utf-8", newline="\n")
    # ruleid: mcmc.python.text-writes-explicit-policy
    path.write_text(contents, newline="\n")
    # ruleid: mcmc.python.text-writes-explicit-policy
    path.write_text(contents, encoding=None, newline="\n")
    # ruleid: mcmc.python.text-writes-explicit-policy
    path.write_text(contents, encoding="utf-8", newline=None)
    # ok: mcmc.python.text-writes-explicit-policy
    path.write_text(contents, encoding="utf-8", newline="")
    # ok: mcmc.python.text-writes-explicit-policy
    path.write_text(contents, encoding="utf-8", newline=newline_policy)
    # Nested reads and text transformations are not file writers.
    # ok: mcmc.python.text-writes-explicit-policy
    path.write_text(contents.strip(), encoding="utf-8", newline="\n")
    # ok: mcmc.python.text-writes-explicit-policy
    path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    # ok: mcmc.python.text-writes-explicit-policy
    path.write_text(open(path, encoding="utf-8").read(), encoding="utf-8", newline="\n")
    # ok: mcmc.python.text-writes-explicit-policy
    path.write_text(path.read_text(encoding=None), encoding="utf-8", newline="\n")
    # ok: mcmc.python.text-writes-explicit-policy
    path.write_text(open(path, encoding=None, newline=None).read(), encoding="utf-8", newline="\n")


def temporary_text_writes(path: Path) -> None:
    # Production writers use both positional and keyword mode forms.
    # ruleid: mcmc.python.text-writes-explicit-policy
    tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False)
    # ok: mcmc.python.text-writes-explicit-policy
    tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False)
    # ruleid: mcmc.python.text-writes-explicit-policy
    tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    )
    # ok: mcmc.python.text-writes-explicit-policy
    tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        delete=False,
    )
    # ruleid: mcmc.python.text-writes-explicit-policy
    named_file(mode="w", encoding="utf-8")
    # ok: mcmc.python.text-writes-explicit-policy
    named_file(mode="w", encoding="utf-8", newline="\n")
    # ruleid: mcmc.python.text-writes-explicit-policy
    named_file(mode="w", encoding="utf-8", newline=None)
    # ruleid: mcmc.python.text-writes-explicit-policy
    tempfile.TemporaryFile("w+", encoding="utf-8")
    # ok: mcmc.python.text-writes-explicit-policy
    tempfile.TemporaryFile("w+", encoding="utf-8", newline="\n")
    # ruleid: mcmc.python.text-writes-explicit-policy
    tempfile.TemporaryFile(mode="w+", encoding="utf-8")
    # ok: mcmc.python.text-writes-explicit-policy
    tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="\n")


def other_file_writers(path: Path, descriptor: int) -> None:
    # ruleid: mcmc.python.text-writes-explicit-policy
    open(path, "w", encoding="utf-8")
    # ok: mcmc.python.text-writes-explicit-policy
    open(path, "w", encoding="utf-8", newline="\n")
    # ruleid: mcmc.python.text-writes-explicit-policy
    open(path, mode="a", encoding="utf-8")
    # ok: mcmc.python.text-writes-explicit-policy
    open(path, mode="a", encoding="utf-8", newline="\n")
    # ruleid: mcmc.python.text-writes-explicit-policy
    path.open("x", encoding="utf-8")
    # ok: mcmc.python.text-writes-explicit-policy
    path.open("x", encoding="utf-8", newline="\n")
    # ruleid: mcmc.python.text-writes-explicit-policy
    path.open(mode="r+", encoding="utf-8")
    # ok: mcmc.python.text-writes-explicit-policy
    path.open(mode="r+", encoding="utf-8", newline="\n")
    # ruleid: mcmc.python.text-writes-explicit-policy
    path.open(mode="r+", encoding=None, newline="\n")
    # ruleid: mcmc.python.text-writes-explicit-policy
    io.open(path, "wt", encoding="utf-8")
    # ok: mcmc.python.text-writes-explicit-policy
    io.open(path, "wt", encoding="utf-8", newline="\n")
    # ruleid: mcmc.python.text-writes-explicit-policy
    io.open(path, mode="a+", encoding="utf-8")
    # ok: mcmc.python.text-writes-explicit-policy
    io.open(path, mode="a+", encoding="utf-8", newline="\n")
    # ruleid: mcmc.python.text-writes-explicit-policy
    os.fdopen(descriptor, "w", encoding="utf-8")
    # ok: mcmc.python.text-writes-explicit-policy
    os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
    # ruleid: mcmc.python.text-writes-explicit-policy
    os.fdopen(descriptor, mode="w", encoding="utf-8")
    # ok: mcmc.python.text-writes-explicit-policy
    os.fdopen(descriptor, mode="w", encoding="utf-8", newline="\n")


def byte_writes_and_reads_are_exempt(path: Path, descriptor: int) -> None:
    # ok: mcmc.python.text-writes-explicit-policy
    path.write_bytes(b"line\r\n")
    # ok: mcmc.python.text-writes-explicit-policy
    os.fdopen(descriptor, "wb")
    # ok: mcmc.python.text-writes-explicit-policy
    path.open("wb")
    # ok: mcmc.python.text-writes-explicit-policy
    open(path, "rb")
    # ok: mcmc.python.text-writes-explicit-policy
    tempfile.NamedTemporaryFile("w+b")
    # ok: mcmc.python.text-writes-explicit-policy
    tempfile.NamedTemporaryFile()
    # ok: mcmc.python.text-writes-explicit-policy
    tempfile.TemporaryFile(mode="wb")
    # ok: mcmc.python.text-writes-explicit-policy
    path.open(encoding="utf-8")
    # ok: mcmc.python.text-writes-explicit-policy
    io.open(path, "r", encoding="utf-8")
    # ok: mcmc.python.text-writes-explicit-policy
    path.read_text(encoding="utf-8")
