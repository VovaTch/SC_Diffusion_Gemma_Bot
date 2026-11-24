"""
Watch a live bot-vs-bot StarCraft game over VNC (headful scbw).

Needs a VNC viewer on the host (`sudo apt install tigervnc-viewer`).
Run from the project root:
    uv run python -m starcraft_diffusion_gemma.watch
    uv run python -m starcraft_diffusion_gemma.watch --bots Stardust "Dave Churchill" --speed 60
    uv run python -m starcraft_diffusion_gemma.watch --scale 3 --window windowed

The viewer is upscaled with gamescope (FSR by default); `--scale 0` disables it.

Unknown args are passed through to `scbw.play`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
WATCH_IMAGE = "starcraft:watch"  # starcraft:game + xdotool
WATCH_DOCKERFILE = ROOT_DIR / "docker" / "watch.dockerfile"
DEFAULT_BOTS = ["Stardust", "Iron bot"]
DEFAULT_MAP = "sscai/(2)Benzene.scx"
FASTEST_SPEED = 42  # ms per frame, BW "Fastest"
GAME_RES = (640, 480)  # container Xvfb / BW resolution
FILTERS = ["fsr", "nis", "linear", "nearest", "pixel"]
WINDOW_MODES = ["windowed", "borderless", "fullscreen"]
BACKENDS = ["default", "sdl", "wayland"]  # gamescope nested backends


@dataclass(frozen=True)
class Upscale:
    """
    gamescope upscaling of the VNC viewer, read by `scripts/vnc-viewer`.

    Args:
      scale (float): Output size as a multiple of the 640x480 game.
      filter (str): gamescope filter, one of FILTERS.
      sharpness (int): FSR/NIS sharpness, 0 (max) to 20 (min).
      window (str): gamescope window mode, one of WINDOW_MODES.
      backend (str): gamescope backend, one of BACKENDS.
    """

    scale: float
    filter: str = "fsr"
    sharpness: int = 2
    window: str = "borderless"
    backend: str = "default"

    def to_env(self) -> dict[str, str]:
        """
        Env vars consumed by `scripts/vnc-viewer`.

        Returns:
          dict[str, str]: VNC_UPSCALE_* variables.
        """
        width, height = (round(side * self.scale) for side in GAME_RES)
        return {
            "VNC_UPSCALE_SIZE": f"{width}x{height}",
            "VNC_UPSCALE_FILTER": self.filter,
            "VNC_UPSCALE_SHARPNESS": str(self.sharpness),
            "VNC_UPSCALE_WINDOW": self.window,
            "VNC_UPSCALE_BACKEND": "" if self.backend == "default" else self.backend,
        }


def build_scbw_args(
    bots: Sequence[str],
    map_name: str,
    speed: int,
    timeout: int,
    show_all: bool,
    auto_launch: bool,
    image: str,
    extra: Sequence[str],
) -> list[str]:
    """
    Build the `scbw.play` command line for a watchable headful game.

    Args:
      bots (Sequence[str]): Bot names, as stored in ~/.scbw/bots.
      map_name (str): Map path relative to ~/.scbw/maps.
      speed (int): Milliseconds per game frame (42 = BW Fastest).
      timeout (int): Real-time limit in seconds.
      show_all (bool): Open a VNC window per player, not just the host.
      auto_launch (bool): Auto-create the lobby via xdotool in the host container.
      image (str): Docker image for the game containers.
      extra (Sequence[str]): Extra args passed verbatim to `scbw.play`.

    Returns:
      list[str]: Full argv, starting with `scbw.play`.
    """
    args = ["scbw.play", "--bots", *bots, "--map", map_name]
    args += ["--game_speed", str(speed), "--timeout", str(timeout)]
    args += ["--docker_image", image]
    if show_all:
        args.append("--show_all")
    if auto_launch:
        args.append("--auto_launch")
    return args + list(extra)


def viewer_env(upscale: Upscale | None) -> dict[str, str]:
    """
    Environment with the `vnc-viewer` shim on PATH and upscale settings.

    Args:
      upscale (Upscale | None): Viewer upscaling, None for native size.

    Returns:
      dict[str, str]: Copy of os.environ with updated PATH.
    """
    env = dict(os.environ)
    env["PATH"] = f"{SCRIPTS_DIR}{os.pathsep}{env.get('PATH', '')}"
    if upscale is not None:
        env.update(upscale.to_env())
    return env


def ensure_image(image: str, dockerfile: Path) -> None:
    """
    Build (or refresh, cached) the game image from its Dockerfile.

    Args:
      image (str): Image tag, e.g. starcraft:watch.
      dockerfile (Path): Dockerfile that builds it.
    """
    build = ["docker", "build", "-q", "-t", image, "-f", str(dockerfile)]
    subprocess.run(
        [*build, str(dockerfile.parent)], check=True, stdout=subprocess.DEVNULL
    )


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    """
    Parse watch options; leftovers go to `scbw.play`.

    Args:
      argv (Sequence[str] | None): Arguments, defaults to sys.argv[1:].

    Returns:
      tuple[argparse.Namespace, list[str]]: Known options and pass-through args.
    """
    parser = argparse.ArgumentParser(
        description="Watch a live bot-vs-bot game over VNC."
    )
    parser.add_argument("--bots", nargs="+", default=DEFAULT_BOTS)
    parser.add_argument("--map", dest="map_name", default=DEFAULT_MAP)
    parser.add_argument(
        "--speed",
        type=int,
        default=FASTEST_SPEED,
        help="ms per frame (higher = slower)",
    )
    parser.add_argument("--timeout", type=int, default=3600, help="real-time seconds")
    parser.add_argument(
        "--show_all", action="store_true", help="VNC window for every player"
    )
    parser.add_argument(
        "--manual_lobby",
        action="store_true",
        help="Don't auto-create the lobby; do it yourself in the VNC window",
    )
    parser.add_argument(
        "--scale", type=float, default=4.0, help="viewer upscale factor, 0 = off"
    )
    parser.add_argument("--filter", choices=FILTERS, default="fsr")
    parser.add_argument(
        "--sharpness", type=int, default=2, help="FSR/NIS, 0 (max) to 20 (min)"
    )
    parser.add_argument(
        "--window",
        choices=WINDOW_MODES,
        default="borderless",
        help="borderless avoids fullscreen flicker (NVIDIA + GNOME)",
    )
    parser.add_argument("--backend", choices=BACKENDS, default="default")
    return parser.parse_known_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Launch a headful game and block until it ends.

    Args:
      argv (Sequence[str] | None): Arguments, defaults to sys.argv[1:].

    Returns:
      int: `scbw.play` exit code.
    """
    opts, extra = parse_args(argv)
    upscale = (
        Upscale(opts.scale, opts.filter, opts.sharpness, opts.window, opts.backend)
        if opts.scale > 0
        else None
    )
    env = viewer_env(upscale)
    if upscale is not None and shutil.which("gamescope") is None:
        print(
            "No gamescope (sudo apt install gamescope) or use --scale 0",
            file=sys.stderr,
        )
        return 1
    if shutil.which("xtigervncviewer", path=env["PATH"]) is None and (
        shutil.which("vncviewer", path=env["PATH"]) is None
    ):
        print("No VNC viewer. Run: sudo apt install tigervnc-viewer", file=sys.stderr)
        return 1
    ensure_image(WATCH_IMAGE, WATCH_DOCKERFILE)
    cmd = build_scbw_args(
        bots=opts.bots,
        map_name=opts.map_name,
        speed=opts.speed,
        timeout=opts.timeout,
        show_all=opts.show_all,
        auto_launch=not opts.manual_lobby,
        image=WATCH_IMAGE,
        extra=extra,
    )
    print("Running:", " ".join(cmd), file=sys.stderr)
    try:
        return subprocess.run(cmd, env=env).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
