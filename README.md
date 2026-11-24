# starcraft-diffusion-gemma

StarCraft: Brood War bot using DiffusionGemma (local vLLM) for high-level planning.

- How to serve the model and use images / tool calls / structured output / decisions:
  [docs/diffusion_gemma.md](docs/diffusion_gemma.md)
- Client: `starcraft_diffusion_gemma/llm/client.py`
- Capability check: `uv run python -m starcraft_diffusion_gemma.llm.demo`

## Watch a bot game live

Headful game via [sc-docker](../sc-docker) (`scbw`), streamed over VNC.

```bash
sudo apt install tigervnc-viewer          # one-time
uv run python -m starcraft_diffusion_gemma.watch
uv run python -m starcraft_diffusion_gemma.watch --bots Stardust "Dave Churchill" --speed 60 --show_all
```

* Defaults: Stardust vs Iron bot on `sscai/(2)Benzene.scx`, `--speed 42` (ms/frame, BW
  "Fastest"; higher = slower). Bots live in `~/.scbw/bots`.
* `scripts/vnc-viewer` is a shim: scbw requires a binary with that exact name. It runs
  the viewer inside [gamescope](https://github.com/ValveSoftware/gamescope) at native
  640x480 and upscales the window: `--scale 4` (default, 2560x1920), `--filter fsr`
  (or `nis`, `linear`, `nearest`, `pixel`), `--sharpness 2` (0 = max),
  `--window windowed|borderless|fullscreen`, `--backend default|sdl|wayland`. Fullscreen flickers on NVIDIA + GNOME Wayland;
  borderless is the default (or try `--backend sdl`). Command cheat sheet:
  [docs/watch_commands.md](docs/watch_commands.md).
  `--scale 0` = plain viewer. In-window: Super+N toggles nearest, Super+I/O sharpness.
* Games run in `starcraft:watch` (`docker/watch.dockerfile`, auto-built): `starcraft:game`
  + xdotool, and the auto-lobby waits 20 s (not 3 s) for the opponent before Start —
  otherwise the game starts 1-player and bots crash on `enemy() is NULL`.
* The lobby is auto-created by xdotool keypresses (`--auto_launch`). If that misses,
  pass `--manual_lobby` and in the VNC window: Multiplayer → Local PC → Create Game →
  pick the map → OK. The other bot joins automatically.
* Unknown args go straight to `scbw.play`. Logs/replays: `~/.scbw/games/GAME_*`.
