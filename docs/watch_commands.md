# Watch / play commands (4K, NVIDIA + GNOME Wayland)

## Watch a bot game

Tested and working (defaults: `--scale 4` → 2560x1920, `--window borderless`, FSR):

```bash
uv run python -m starcraft_diffusion_gemma.watch
# explicit equivalent
uv run python -m starcraft_diffusion_gemma.watch --scale 4 --window borderless
```

Variants:

```bash
# other bots / slower game (ms per frame, 42 = Fastest)
uv run python -m starcraft_diffusion_gemma.watch --bots Stardust "Dave Churchill" --speed 60
# crisp pixels instead of FSR
uv run python -m starcraft_diffusion_gemma.watch --filter nearest
# sharper FSR (0 = max, 20 = min)
uv run python -m starcraft_diffusion_gemma.watch --sharpness 0
# plain 640x480 viewer, no gamescope
uv run python -m starcraft_diffusion_gemma.watch --scale 0
```

Notes:

* `--scale 4.5` (2880x2160) gets cut off by the GNOME top bar; use 4.
* `--window fullscreen` flickers on this setup; stay on borderless.
  If needed, try `--backend sdl`.
* In-window hotkeys: Super+N toggles nearest-neighbour, Super+I / Super+O change
  FSR sharpness.

## Play yourself through gamescope

Launcher: `~/wine/starcraft/sc.sh`.

```bash
~/wine/starcraft/sc.sh                    # 2560x1920 window
~/wine/starcraft/sc.sh full               # borderless 4K, no flicker
SC_FILTER=fsr SC_SCALER=fit ~/wine/starcraft/sc.sh full  # FSR, fill height
~/wine/starcraft/sc.sh fullsdl            # true fullscreen, SDL backend (test)
```

Generic form:

Same fix: borderless (`-b`) instead of fullscreen (`-f`). `-w/-h` is the game's
render resolution, `-W/-H` is the output window.

```bash
# Original Brood War (640x480) → 2560x1920 FSR, borderless
gamescope -w 640 -h 480 -W 2560 -H 1920 -b -S fit -F fsr --sharpness 2 -- <game command>

# Higher-res game (e.g. Remastered at 1080p) → 4K-height FSR, borderless
gamescope -w 1440 -h 1080 -W 2560 -H 1920 -b -S fit -F fsr --sharpness 2 -- <game command>
```

`<game command>` is how you launch the game, e.g. `wine StarCraft.exe`,
`lutris lutris:rungame/<slug>`, or in Steam launch options:
`gamescope -w 640 -h 480 -W 2560 -H 1920 -b -S fit -F fsr -- %command%`.

If borderless still flickers, add `--backend sdl`, then `--force-composition`.
