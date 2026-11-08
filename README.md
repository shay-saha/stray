# Stray

Stray combines a native wildlife migration atlas with WildCast's movement-model research and inference code.

| Folder | Purpose |
| --- | --- |
| [htb-gui](htb-gui/README.md) | C++20/OpenGL Earth viewer, glowing particles, forecast playback and a local CUDA inference service |
| [htb](htb/README.md) | WildCast models, observation preparation, training, evaluation and packaging tools |

The folders remain siblings because the viewer uses the model project's Python environment at `../htb/.venv`.

## Run the atlas

Create the model environment and install its Python dependencies:

```sh
python3 -m venv htb/.venv
htb/.venv/bin/python -m pip install -e './htb[dev]'
cd htb-gui
./run-local
```

Install a PyTorch build suitable for the machine's CUDA/GPU configuration. The viewer also needs CMake, a C++20 compiler, OpenGL, libcurl and nlohmann-json; GLFW and ImGui source archives are bundled. Detailed native package instructions are in [the viewer README](htb-gui/README.md).

Supply the verified local model deployment at `htb-gui/models/server-repaired-v2`. It contains the trained checkpoint, matching feature/model implementation and actual GPS observation archive. The inference service verifies that package at startup. These research assets are stored outside Git and must be copied separately when setting up another machine.

## Play the completed 50-year scenario

Keep `forecast.json` and `positions.f32` together under `htb-gui/runs/server-50y-20261108`, then run:

```sh
cd htb-gui
ATLAS_FORECAST=runs/server-50y-20261108/forecast.json ./run-local --no-build
```

Build once with `./run-local` before using `--no-build`. Press **Play** and select a playback speed. Imported playback uses the stored model trajectories without requesting new inference. Run `./run-local` without `ATLAS_FORECAST` for continuous local GPU inference.

The demo covers 2026–2076 for 6,834 recorded identities across 118 taxon labels and 165 study/species cohorts. It is a recursive model scenario, not a validated 50-year ecological forecast. Study/species cohorts do not establish biological colony membership. [Model and data provenance](htb-gui/docs/model-deployment.md)

## Trails and map controls

Trail history ranges from one hour through 50 years, with intermediate daily, weekly, monthly and yearly options. Long windows use bounded, fading grains along the actual observation/model timelines. The particle display keeps one marker per recorded animal. The Earth camera stops at city/state scale and retains a modest terrain tilt.

## Checks

```sh
cmake -S htb-gui -B htb-gui/build -DCMAKE_BUILD_TYPE=Release
cmake --build htb-gui/build --parallel 6
ctest --test-dir htb-gui/build --output-on-failure
cd htb
.venv/bin/python -m pytest tests/test_migration_repaired_features.py tests/test_migration_protocol.py tests/test_migration_model.py
```

The viewer Python integration/rendering tests require the separately supplied model assets and saved forecasts. See each project's README for its complete validation workflow.

## Local research assets

Git tracks source, configuration, documentation, tests, dependency licenses and the atlas's bundled map textures. Training datasets, checkpoints, generated reports, forecast archives, virtual environments, logs and credentials belong in local storage. The `.gitignore` rules cover both projects. On the original development machine, ignored symlinks in this checkout reuse the existing runtime assets without duplicating them.

Upstream model and renderer dependencies retain their original license notices. See `htb/third_party` and [the renderer dependency notes](htb-gui/third_party/README.md).
