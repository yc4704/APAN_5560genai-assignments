# Assignment 4: Advanced Image Generation

This implementation follows the two Module 8 practicals. It trains an energy
model and a diffusion model on CIFAR-10 and adds both generators to the
existing FastAPI application. The theory answers are separate.

## Run the notebooks

Open the notebooks from the repository root:

1. `Assignment4_CIFAR10_Energy.ipynb`
2. `Assignment4_CIFAR10_Diffusion.ipynb`

Each notebook includes the complete model definitions, data loading,
training, sample grids, checkpoints, reload verification, and API export.
Code cells contain no comments. English Markdown cells explain each step.

Use a Python environment containing the project dependencies, Matplotlib,
tqdm, and Jupyter. For example, from this directory:

```powershell
uv run --with jupyterlab --with matplotlib --with tqdm jupyter lab
```

Select the kernel associated with that environment. Run cells from top to
bottom. CIFAR-10 is downloaded only if it is not already present. Both
notebooks use RGB 32 x 32 images, with a 45,000/5,000 train/validation split
and the separate 10,000-image test split. The model does not condition on class labels.

The default ten epochs is an initial experiment setting, not a required
minimum or a claim of sufficient training. A CPU can execute the code but
full training can take a long time. A CUDA-enabled PyTorch environment is
used automatically when available. Inspect the generated grids before
deciding whether to train longer. Do not present smoke-test outputs as
trained model quality.

## Classroom methods and assignment adaptations

| Component | Retained method | Necessary adaptation |
| --- | --- | --- |
| Energy network | Four stride-2 convolutions, Swish, two linear layers, scalar output | Three input channels for CIFAR-10 |
| Energy sampling | Noise injection, input gradients, gradient clipping, pixel clipping | Preserve model mode and explicitly enable input gradients |
| Replay buffer | Approximately 5% new noise, capacity 8,192 | Use the current batch size and handle an all-new batch |
| Energy objective | Real energy minus generated energy, plus squared-energy regularization | Add a complete training loop and example-weighted metrics |
| Diffusion embedding | Classroom frequencies 1 to 1,000, sine/cosine concatenation | Keep this practical convention distinct from the theory question |
| UNet | Residual blocks, average-pooling down blocks, bilinear up blocks, skip connections | Three channels, 32 x 32 inputs, omit unused layers |
| Diffusion training | Offset cosine schedule, noise prediction, EMA decay 0.8 | Calculate training-only RGB normalization and use MSE explicitly |
| Deployment | Existing FastAPI, PNG responses, Docker | Lazy-load new checkpoints and return 503 when weights are absent |

The EBM learning rate, regularization coefficient, and sampling settings are
explicit initial choices because the supplied HTML does not include a
complete run configuration. Diffusion uses the classroom offset cosine
schedule instead of applying the linear schedule's cumulative product to
an unordered batch of random times. No pretrained external generator is used.

## Outputs and resuming

Each fresh run creates a timestamped directory under `../ASS4/outputs/energy_*`
or `../ASS4/outputs/diffusion_*` containing:

- `history.json`: epoch metrics.
- `samples/epoch_*.png`: generated images for each epoch.
- `checkpoints/energy_latest.pth` or `diffusion_latest.pth`: resumable state.
- `checkpoints/diffusion_best.pth`: lowest observed validation noise MSE.

The two supplementary notebooks use configurable `RESUME_PATH` values.
Relative paths are resolved from the repository root, regardless of its
name or location. Defaults are `checkpoints/assignment4/energy_epoch_010.pth`
and `checkpoints/assignment4/diffusion_epoch_010.pth`. These full training
states are not included in Git; the smaller inference weights are included.

To recreate the continuation inputs on a fresh clone:

1. Run the corresponding main notebook through 10 epochs with `RESUME_PATH = None`.
2. Before any further training, copy its latest training checkpoint to the
   supplementary notebook's default location. In a new code cell in that
   completed main notebook, run the appropriate block below.

Energy:

```python
import shutil
resume_copy = PROJECT_DIR / "checkpoints" / "assignment4" / "energy_epoch_010.pth"
resume_copy.parent.mkdir(parents=True, exist_ok=True)
if resume_copy.exists():
    raise FileExistsError(f"Choose a new destination to preserve {resume_copy}")
shutil.copy2(CHECKPOINT_DIR / "energy_latest.pth", resume_copy)
```

Diffusion:

```python
import shutil
resume_copy = PROJECT_DIR / "checkpoints" / "assignment4" / "diffusion_epoch_010.pth"
resume_copy.parent.mkdir(parents=True, exist_ok=True)
if resume_copy.exists():
    raise FileExistsError(f"Choose a new destination to preserve {resume_copy}")
shutil.copy2(CHECKPOINT_DIR / "diffusion_latest.pth", resume_copy)
```

Alternatively set `RESUME_PATH` in the supplement directly to that main
run's latest training checkpoint. It must be an epoch-10 training state,
not `models/cifar10_*.pth`. The restore cell checks the epoch. Energy restores
its optimizer, replay buffer, fixed noise, and history; Diffusion restores
both networks, optimizer, normalization, fixed noise, and history.
Both supplements create new timestamped run directories; they leave the
input checkpoint and deployed API weights intact. Saved notebook outputs
are historical and can contain paths from the original execution. Random
states are not fully restored, so reruns are not bit-for-bit reproductions.

The included API inference weights are:

- `models/cifar10_energy.pth`: main Energy experiment, epoch 10.
- `models/cifar10_diffusion.pth`: main Diffusion experiment, selected epoch 10.

Energy exports the final epoch. Diffusion exports selected EMA weights and
normalization statistics. The main Diffusion notebook first exports into
its run directory, then installs the selected weights into `models/` with
a backup of any previous file. Energy exports directly into `models/`.
Restart the server after replacing weights because models are cached.
Neither API silently falls back to an untrained generator.

## API and Docker

The existing Dockerfile already copies `app`, `helper_lib`, and `models`.
After both exports exist, build from the repository root:

```powershell
docker build -t genai-ass4 .
docker run --rm --name genai-ass4 -p 127.0.0.1:8001:80 genai-ass4
```

Open `http://127.0.0.1:8001/docs` or save generated images:

```powershell
curl.exe --fail "http://127.0.0.1:8001/generate-energy?num_samples=4&steps=60" --output energy_samples.png
curl.exe --fail "http://127.0.0.1:8001/generate-diffusion?num_samples=4&steps=20" --output diffusion_samples.png
```

Both endpoints return PNG grids. `num_samples` accepts 1-16 and `steps`
accepts 1-200. Invalid parameters return HTTP 422. Missing model weights
return HTTP 503 with the required notebook/export step. For initial CPU
testing, request one image with two steps; this checks execution only.
More sampling steps increase inference time and do not replace training.

## Tests

From the repository root, use the project environment:

```powershell
uv run python -m unittest discover -s tests -v
```

Tests check input-gradient sampling, weight updates, variable-sized replay
batches, schedule endpoints, EMA updates, checkpoint loading, PNG output,
API query validation, missing weights, and basic existing-route behavior.
Temporary test weights do not replace exported assignment models.

Verification on November 18, 2026: all 12 tests passed in the project Python
environment. The included Energy and Diffusion epoch-10 weights returned
HTTP 200 and valid 32 x 32 RGB PNGs through FastAPI TestClient at their default
60 and 20 sampling steps respectively (one image per request).

The model/unit tests use temporary synthetic checkpoints to check code behavior; they
do not replace the included trained weights. The included epoch-10 models
are also checked separately through the FastAPI routes with default sampling
steps. These checks validate execution and PNG responses, not image realism.
Docker instructions are provided above; a local API check alone is not a
Docker execution test.

## Submission contents

The repository includes four executed notebooks, both reusable model
modules, API services and routes, selected inference weights, and tests.
The theory answers are submitted separately. Full training checkpoints and
downloaded datasets remain local; the procedure above recreates the inputs
needed for optional continuation experiments.
