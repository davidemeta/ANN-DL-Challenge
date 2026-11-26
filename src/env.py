"""Platform detection: the same code runs on a local Mac, Google Colab and Kaggle.

    from env import ENV
    ENV.data_dir    # folder with train/, test/, train.csv, sample_submission.csv
    ENV.out_dir     # checkpoints, logits, submissions (persistent where possible)
    ENV.cache_dir   # fast local disk for the decoded .npy cache (rebuildable, not persistent)
    ENV.device      # torch.device: cuda > mps > cpu
    ENV.num_workers # DataLoader workers suited to the machine

Every path can be overridden with an environment variable:
AN2DL_DATA_DIR, AN2DL_OUT_DIR, AN2DL_CACHE_DIR.

Data per platform:
- local:  <repo>/data (symlink to the kagglehub cache or a manual download)
- kaggle: the competition attached to the notebook, found under /kaggle/input
- colab:  downloaded with kagglehub on first use; the token is read from the Colab secret
          KAGGLE_API_TOKEN (Secrets panel, notebook access on). It is never printed.
"""
import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path

COMPETITION = 'an2dl2627'
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = ('train.csv', 'sample_submission.csv')


def detect_platform():
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') or os.path.isdir('/kaggle/input'):
        return 'kaggle'
    if 'google.colab' in sys.modules or os.environ.get('COLAB_RELEASE_TAG'):
        return 'colab'
    return 'local'


def _is_data_dir(path):
    return path is not None and all(os.path.isfile(os.path.join(path, f)) for f in DATA_FILES)


def _find_kaggle_input():
    for csv in sorted(glob.glob('/kaggle/input/**/sample_submission.csv', recursive=True)):
        folder = os.path.dirname(csv)
        if _is_data_dir(folder):
            return folder
    return None


def _colab_download():
    """Download the competition with kagglehub (cached in ~/.cache/kagglehub for the session)."""
    if not os.environ.get('KAGGLE_API_TOKEN'):
        try:
            from google.colab import userdata
            os.environ['KAGGLE_API_TOKEN'] = userdata.get('KAGGLE_API_TOKEN')
        except Exception as e:
            raise RuntimeError(
                'No Kaggle credentials on Colab: add the secret KAGGLE_API_TOKEN in the Secrets '
                'panel (key icon) and enable notebook access, or set AN2DL_DATA_DIR.') from e
    import kagglehub
    return kagglehub.competition_download(COMPETITION)


def _colab_drive_root():
    drive = '/content/drive/MyDrive'
    return os.path.join(drive, 'an2dl') if os.path.isdir(drive) else None


def resolve_data_dir(platform):
    override = os.environ.get('AN2DL_DATA_DIR')
    if override:
        if not _is_data_dir(override):
            raise FileNotFoundError(f'AN2DL_DATA_DIR={override} has no {DATA_FILES}')
        return override
    if platform == 'kaggle':
        found = _find_kaggle_input()
        if found:
            return found
        raise FileNotFoundError('Competition data not found under /kaggle/input: '
                                'add the competition with "Add Input" in the notebook sidebar.')
    if platform == 'colab':
        return _colab_download()
    local = str(REPO_ROOT / 'data')
    if _is_data_dir(local):
        return local
    raise FileNotFoundError(f'No data at {local}: create the symlink or set AN2DL_DATA_DIR.')


def resolve_out_dir(platform):
    override = os.environ.get('AN2DL_OUT_DIR')
    if override:
        return override
    if platform == 'kaggle':
        return '/kaggle/working/outputs'       # saved with the notebook version
    if platform == 'colab':
        drive = _colab_drive_root()           # mount Drive first so runs survive a disconnect
        return os.path.join(drive, 'outputs') if drive else '/content/outputs'
    return str(REPO_ROOT / 'outputs')          # gitignored


def resolve_cache_dir(platform):
    override = os.environ.get('AN2DL_CACHE_DIR')
    if override:
        return override
    if platform == 'kaggle':
        return '/kaggle/temp/cache'            # not saved with the version (test cache is ~2.9 GB)
    if platform == 'colab':
        return '/content/cache'                # local SSD, much faster than Drive
    return str(REPO_ROOT / 'outputs' / 'cache')


def pick_device():
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def default_num_workers(platform):
    n = os.cpu_count() or 2
    if platform == 'local':
        return min(4, n)    # macOS spawns workers: few and persistent
    return max(1, min(8, n))


@dataclass
class Env:
    platform: str
    data_dir: str
    out_dir: str
    cache_dir: str
    num_workers: int

    @property
    def device(self):
        return pick_device()

    def describe(self):
        import torch
        dev = self.device
        gpu = torch.cuda.get_device_name(0) if dev.type == 'cuda' else dev.type
        n_gpu = torch.cuda.device_count() if dev.type == 'cuda' else 0
        return (f'platform={self.platform} device={gpu} (x{max(n_gpu, 1)}) '
                f'workers={self.num_workers}\n  data={self.data_dir}\n  out={self.out_dir}\n'
                f'  cache={self.cache_dir}')


def get_env():
    platform = detect_platform()
    env = Env(platform=platform,
              data_dir=resolve_data_dir(platform),
              out_dir=resolve_out_dir(platform),
              cache_dir=resolve_cache_dir(platform),
              num_workers=default_num_workers(platform))
    os.makedirs(env.out_dir, exist_ok=True)
    os.makedirs(env.cache_dir, exist_ok=True)
    return env


class _LazyEnv:
    """Resolve paths on first attribute access, so importing this module never downloads data."""
    _env = None

    def __getattr__(self, name):
        if _LazyEnv._env is None:
            _LazyEnv._env = get_env()
        return getattr(_LazyEnv._env, name)


ENV = _LazyEnv()

if __name__ == '__main__':
    print(get_env().describe())
