import os
import os.path as osp

# Prevent numpy over multithreading
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from diffusion_lib.denoising_diffusion_pytorch_1d import GaussianDiffusion1D, Trainer1D
from models import EBM, DiffusionWrapper
from models import SudokuEBM, SudokuDenoise, SudokuLatentEBM, AutoencodeModel
from models import GraphEBM, GraphReverse, GNNConvEBM, GNNDiffusionWrapper, GNNConvDiffusionWrapper, GNNConv1DEBMV2, GNNConv1DV2DiffusionWrapper, GNNConv1DReverse
from dataset import Addition, LowRankDataset, Inverse
from reasoning_dataset import FamilyTreeDataset, GraphConnectivityDataset, FamilyDatasetWrapper, GraphDatasetWrapper
from planning_dataset import PlanningDataset, PlanningDatasetOnline
from sat_dataset import SATNetDataset, SudokuDataset, SudokuRRNDataset, SudokuRRNLatentDataset
import torch

# Import curriculum configuration with error handling
try:
    from curriculum_config import get_curriculum_by_name, DEFAULT_CURRICULUM
    CURRICULUM_AVAILABLE = True
except ImportError:
    print("Warning: curriculum_config module not found. Curriculum features disabled.")
    CURRICULUM_AVAILABLE = False

import argparse

try:
    import mkl
    mkl.set_num_threads(1)
except ImportError:
    print('Warning: MKL not initialized.')


def str2bool(x):
    if isinstance(x, bool):
        return x
    x = x.lower()
    if x[0] in ['0', 'n', 'f']:
        return False
    elif x[0] in ['1', 'y', 't']:
        return True
    raise ValueError('Invalid value: {}'.format(x))


parser = argparse.ArgumentParser(description='Train Diffusion Reasoning Model')

parser.add_argument('--dataset', default='inverse', type=str, help='dataset to evaluate')
parser.add_argument('--inspect-dataset', action='store_true', help='run an IPython embed interface after loading the dataset')
parser.add_argument('--model', default='mlp', type=str, choices=['mlp', 'mlp-reverse', 'sudoku', 'sudoku-latent', 'sudoku-reverse', 'gnn', 'gnn-reverse', 'gnn-conv', 'gnn-conv-1d', 'gnn-conv-1d-v2', 'gnn-conv-1d-v2-reverse'])
parser.add_argument('--load-milestone', type=str, default=None, help='load a model from a milestone')
parser.add_argument('--batch_size', default=2048, type=int, help='size of batch of input to use')
parser.add_argument('--diffusion_steps', default=10, type=int, help='number of diffusion time steps (default: 10)')
parser.add_argument('--rank', default=20, type=int, help='rank of matrix to use')
parser.add_argument('--data-workers', type=int, default=None, help='number of workers to use for data loading')
parser.add_argument('--supervise-energy-landscape', type=str2bool, default=False)
parser.add_argument('--use-innerloop-opt', type=str2bool, default=False)
parser.add_argument('--cond_mask', type=str2bool, default=False)
parser.add_argument('--evaluate', action='store_true', default=False)
parser.add_argument('--latent', action='store_true', default=False)
parser.add_argument('--ood', action='store_true', default=False)
parser.add_argument('--baseline', action='store_true', default=False)
# CSV logging arguments
parser.add_argument('--save-csv-logs', action='store_true', default=False,
                   help='Save training and validation metrics to CSV files')
parser.add_argument('--csv-log-interval', type=int, default=100,
                   help='Interval for logging training metrics to CSV')
parser.add_argument('--csv-log-dir', type=str, default='./csv_logs',
                   help='Directory to save CSV log files')
                   
parser.add_argument('--train-num-steps', type=int, default=1000,
                   help='Total number of training steps')

# Curriculum configuration arguments
parser.add_argument('--curriculum-config', type=str, default='default',
                   choices=['default', 'aggressive', 'conservative', 'none'],
                   help='Choice of curriculum configuration (default: default)')
parser.add_argument('--disable-curriculum', type=str2bool, default=False,
                   help='Boolean to force legacy behavior (default: False)')


if __name__ == "__main__":
    FLAGS = parser.parse_args()

    validation_dataset = None
    extra_validation_datasets = dict()
    extra_validation_every_mul = 10
    save_and_sample_every = 1000
    validation_batch_size = 256

    if FLAGS.dataset == "addition":
        dataset = Addition("train", FLAGS.rank, FLAGS.ood)
        validation_dataset = dataset
        metric = 'mse'
    elif FLAGS.dataset == "inverse":
        dataset = Inverse("train", FLAGS.rank, FLAGS.ood)
        validation_dataset = Inverse("val", FLAGS.rank, FLAGS.ood)
        metric = 'mse'
        save_and_sample_every = 100  # Lower interval for short training runs
    elif FLAGS.dataset == "lowrank":
        dataset = LowRankDataset("train", FLAGS.rank, FLAGS.ood)
        validation_dataset = dataset
        metric = 'mse'
    elif FLAGS.dataset == 'parents':
        dataset = FamilyDatasetWrapper(FamilyTreeDataset((12, 12), epoch_size=int(1e5), task='parents'))
        metric = 'bce'
    elif FLAGS.dataset == 'uncle':
        dataset = FamilyDatasetWrapper(FamilyTreeDataset((12, 12), epoch_size=int(1e5), task='uncle'))
        metric = 'bce'
    elif FLAGS.dataset == 'connectivity':
        dataset = GraphDatasetWrapper(GraphConnectivityDataset((12, 12), 0.1, epoch_size=int(2048 * 1000), gen_method='dnc'))
        extra_validation_datasets = {
            'connectivity-13': GraphDatasetWrapper(GraphConnectivityDataset((13, 13), 0.1, epoch_size=int(1e3), gen_method='dnc')),
            'connectivity-15': GraphDatasetWrapper(GraphConnectivityDataset((15, 15), 0.1, epoch_size=int(1e3), gen_method='dnc')),
            'connectivity-18': GraphDatasetWrapper(GraphConnectivityDataset((18, 18), 0.1, epoch_size=int(1e3), gen_method='dnc')),
            'connectivity-20': GraphDatasetWrapper(GraphConnectivityDataset((20, 20), 0.1, epoch_size=int(1e3), gen_method='dnc'))
        }
        validation_batch_size = 64
        metric = 'bce'
    elif FLAGS.dataset == 'connectivity-2':
        dataset = GraphDatasetWrapper(GraphConnectivityDataset((12, 12), 0.2, epoch_size=int(2048 * 1000), gen_method='dnc'))
        extra_validation_datasets = {
            'connectivity-13': GraphDatasetWrapper(GraphConnectivityDataset((13, 13), 0.2, epoch_size=int(1e3), gen_method='dnc')),
            'connectivity-15': GraphDatasetWrapper(GraphConnectivityDataset((15, 15), 0.2, epoch_size=int(1e3), gen_method='dnc')),
            'connectivity-18': GraphDatasetWrapper(GraphConnectivityDataset((18, 18), 0.2, epoch_size=int(1e3), gen_method='dnc')),
            'connectivity-20': GraphDatasetWrapper(GraphConnectivityDataset((20, 20), 0.1, epoch_size=int(1e3), gen_method='dnc'))
        }
        validation_batch_size = 64
        metric = 'bce'
    elif FLAGS.dataset.startswith('parity'):
        dataset = SATNetDataset(FLAGS.dataset)
        metric = 'bce'
    elif FLAGS.dataset == 'sudoku':
        train_dataset = SudokuDataset(FLAGS.dataset, split='train')
        validation_dataset = SudokuDataset(FLAGS.dataset, split='val')
        extra_validation_datasets = {'sudoku-rrn-test': SudokuRRNDataset('sudoku-rrn', split='test')}
        dataset = train_dataset
        metric = 'sudoku'
        validation_batch_size = 64
        assert FLAGS.cond_mask
    elif FLAGS.dataset == 'sudoku-rrn':
        train_dataset = SudokuRRNDataset(FLAGS.dataset, split='train')
        validation_dataset = SudokuRRNDataset(FLAGS.dataset, split='test')
        save_and_sample_every = 10000
        dataset = train_dataset
        metric = 'sudoku'
        assert FLAGS.cond_mask
    elif FLAGS.dataset == 'sudoku-rrn-latent':
        train_dataset = SudokuRRNLatentDataset(FLAGS.dataset, split='train')
        validation_dataset = SudokuRRNLatentDataset(FLAGS.dataset, split='validation')
        save_and_sample_every = 10000
        dataset = train_dataset
        metric = 'sudoku_latent'
    elif FLAGS.dataset == 'sort':
        train_dataset = PlanningDataset(FLAGS.dataset, split='train', num_identifier=100000)
        validation_dataset = PlanningDataset(FLAGS.dataset, split='validation', num_identifier=100000)
        extra_validation_datasets = {
            # it's fine to keep split=train because this is a generalization dataset.
            'sort-15': PlanningDataset(FLAGS.dataset + '-15', split='train', num_identifier=10000)
        }
        dataset = train_dataset
        metric = 'sort'
    elif FLAGS.dataset == 'sort-2':
        train_dataset = PlanningDatasetOnline('list-sorting-2', n=10)
        validation_dataset = PlanningDatasetOnline('list-sorting-2', n=10)
        extra_validation_datasets = {
            'sort-15': PlanningDatasetOnline('list-sorting-2', n=15)
        }
        dataset = train_dataset
        metric = 'sort-2'
    elif FLAGS.dataset == 'shortest-path':
        train_dataset = PlanningDataset(FLAGS.dataset, split='train', num_identifier=10000)
        validation_dataset = PlanningDataset(FLAGS.dataset, split='validation', num_identifier=10000)
        dataset = train_dataset
        metric = 'bce'
    elif FLAGS.dataset == 'shortest-path-1d':
        train_dataset = PlanningDataset(FLAGS.dataset, split='train', num_identifier=100000)
        validation_dataset = PlanningDataset(FLAGS.dataset, split='validation', num_identifier=100000)
        extra_validation_datasets = {
            # it's fine to keep split=train because this is a generalization dataset.
            'shortest-path-25': PlanningDataset('shortest-path-25-1d', split='train', num_identifier=10000)
        }
        dataset = train_dataset
        metric = 'shortest-path-1d'
        validation_batch_size = 64
    elif FLAGS.dataset == 'shortest-path-10-1d':
        train_dataset = PlanningDataset(FLAGS.dataset, split='train', num_identifier=100000)
        validation_dataset = PlanningDataset(FLAGS.dataset, split='validation', num_identifier=100000)
        extra_validation_datasets = {
            # it's fine to keep split=train because this is a generalization dataset.
            'shortest-path-15': PlanningDataset('shortest-path-15-1d', split='train', num_identifier=10000)
        }
        dataset = train_dataset
        metric = 'shortest-path-1d'
        validation_batch_size = 128
    elif FLAGS.dataset == 'shortest-path-15-1d':
        train_dataset = PlanningDataset(FLAGS.dataset, split='train', num_identifier=100000)
        validation_dataset = PlanningDataset(FLAGS.dataset, split='validation', num_identifier=100000)
        extra_validation_datasets = {
            # it's fine to keep split=train because this is a generalization dataset.
            'shortest-path-20': PlanningDataset('shortest-path-1d', split='train', num_identifier=10000)
        }
        dataset = train_dataset
        metric = 'shortest-path-1d'
        validation_batch_size = 128
    else:
        assert False

    if FLAGS.inspect_dataset:
        from IPython import embed
        embed()
        exit()

    if FLAGS.model == 'mlp':
        model = EBM(
            inp_dim = dataset.inp_dim,
            out_dim = dataset.out_dim,
        )
        model = DiffusionWrapper(model)
    elif FLAGS.model == 'mlp-reverse':
        model = EBM(
            inp_dim = dataset.inp_dim,
            out_dim = dataset.out_dim,
            is_ebm = False,
        )
    elif FLAGS.model == 'sudoku':
        model = SudokuEBM(
            inp_dim = dataset.inp_dim,
            out_dim = dataset.out_dim,
        )
        model = DiffusionWrapper(model)
    elif FLAGS.model == 'sudoku-latent':
        model = SudokuLatentEBM(
            inp_dim = dataset.inp_dim,
            out_dim = dataset.out_dim,
        )
        model = DiffusionWrapper(model)
    elif FLAGS.model == 'sudoku-reverse':
        model = SudokuDenoise(
            inp_dim = dataset.inp_dim,
            out_dim = dataset.out_dim,
        )
    elif FLAGS.model == 'gnn':
        model = GraphEBM(
            inp_dim = dataset.inp_dim,
            out_dim = dataset.out_dim,
        )
        model = GNNDiffusionWrapper(model)
    elif FLAGS.model == 'gnn-reverse':
        model = GraphReverse(
            inp_dim = dataset.inp_dim,
            out_dim = dataset.out_dim,
        )
    elif FLAGS.model == 'gnn-conv':
        model = GNNConvEBM(inp_dim = dataset.inp_dim, out_dim = dataset.out_dim)
        model = GNNConvDiffusionWrapper(model)
    elif FLAGS.model == 'gnn-conv-1d':
        model = GNNConvEBM(inp_dim = dataset.inp_dim, out_dim = dataset.out_dim, use_1d = True)
        model = GNNConvDiffusionWrapper(model)
    elif FLAGS.model == 'gnn-conv-1d-v2':
        model = GNNConv1DEBMV2(inp_dim = dataset.inp_dim, out_dim = dataset.out_dim)
        model = GNNConv1DV2DiffusionWrapper(model)
    elif FLAGS.model == 'gnn-conv-1d-v2-reverse':
        model = GNNConv1DReverse(inp_dim = dataset.inp_dim, out_dim = dataset.out_dim)
    else:
        assert False

    kwargs = dict()
    if FLAGS.baseline:
        kwargs['baseline'] = True

    if FLAGS.dataset in ['addition', 'inverse', 'lowrank']:
        kwargs['continuous'] = True

    if FLAGS.dataset in ['sudoku', 'sudoku_latent', 'sudoku-rrn', 'sudoku-rrn-latent']:
        kwargs['sudoku'] = True

    if FLAGS.dataset in ['connectivity', 'connectivity-2']:
        kwargs['connectivity'] = True

    if FLAGS.dataset in ['shortest-path', 'shortest-path-1d']:
        kwargs['shortest_path'] = True

    # Configure curriculum
    curriculum_config = None
    if CURRICULUM_AVAILABLE and not FLAGS.disable_curriculum and FLAGS.curriculum_config != 'none':
        try:
            curriculum_config = get_curriculum_by_name(FLAGS.curriculum_config)
            # Update curriculum with actual training steps
            curriculum_config.total_steps = FLAGS.train_num_steps
            print(f"Using curriculum: {FLAGS.curriculum_config}")
            print(f"Curriculum stages: {len(curriculum_config.stages)}")
            for (start_pct, end_pct), stage in curriculum_config.stages.items():
                start_step = int(start_pct * FLAGS.train_num_steps)
                end_step = int(end_pct * FLAGS.train_num_steps)
                print(f"  {stage.name}: steps {start_step}-{end_step} ({stage.focus})")
        except Exception as e:
            print(f"Warning: Failed to load curriculum '{FLAGS.curriculum_config}': {e}")
            print("Falling back to legacy behavior")
            curriculum_config = None
    else:
        if FLAGS.disable_curriculum:
            print("Curriculum disabled by --disable-curriculum flag")
        elif FLAGS.curriculum_config == 'none':
            print("Curriculum disabled by --curriculum-config=none")
        elif not CURRICULUM_AVAILABLE:
            print("Curriculum not available (curriculum_config module not found)")
        else:
            print("Using legacy training behavior")

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = 32,
        objective = 'pred_noise',  # Alternative pred_x0
        timesteps = FLAGS.diffusion_steps,  # number of steps
        sampling_timesteps = FLAGS.diffusion_steps,  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper]),
        supervise_energy_landscape = FLAGS.supervise_energy_landscape,
        use_innerloop_opt = FLAGS.use_innerloop_opt,
        show_inference_tqdm = False,
        use_adversarial_corruption = FLAGS.use_adversarial_corruption,
        anm_warmup_steps = FLAGS.anm_warmup_steps,
        anm_adversarial_steps = FLAGS.anm_adversarial_steps,
        anm_distance_penalty = FLAGS.anm_distance_penalty,
        curriculum_config = curriculum_config,
        disable_curriculum = FLAGS.disable_curriculum or FLAGS.curriculum_config == 'none',
        **kwargs
    )

    result_dir = osp.join('results', f'ds_{FLAGS.dataset}', f'model_{FLAGS.model}')
    if FLAGS.diffusion_steps != 100:
        result_dir = result_dir + f'_diffsteps_{FLAGS.diffusion_steps}'
    os.makedirs(result_dir, exist_ok=True)

    if FLAGS.latent:
        # Load the decoder
        autoencode_model = AutoencodeModel(729, 729)
        ckpt = torch.load("results/autoencode_sudoku-rrn/model_mlp_diffsteps_10/model-1.pt")
        model_ckpt = ckpt['model']
        autoencode_model.load_state_dict(model_ckpt)
    else:
        autoencode_model = None


    trainer = Trainer1D(
        diffusion,
        dataset,
        dataset_name = FLAGS.dataset,  # Pass dataset name for accuracy computation
        train_batch_size = FLAGS.batch_size,
        validation_batch_size = validation_batch_size,
        train_lr = 1e-4,
        train_num_steps = FLAGS.train_num_steps,         # total training steps
        gradient_accumulate_every = 1,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        data_workers = FLAGS.data_workers,
        amp = False,                      # turn on mixed precision
        metric = metric,
        results_folder = result_dir,
        cond_mask = FLAGS.cond_mask,
        validation_dataset = validation_dataset,
        extra_validation_datasets = extra_validation_datasets,
        extra_validation_every_mul = extra_validation_every_mul,
        save_and_sample_every = save_and_sample_every,
        evaluate_first = FLAGS.evaluate,  # run one evaluation first
        latent = FLAGS.latent,  # whether we are doing reasoning in the latent space
        autoencode_model = autoencode_model,
        save_csv_logs = FLAGS.save_csv_logs,
        csv_log_interval = FLAGS.csv_log_interval,
        csv_log_dir = FLAGS.csv_log_dir
    )

    if FLAGS.load_milestone is not None:
        trainer.load(FLAGS.load_milestone)

    trainer.train()
