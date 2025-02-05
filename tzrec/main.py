# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import itertools
import json
import os
import shutil
from collections import OrderedDict
from queue import Queue
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Union

import pyarrow as pa
import torch
from torch import distributed as dist
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchrec.distributed.model_parallel import (
    DistributedModelParallel,
)

# NOQA
from torchrec.distributed.train_pipeline import TrainPipelineSparseDist
from torchrec.inference.state_dict_transform import (
    state_dict_gather,
    state_dict_to_device,
)
from torchrec.optim.apply_optimizer_in_backward import (
    apply_optimizer_in_backward,  # NOQA
)
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter

from tzrec.acc.aot_utils import export_model_aot
from tzrec.acc.quant_utils import quantize_embeddings
from tzrec.acc.trt_utils import export_model_trt, get_trt_max_batch_size
from tzrec.acc.utils import (
    export_acc_config,
    is_aot,
    is_cuda_export,
    is_input_tile_emb,
    is_quant,
    is_trt,
    is_trt_predict,
    write_mapping_file_for_input_tile,
)
from tzrec.constant import PREDICT_QUEUE_TIMEOUT, Mode
from tzrec.datasets.dataset import BaseDataset, BaseWriter, create_writer
from tzrec.datasets.utils import Batch, RecordBatchTensor
from tzrec.features.feature import (
    BaseFeature,
    create_feature_configs,
    create_features,
    create_fg_json,
)
from tzrec.models.match_model import (
    MatchModel,
    MatchTower,
    MatchTowerWoEG,
    TowerWoEGWrapper,
    TowerWrapper,
)
from tzrec.models.model import BaseModel, CudaExportWrapper, ScriptWrapper, TrainWrapper
from tzrec.models.tdm import TDM, TDMEmbedding
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.optim import optimizer_builder
from tzrec.optim.lr_scheduler import BaseLR
from tzrec.protos.data_pb2 import DataConfig, DatasetType
from tzrec.protos.eval_pb2 import EvalConfig
from tzrec.protos.feature_pb2 import FeatureConfig
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.pipeline_pb2 import EasyRecConfig
from tzrec.protos.train_pb2 import TrainConfig
from tzrec.utils import checkpoint_util, config_util
from tzrec.utils.fx_util import symbolic_trace
from tzrec.utils.logging_util import ProgressLogger, logger
from tzrec.utils.plan_util import create_planner, get_default_sharders
from tzrec.version import __version__ as tzrec_version


def init_process_group() -> Tuple[torch.device, str]:
    """Init process_group, device, rank, backend."""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        device: torch.device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device: torch.device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend)
    return device, backend


def _create_features(
    feature_configs: List[FeatureConfig], data_config: DataConfig
) -> List[BaseFeature]:
    neg_fields = None
    if data_config.HasField("sampler"):
        sampler_type = data_config.WhichOneof("sampler")
        if sampler_type != "tdm_sampler":
            neg_fields = list(
                getattr(data_config, data_config.WhichOneof("sampler")).attr_fields
            )

    features = create_features(
        feature_configs,
        fg_mode=data_config.fg_mode,
        neg_fields=neg_fields,
        fg_encoded_multival_sep=data_config.fg_encoded_multival_sep,
        force_base_data_group=data_config.force_base_data_group,
    )
    return features


def _get_dataloader(
    data_config: DataConfig,
    features: List[BaseFeature],
    input_path: str,
    reserved_columns: Optional[List[str]] = None,
    mode: Mode = Mode.TRAIN,
    gl_cluster: Optional[Dict[str, Union[int, str]]] = None,
    debug_level: int = 0,
) -> DataLoader:
    """Build dataloader.

    Args:
        data_config (DataConfig): dataloader config.
        features (list): list of feature.
        input_path (str): input data path.
        reserved_columns (list): reserved columns in predict mode.
        mode (Mode): train or eval or predict.
        gl_cluster (dict, bool): if set, reuse the graphlearn cluster.
        debug_level (int): dataset debug level, when mode=predict and
            debug_level > 0, will dump fg encoded data to debug_str

    Return:
        dataloader (dataloader): a DataLoader.
    """
    dataset_name = DatasetType.Name(data_config.dataset_type)
    # pyre-ignore [16]
    dataset_cls = BaseDataset.create_class(dataset_name)
    dataset = dataset_cls(
        data_config=data_config,
        features=features,
        input_path=input_path,
        reserved_columns=reserved_columns,
        mode=mode,
        debug_level=debug_level,
    )

    kwargs = {}
    if data_config.num_workers < 1:
        num_workers = 1
    else:
        num_workers = data_config.num_workers
        # check number of files is valid or not for file based dataset.
        if hasattr(dataset._reader, "num_files"):
            num_files = dataset._reader.num_files()
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            if num_files >= world_size:
                num_files_per_worker = num_files // world_size
                if num_files_per_worker < num_workers:
                    logger.info(
                        f"data_config.num_workers reset to {num_files_per_worker}"
                    )
                    num_workers = num_files_per_worker
            else:
                raise ValueError(
                    f"Number of files in the dataset[{input_path}] must greater "
                    f"than world_size: {world_size}, but got {num_files}"
                )

        kwargs["num_workers"] = num_workers
        kwargs["persistent_workers"] = True

    if mode == Mode.TRAIN:
        # When in train_and_eval mode, use 2x worker in gl cluster
        # for train_dataloader and eval_dataloader
        dataset.launch_sampler_cluster(num_client_per_rank=num_workers * 2)
    else:
        if gl_cluster:
            # Reuse the gl cluster for eval_dataloader
            dataset.launch_sampler_cluster(
                num_client_per_rank=num_workers * 2,
                client_id_bias=num_workers,
                cluster=gl_cluster,
            )
        else:
            dataset.launch_sampler_cluster(num_client_per_rank=num_workers)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=None,
        pin_memory=data_config.pin_memory if mode != Mode.PREDICT else False,
        collate_fn=lambda x: x,
        **kwargs,
    )
    return dataloader


def _create_model(
    model_config: ModelConfig,
    features: List[BaseFeature],
    labels: List[str],
    sample_weights: Optional[List[str]] = None,
) -> BaseModel:
    """Build model.

    Args:
        model_config (ModelConfig): easyrec model config.
        features (list): list of features.
        labels (list): list of label names.
        sample_weights (list): list of sample weight names.

    Return:
        model: a EasyRec Model.
    """
    model_cls_name = config_util.which_msg(model_config, "model")
    # pyre-ignore [16]
    model_cls = BaseModel.create_class(model_cls_name)

    model = model_cls(model_config, features, labels, sample_weights=sample_weights)
    return model


def _evaluate(
    model: nn.Module,
    eval_dataloader: DataLoader,
    eval_config: EvalConfig,
    eval_result_filename: Optional[str] = None,
    global_step: Optional[int] = None,
    eval_summary_writer: Optional[SummaryWriter] = None,
) -> Dict[str, torch.Tensor]:
    """Evaluate the model."""
    is_rank_zero = int(os.environ.get("RANK", 0)) == 0
    is_local_rank_zero = int(os.environ.get("LOCAL_RANK", 0)) == 0
    model.eval()
    pipeline = TrainPipelineSparseDist(
        model,
        # pyre-fixme [6]
        None,
        model.device,
        execute_all_batches=True,
    )

    use_step = eval_config.num_steps and eval_config.num_steps > 0
    iterator = iter(eval_dataloader)
    step_iter = range(eval_config.num_steps) if use_step else itertools.count(0)

    desc_suffix = ""
    if global_step:
        desc_suffix = f" model-{global_step}"
    _model = model.module.model

    plogger = None
    if is_local_rank_zero:
        plogger = ProgressLogger(desc=f"Evaluating{desc_suffix}")

    with torch.no_grad():
        i_step = 0
        for i_step in step_iter:
            try:
                losses, predictions, batch = pipeline.progress(iterator)
                _model.update_metric(predictions, batch, losses)
                if (
                    plogger is not None
                    and i_step % eval_config.log_step_count_steps == 0
                ):
                    plogger.log(i_step)
            except StopIteration:
                break
        if plogger is not None:
            plogger.log(i_step)

    metric_result = _model.compute_metric()
    if is_rank_zero:
        metric_str = " ".join([f"{k}:{v:0.6f}" for k, v in metric_result.items()])
        logger.info(f"Eval Result{desc_suffix}: {metric_str}")
        metric_result = {k: v.item() for k, v in metric_result.items()}
        if eval_result_filename:
            with open(eval_result_filename, "w") as f:
                json.dump(metric_result, f, indent=4)
        if eval_summary_writer:
            for k, v in metric_result.items():
                eval_summary_writer.add_scalar(f"metric/{k}", v, global_step or 0)
    return metric_result


def _log_train(
    step: int,
    losses: Dict[str, torch.Tensor],
    param_groups: List[Dict[str, Any]],
    plogger: Optional[ProgressLogger] = None,
    summary_writer: Optional[SummaryWriter] = None,
) -> None:
    """Logging current training step."""
    if plogger is not None:
        loss_strs = []
        lr_strs = []
        for k, v in losses.items():
            loss_strs.append(f"{k}:{v:.5f}")
        for i, g in enumerate(param_groups):
            lr_strs.append(f"lr_g{i}:{g['lr']:.5f}")
        total_loss = sum(losses.values())
        plogger.log(
            step,
            f"{' '.join(lr_strs)} {' '.join(loss_strs)} total_loss: {total_loss:.5f}",
        )
    if summary_writer is not None:
        total_loss = sum(losses.values())
        for k, v in losses.items():
            summary_writer.add_scalar(f"loss/{k}", v, step)
        summary_writer.add_scalar("loss/total", total_loss, step)
        for i, g in enumerate(param_groups):
            summary_writer.add_scalar(f"lr/g{i}", g["lr"], step)


def _train_and_evaluate(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader],
    lr_scheduler: List[BaseLR],
    model_dir: str,
    train_config: TrainConfig,
    eval_config: EvalConfig,
    skip_steps: int = -1,
    ckpt_path: Optional[str] = None,
    eval_result_filename: str = "train_eval_result.txt",
) -> None:
    """Train and evaluate the model."""
    is_rank_zero = int(os.environ.get("RANK", 0)) == 0
    is_local_rank_zero = int(os.environ.get("LOCAL_RANK", 0)) == 0
    model.train()

    assert train_config.num_steps ^ train_config.num_epochs, (
        "train_config.num_epochs or train_config.num_steps should be set, "
        "and can not be set at the same time."
    )
    use_epoch = train_config.num_epochs and train_config.num_epochs > 0
    use_step = train_config.num_steps and train_config.num_steps > 0
    epoch_iter = range(train_config.num_epochs) if use_epoch else itertools.count(0, 0)
    step_iter = range(train_config.num_steps) if use_step else itertools.count(0)

    plogger = None
    summary_writer = None
    eval_summary_writer = None
    if is_local_rank_zero:
        plogger = ProgressLogger(desc="Training Epoch 0", start_n=skip_steps)
    if is_rank_zero and train_config.use_tensorboard:
        summary_writer = SummaryWriter(model_dir)
        eval_summary_writer = SummaryWriter(os.path.join(model_dir, "eval_val"))

    if train_config.is_profiling:
        if is_rank_zero:
            logger.info(str(model))
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(model_dir, "train_eval_trace")
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()

    last_ckpt_step = -1
    i_step = 0
    losses = {}
    for i_epoch in epoch_iter:
        pipeline = TrainPipelineSparseDist(
            model, optimizer, model.device, execute_all_batches=True
        )
        if plogger is not None:
            plogger.set_description(f"Training Epoch {i_epoch}")

        train_iterator = iter(train_dataloader)

        # Restore model and optimizer checkpoint, because optimizer's state
        # is lazy init, we should do a dummy step before restore.
        if i_step == 0 and ckpt_path is not None:
            peek_batch = next(train_iterator)
            pipeline.progress(iter([peek_batch]))
            train_iterator = itertools.chain([peek_batch], train_iterator)
            checkpoint_util.restore_model(
                ckpt_path, model, optimizer, train_config.fine_tune_ckpt_param_map
            )

        for i_step in step_iter:
            if i_step <= skip_steps:
                continue
            try:
                losses, _, _ = pipeline.progress(train_iterator)

                if i_step % train_config.log_step_count_steps == 0:
                    _log_train(
                        i_step,
                        losses,
                        param_groups=optimizer.param_groups,
                        plogger=plogger,
                        summary_writer=summary_writer,
                    )

                for lr in lr_scheduler:
                    if not lr.by_epoch:
                        lr.step()
            except StopIteration:
                step_iter = itertools.chain([i_step], step_iter)
                i_step -= 1
                break
            if train_config.save_checkpoints_steps > 0 and i_step > 0:
                if i_step % train_config.save_checkpoints_steps == 0:
                    last_ckpt_step = i_step
                    checkpoint_util.save_model(
                        os.path.join(model_dir, f"model.ckpt-{i_step}"),
                        model,
                        optimizer,
                    )
                    if eval_dataloader is not None:
                        _evaluate(
                            model,
                            eval_dataloader,
                            eval_config,
                            global_step=i_step,
                            eval_summary_writer=eval_summary_writer,
                        )
                        model.train()
            if train_config.is_profiling:
                prof.step()

        if use_step and i_step >= train_config.num_steps - 1:
            break

        for lr in lr_scheduler:
            if lr.by_epoch:
                lr.step()

    _log_train(
        i_step,
        losses,
        param_groups=optimizer.param_groups,
        plogger=plogger,
        summary_writer=summary_writer,
    )
    if summary_writer is not None:
        summary_writer.close()
    if train_config.is_profiling:
        prof.stop()
    if last_ckpt_step != i_step:
        checkpoint_util.save_model(
            os.path.join(model_dir, f"model.ckpt-{i_step}"),
            model,
            optimizer,
        )
        if eval_dataloader is not None:
            _evaluate(
                model,
                eval_dataloader,
                eval_config,
                os.path.join(model_dir, eval_result_filename),
                global_step=i_step,
                eval_summary_writer=eval_summary_writer,
            )
            model.train()


def train_and_evaluate(
    pipeline_config_path: str,
    train_input_path: Optional[str] = None,
    eval_input_path: Optional[str] = None,
    model_dir: Optional[str] = None,
    continue_train: Optional[bool] = True,
    fine_tune_checkpoint: Optional[str] = None,
    edit_config_json: Optional[str] = None,
) -> None:
    """Train and evaluate a EasyRec model.

    Args:
        pipeline_config_path (str): path to EasyRecConfig object.
        train_input_path (str, optional): train data path.
        eval_input_path (str, optional): eval data path.
        model_dir (str, optionl): model directory.
        continue_train (bool, optional): whether to restart train from
            an existing checkpoint.
        fine_tune_checkpoint (str, optional): path to an existing
            finetune checkpoint.
        edit_config_json (str, optional): edit pipeline config json str.
    """
    pipeline_config = config_util.load_pipeline_config(pipeline_config_path)
    if fine_tune_checkpoint:
        pipeline_config.train_config.fine_tune_checkpoint = fine_tune_checkpoint
    if train_input_path:
        pipeline_config.train_input_path = train_input_path
    if eval_input_path:
        pipeline_config.eval_input_path = eval_input_path
    if model_dir:
        pipeline_config.model_dir = model_dir
    if edit_config_json:
        edit_config_json = json.loads(edit_config_json)
        config_util.edit_config(pipeline_config, edit_config_json)

    device, _ = init_process_group()
    is_rank_zero = int(os.environ.get("RANK", 0)) == 0
    is_local_rank_zero = int(os.environ.get("LOCAL_RANK", 0)) == 0

    data_config = pipeline_config.data_config
    # Build feature
    features = _create_features(list(pipeline_config.feature_configs), data_config)

    # Build dataloader
    train_dataloader = _get_dataloader(
        data_config, features, pipeline_config.train_input_path, mode=Mode.TRAIN
    )
    eval_dataloader = None
    if pipeline_config.eval_input_path:
        # pyre-ignore [16]
        gl_cluster = train_dataloader.dataset.get_sampler_cluster()
        eval_dataloader = _get_dataloader(
            data_config,
            features,
            pipeline_config.eval_input_path,
            mode=Mode.EVAL,
            gl_cluster=gl_cluster,
        )

    # Build model
    model = _create_model(
        pipeline_config.model_config,
        features,
        list(data_config.label_fields),
        sample_weights=list(data_config.sample_weight_fields),
    )
    model = TrainWrapper(model)

    sparse_optim_cls, sparse_optim_kwargs = optimizer_builder.create_sparse_optimizer(
        pipeline_config.train_config.sparse_optimizer
    )
    apply_optimizer_in_backward(
        sparse_optim_cls, model.model.sparse_parameters(), sparse_optim_kwargs
    )

    planner = create_planner(
        device=device,
        # pyre-ignore [16]
        batch_size=train_dataloader.dataset.sampled_batch_size,
    )

    plan = planner.collective_plan(
        model, get_default_sharders(), dist.GroupMember.WORLD
    )
    if is_rank_zero:
        logger.info(str(plan))

    model = DistributedModelParallel(
        module=model,
        device=device,
        plan=plan,
    )

    dense_optim_cls, dense_optim_kwargs = optimizer_builder.create_dense_optimizer(
        pipeline_config.train_config.dense_optimizer
    )
    dense_optimizer = KeyedOptimizerWrapper(
        dict(in_backward_optimizer_filter(model.named_parameters())),
        lambda params: dense_optim_cls(params, **dense_optim_kwargs),
    )
    optimizer = CombinedOptimizer([model.fused_optimizer, dense_optimizer])
    sparse_lr = optimizer_builder.create_scheduler(
        model.fused_optimizer, pipeline_config.train_config.sparse_optimizer
    )
    dense_lr = optimizer_builder.create_scheduler(
        dense_optimizer, pipeline_config.train_config.dense_optimizer
    )

    ckpt_path = None
    skip_steps = -1
    if pipeline_config.train_config.fine_tune_checkpoint:
        ckpt_path, _ = checkpoint_util.latest_checkpoint(
            pipeline_config.train_config.fine_tune_checkpoint
        )
    if os.path.exists(pipeline_config.model_dir):
        # TODO(hongsheng.jhs): save and restore dataloader state.
        latest_ckpt_path, skip_steps = checkpoint_util.latest_checkpoint(
            pipeline_config.model_dir
        )
        if latest_ckpt_path:
            if continue_train:
                ckpt_path = latest_ckpt_path
            else:
                raise RuntimeError(
                    f"model_dir[{pipeline_config.model_dir}] already exists "
                    "and not empty(if you want to continue train on current "
                    "model_dir please delete dir model_dir or specify "
                    "--continue_train)"
                )

    # use barrier to sync all workers, prevent rank zero save_message and create
    # model_dir first, other slow rank find model_dir already exists and
    # do continue train improperly.
    dist.barrier()

    if is_rank_zero:
        config_util.save_message(
            pipeline_config, os.path.join(pipeline_config.model_dir, "pipeline.config")
        )
        with open(os.path.join(pipeline_config.model_dir, "version"), "w") as f:
            f.write(tzrec_version + "\n")

    _train_and_evaluate(
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        [sparse_lr, dense_lr],
        pipeline_config.model_dir,
        train_config=pipeline_config.train_config,
        eval_config=pipeline_config.eval_config,
        skip_steps=skip_steps,
        ckpt_path=ckpt_path,
    )
    if is_local_rank_zero:
        logger.info("Train and Evaluate Finished.")


def evaluate(
    pipeline_config_path: str,
    checkpoint_path: Optional[str] = None,
    eval_input_path: Optional[str] = None,
    eval_result_filename: str = "eval_result.txt",
) -> None:
    """Evaluate a EasyRec model.

    Args:
        pipeline_config_path (str): path to EasyRecConfig object.
        checkpoint_path (str, optional): if specified, will use this model instead of
            model specified by model_dir in pipeline_config_path
        eval_input_path (str, optional): eval data path, default use eval data in
            pipeline_config, could be a path or a list of paths
        eval_result_filename (str): evaluation result metrics save path.
    """
    pipeline_config = config_util.load_pipeline_config(pipeline_config_path)

    device, _ = init_process_group()
    is_rank_zero = int(os.environ.get("RANK", 0)) == 0
    is_local_rank_zero = int(os.environ.get("LOCAL_RANK", 0)) == 0

    data_config = pipeline_config.data_config
    # Build feature
    features = _create_features(list(pipeline_config.feature_configs), data_config)

    eval_dataloader = _get_dataloader(
        data_config,
        features,
        eval_input_path or pipeline_config.eval_input_path,
        mode=Mode.EVAL,
    )

    # Build model
    model = _create_model(
        pipeline_config.model_config,
        features,
        list(data_config.label_fields),
        sample_weights=list(data_config.sample_weight_fields),
    )
    model = TrainWrapper(model)

    planner = create_planner(
        device=device,
        # pyre-ignore [16]
        batch_size=eval_dataloader.dataset.sampled_batch_size,
    )
    plan = planner.collective_plan(
        model, get_default_sharders(), dist.GroupMember.WORLD
    )
    if is_rank_zero:
        logger.info(str(plan))

    model = DistributedModelParallel(module=model, device=device, plan=plan)

    global_step = None
    if not checkpoint_path:
        checkpoint_path, global_step = checkpoint_util.latest_checkpoint(
            pipeline_config.model_dir
        )
    if checkpoint_path:
        checkpoint_util.restore_model(checkpoint_path, model)
    else:
        raise ValueError("Eval checkpoint path should be specified.")

    summary_writer = None
    if is_rank_zero:
        summary_writer = SummaryWriter(os.path.join(pipeline_config.model_dir, "eval"))
    _evaluate(
        model,
        eval_dataloader,
        eval_config=pipeline_config.eval_config,
        eval_result_filename=os.path.join(
            pipeline_config.model_dir, eval_result_filename
        ),
        global_step=global_step,
        eval_summary_writer=summary_writer,
    )
    if is_local_rank_zero:
        logger.info("Evaluate Finished.")


def _script_model(
    pipeline_config: EasyRecConfig,
    model: nn.Module,
    state_dict: Dict[str, Any],
    dataloader: DataLoader,
    save_dir: str,
) -> None:
    is_rank_zero = int(os.environ.get("RANK", 0)) == 0
    is_trt_convert = is_trt()
    if is_rank_zero:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model = model.to_empty(device="cpu")
        logger.info("gather states to cpu model...")

    state_dict_gather(state_dict, model.state_dict())

    dist.barrier()

    if is_rank_zero:
        batch = next(iter(dataloader))

        if is_cuda_export():
            model = model.cuda()

        if is_quant():
            logger.info("quantize embeddings...")
            quantize_embeddings(model, dtype=torch.qint8, inplace=True)

        model.eval()

        if is_trt_convert:
            data_cuda = batch.to_dict(sparse_dtype=torch.int64)
            result = model(data_cuda, "cuda:0")
            result_info = {k: (v.size(), v.dtype) for k, v in result.items()}
            logger.info(f"Model Outputs: {result_info}")
            export_model_trt(model, data_cuda, save_dir)

        elif is_aot():
            data_cuda = batch.to_dict(sparse_dtype=torch.int64)
            result = model(data_cuda)
            export_model_aot(model, data_cuda, save_dir)
        else:
            data = batch.to_dict(sparse_dtype=torch.int64)
            result = model(data)
            result_info = {k: (v.size(), v.dtype) for k, v in result.items()}
            logger.info(f"Model Outputs: {result_info}")

            gm = symbolic_trace(model)
            with open(os.path.join(save_dir, "gm.code"), "w") as f:
                f.write(gm.code)

            scripted_model = torch.jit.script(gm)
            scripted_model.save(os.path.join(save_dir, "scripted_model.pt"))

        features = model._features
        feature_configs = create_feature_configs(features, asset_dir=save_dir)
        pipeline_config = copy.copy(pipeline_config)
        pipeline_config.ClearField("feature_configs")
        pipeline_config.feature_configs.extend(feature_configs)
        config_util.save_message(
            pipeline_config, os.path.join(save_dir, "pipeline.config")
        )
        logger.info("saving fg json...")
        fg_json = create_fg_json(features, asset_dir=save_dir)
        with open(os.path.join(save_dir, "fg.json"), "w") as f:
            json.dump(fg_json, f, indent=4)
        with open(os.path.join(save_dir, "model_acc.json"), "w") as f:
            json.dump(export_acc_config(), f, indent=4)


def export(
    pipeline_config_path: str,
    export_dir: str,
    checkpoint_path: Optional[str] = None,
    asset_files: Optional[str] = None,
) -> None:
    """Export a EasyRec model.

    Args:
        pipeline_config_path (str): path to EasyRecConfig object.
        export_dir (str): base directory where the model should be exported.
        checkpoint_path (str, optional): if specified, will use this model instead of
            model specified by model_dir in pipeline_config_path.
        asset_files (str, optional): more files will be copied to export_dir.
    """
    pipeline_config = config_util.load_pipeline_config(pipeline_config_path)
    ori_pipeline_config = copy.copy(pipeline_config)

    device, _ = init_process_group()
    is_rank_zero = int(os.environ.get("RANK", 0)) == 0
    if is_rank_zero:
        if os.path.exists(export_dir):
            raise RuntimeError(f"directory {export_dir} already exist.")

    assets = []
    if asset_files:
        assets = asset_files.split(",")

    data_config = pipeline_config.data_config
    is_trt_convert = is_trt()
    if is_trt_convert:
        # export batch_size too large may OOM in trt convert phase
        max_batch_size = get_trt_max_batch_size()
        data_config.batch_size = min(data_config.batch_size, max_batch_size)
        logger.info("using new batch_size: %s in trt export", data_config.batch_size)

    # Build feature
    features = _create_features(list(pipeline_config.feature_configs), data_config)

    # make dataparser to get user feats before create model
    data_config.num_workers = 1
    dataloader = _get_dataloader(
        data_config, features, pipeline_config.train_input_path, mode=Mode.PREDICT
    )

    # Build model
    model = _create_model(
        pipeline_config.model_config,
        features,
        list(data_config.label_fields),
    )
    model = ScriptWrapper(model)

    planner = create_planner(
        device=device,
        batch_size=data_config.batch_size,
    )
    plan = planner.collective_plan(
        model, get_default_sharders(), dist.GroupMember.WORLD
    )
    if is_rank_zero:
        logger.info(str(plan))

    model = DistributedModelParallel(module=model, device=device, plan=plan)

    if not checkpoint_path:
        checkpoint_path, _ = checkpoint_util.latest_checkpoint(
            pipeline_config.model_dir
        )
    if checkpoint_path:
        if is_input_tile_emb():
            remap_file_path = os.path.join(export_dir, "emb_ckpt_mapping.txt")
            if is_rank_zero:
                if not os.path.exists(export_dir):
                    os.makedirs(export_dir)
                write_mapping_file_for_input_tile(model.state_dict(), remap_file_path)

            dist.barrier()
            checkpoint_util.restore_model(
                checkpoint_path, model, ckpt_param_map_path=remap_file_path
            )
        else:
            checkpoint_util.restore_model(checkpoint_path, model)
    else:
        raise ValueError("checkpoint path should be specified.")

    checkpoint_pg = dist.new_group(backend="gloo")
    if is_rank_zero:
        logger.info("copy sharded state_dict to cpu...")
    cpu_state_dict = state_dict_to_device(
        model.state_dict(), pg=checkpoint_pg, device=torch.device("cpu")
    )

    cpu_model = _create_model(
        pipeline_config.model_config,
        features,
        list(data_config.label_fields),
    )

    InferWrapper = CudaExportWrapper if is_aot() else ScriptWrapper
    if isinstance(cpu_model, MatchModel):
        for name, module in cpu_model.named_children():
            if isinstance(module, MatchTower) or isinstance(module, MatchTowerWoEG):
                wrapper = (
                    TowerWrapper if isinstance(module, MatchTower) else TowerWoEGWrapper
                )
                tower = InferWrapper(wrapper(module, name))
                tower_export_dir = os.path.join(export_dir, name.replace("_tower", ""))
                _script_model(
                    ori_pipeline_config,
                    tower,
                    cpu_state_dict,
                    dataloader,
                    tower_export_dir,
                )
                for asset in assets:
                    shutil.copy(asset, tower_export_dir)
    elif isinstance(cpu_model, TDM):
        for name, module in cpu_model.named_children():
            if isinstance(module, EmbeddingGroup):
                emb_module = InferWrapper(TDMEmbedding(module, name))
                _script_model(
                    ori_pipeline_config,
                    emb_module,
                    cpu_state_dict,
                    dataloader,
                    os.path.join(export_dir, "embedding"),
                )
                break
        _script_model(
            ori_pipeline_config,
            InferWrapper(cpu_model),
            cpu_state_dict,
            dataloader,
            os.path.join(export_dir, "model"),
        )
        for asset in assets:
            shutil.copy(asset, os.path.join(export_dir, "model"))
    else:
        _script_model(
            ori_pipeline_config,
            InferWrapper(cpu_model),
            cpu_state_dict,
            dataloader,
            export_dir,
        )
        for asset in assets:
            shutil.copy(asset, export_dir)


def predict(
    predict_input_path: str,
    predict_output_path: str,
    scripted_model_path: str,
    reserved_columns: Optional[str] = None,
    output_columns: Optional[str] = None,
    batch_size: Optional[int] = None,
    is_profiling: bool = False,
    debug_level: int = 0,
    dataset_type: Optional[str] = None,
    predict_threads: Optional[int] = None,
    writer_type: Optional[str] = None,
    edit_config_json: Optional[str] = None,
) -> None:
    """Evaluate a EasyRec model.

    Args:
        predict_input_path (str): inference input data path.
        predict_output_path (str): inference output data path.
        scripted_model_path (str): path to scripted model.
        reserved_columns (str, optional): columns to reserved in output.
        output_columns (str, optional): columns of model output.
        batch_size (int, optional): predict batch_size.
        is_profiling (bool): profiling predict process or not.
        debug_level (int, optional): debug level for debug parsed inputs etc.
        dataset_type (str, optional): dataset type, default use the type in pipeline.
        predict_threads (int, optional): predict threads num, default will
            use num_workers in data_config.
        writer_type (int, optional): data writer type, default will be same as
            dataset_type in data_config.
        edit_config_json (str, optional): edit pipeline config json str.
    """
    reserved_cols: Optional[List[str]] = None
    output_cols: Optional[List[str]] = None
    if reserved_columns is not None:
        reserved_cols = [x.strip() for x in reserved_columns.split(",")]
    if output_columns is not None:
        output_cols = [x.strip() for x in output_columns.split(",")]

    pipeline_config = config_util.load_pipeline_config(
        os.path.join(scripted_model_path, "pipeline.config"), allow_unknown_field=True
    )
    if batch_size:
        pipeline_config.data_config.batch_size = batch_size

    is_trt_convert: bool = is_trt_predict(scripted_model_path)
    if is_trt_convert:
        # predict batch_size too large may out of range
        max_batch_size = get_trt_max_batch_size()
        pipeline_config.data_config.batch_size = min(
            pipeline_config.data_config.batch_size, max_batch_size
        )
        logger.info(
            "using new batch_size: %s in trt predict",
            pipeline_config.data_config.batch_size,
        )

    if dataset_type:
        pipeline_config.data_config.dataset_type = getattr(DatasetType, dataset_type)
    if edit_config_json:
        edit_config_json = json.loads(edit_config_json)
        config_util.edit_config(pipeline_config, edit_config_json)

    device_and_backend = init_process_group()
    device: torch.device = device_and_backend[0]

    is_rank_zero = int(os.environ.get("RANK", 0)) == 0
    is_local_rank_zero = int(os.environ.get("LOCAL_RANK", 0)) == 0

    data_config: DataConfig = pipeline_config.data_config
    data_config.drop_remainder = False
    # Build feature
    features = _create_features(list(pipeline_config.feature_configs), data_config)

    infer_dataloader = _get_dataloader(
        data_config,
        features,
        predict_input_path,
        reserved_columns=reserved_cols,
        mode=Mode.PREDICT,
        debug_level=debug_level,
    )
    infer_iterator = iter(infer_dataloader)

    if writer_type is None:
        writer_type = DatasetType.Name(data_config.dataset_type).replace(
            "Dataset", "Writer"
        )
    writer: BaseWriter = create_writer(
        predict_output_path,
        writer_type,
        quota_name=data_config.odps_data_quota_name,
    )

    # disable jit compile， as it compile too slow now.
    if "PYTORCH_TENSOREXPR_FALLBACK" not in os.environ:
        os.environ["PYTORCH_TENSOREXPR_FALLBACK"] = "2"

    model: torch.jit.ScriptModule = torch.jit.load(
        os.path.join(scripted_model_path, "scripted_model.pt"), map_location=device
    )
    model.eval()

    if is_local_rank_zero:
        plogger = ProgressLogger(desc="Predicting", miniters=10)

    if is_profiling:
        if is_rank_zero:
            logger.info(str(model))
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(scripted_model_path, "predict_trace")
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()

    if predict_threads is None:
        predict_threads = max(data_config.num_workers, 1)
    data_queue: Queue[Optional[Batch]] = Queue(maxsize=predict_threads * 2)
    pred_queue: Queue[
        Tuple[Optional[Dict[str, torch.Tensor]], Optional[RecordBatchTensor]]
    ] = Queue(maxsize=predict_threads * 2)

    def _forward(batch: Batch) -> Tuple[Dict[str, torch.Tensor], RecordBatchTensor]:
        with torch.no_grad():
            parsed_inputs = batch.to_dict(sparse_dtype=torch.int64)
            # when predicting with a model exported using INPUT_TILE,
            #  we set the batch size tensor to 1 to disable tiling.
            parsed_inputs["batch_size"] = torch.tensor(1, dtype=torch.int64)
            if is_trt_convert:
                predictions = model(parsed_inputs)
            else:
                predictions = model(parsed_inputs, device)
            predictions = {k: v.to("cpu") for k, v in predictions.items()}
            return predictions, batch.reserves

    def _write(
        predictions: Dict[str, torch.Tensor],
        reserves: RecordBatchTensor,
        output_cols: List[str],
    ) -> None:
        output_dict = OrderedDict()
        for c in output_cols:
            v = predictions[c]
            v = v.tolist() if v.ndim > 1 else v.numpy()
            output_dict[c] = pa.array(v)
        reserve_batch_record = reserves.get()
        if reserve_batch_record is not None:
            for k, v in zip(
                reserve_batch_record.column_names, reserve_batch_record.columns
            ):
                output_dict[k] = v
        writer.write(output_dict)

    def _write_loop(output_cols: List[str]) -> None:
        while True:
            predictions, reserves = pred_queue.get(timeout=PREDICT_QUEUE_TIMEOUT)
            if predictions is None:
                break
            assert predictions is not None and reserves is not None
            _write(predictions, reserves, output_cols)

    def _forward_loop() -> None:
        while True:
            batch = data_queue.get(timeout=PREDICT_QUEUE_TIMEOUT)
            if batch is None:
                break
            assert batch is not None
            pred = _forward(batch)
            pred_queue.put(pred, timeout=PREDICT_QUEUE_TIMEOUT)

    forward_t_list = []
    write_t = None
    i_step = 0
    while True:
        try:
            batch = next(infer_iterator)

            if i_step == 0:
                # lazy init writer and create write and forward thread
                predictions, reserves = _forward(batch)
                if output_cols is None:
                    output_cols = sorted(predictions.keys())
                _write(predictions, reserves, output_cols)
                for _ in range(predict_threads):
                    t = Thread(target=_forward_loop)
                    t.start()
                    forward_t_list.append(t)
                write_t = Thread(target=_write_loop, args=(output_cols,))
                write_t.start()
            else:
                data_queue.put(batch, timeout=PREDICT_QUEUE_TIMEOUT)

            if is_local_rank_zero:
                plogger.log(i_step)
            if is_profiling:
                prof.step()
            i_step += 1
        except StopIteration:
            break

    for _ in range(predict_threads):
        data_queue.put(None, timeout=PREDICT_QUEUE_TIMEOUT)
    for t in forward_t_list:
        t.join()
    pred_queue.put((None, None), timeout=PREDICT_QUEUE_TIMEOUT)
    assert write_t is not None
    write_t.join()
    writer.close()

    if is_profiling:
        prof.stop()
    if is_local_rank_zero:
        logger.info("Predict Finished.")
