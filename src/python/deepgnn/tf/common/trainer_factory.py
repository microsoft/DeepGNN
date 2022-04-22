# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from deepgnn import TrainerType, get_logger


def get_trainer(param):
    if param.eager:
        ## Current TF2 Trainer only use 2.x code.
        ## * no v1.Session, placeholder.
        ## * use for-loop to access dataset.
        return _get_tf2_trainer(param)
    else:
        ## TF1Trainer will disable v2 behavior(`tf.disable_v2_behavior()`),
        ## It use `tf.compat.v1` to manage session and dataset, and other 1.x-style functionality.
        ## As Tensorflow 2.x support `tf.compat.v1` API, we can still run TF1Trainer in Tensorflow 2.x.
        return _get_tf1_trainer(param)


def _get_tf1_trainer(param):
    """
    Supported Trainer:
      * PSTrainer (parameter server)
      * HorovodTFTrainer
    """
    if param.trainer == TrainerType.HVD:
        ## Only import hvd_trainer if needed as docker images may not install horovod.
        from deepgnn.tf.common.horovod_trainer import HorovodTFTrainer

        return HorovodTFTrainer(
            trainer=param.trainer,
            model_dir=param.model_dir,
            seed=param.seed,
            user_name=param.user_name,
            job_id=param.job_id,
            gpu=param.gpu,
            log_save_steps=param.log_save_steps,
            summary_save_steps=param.summary_save_steps,
            profiler_save_secs=param.profiler_save_secs,
            checkpoint_save_secs=param.checkpoint_save_secs,
            logger=get_logger(),
        )
    else:
        from deepgnn.tf.common.ps_trainer import PSTrainer

        return PSTrainer(
            trainer=param.trainer,
            model_dir=param.model_dir,
            seed=param.seed,
            ps_hosts=param.ps_hosts,
            job_name=param.job_name,
            worker_hosts=param.worker_hosts,
            task_index=param.task_index,
            user_name=param.user_name,
            job_id=param.job_id,
            gpu=param.gpu,
            log_save_steps=param.log_save_steps,
            summary_save_steps=param.summary_save_steps,
            profiler_save_secs=param.profiler_save_secs,
            checkpoint_save_secs=param.checkpoint_save_secs,
            logger=get_logger(),
        )


def _get_tf2_trainer(param):
    # TODO: support ParameterServerStrategy
    if param.trainer == TrainerType.HVD:
        from deepgnn.tf.common.tf2_horovod_trainer import HorovodEagerTrainer

        return HorovodEagerTrainer(
            trainer=param.trainer,
            model_dir=param.model_dir,
            seed=param.seed,
            user_name=param.user_name,
            job_id=param.job_id,
            gpu=param.gpu,
            log_save_steps=param.log_save_steps,
            summary_save_steps=param.summary_save_steps,
            checkpoint_save_secs=param.checkpoint_save_secs,
            profile_batch=param.profile_batch,
            logger=get_logger(),
        )
    else:
        from deepgnn.tf.common.tf2_trainer import EagerTrainer

        return EagerTrainer(
            model_dir=param.model_dir,
            seed=param.seed,
            user_name=param.user_name,
            job_id=param.job_id,
            gpu=param.gpu,
            log_save_steps=param.log_save_steps,
            summary_save_steps=param.summary_save_steps,
            checkpoint_save_secs=param.checkpoint_save_secs,
            profile_batch=param.profile_batch,
            logger=get_logger(),
        )
