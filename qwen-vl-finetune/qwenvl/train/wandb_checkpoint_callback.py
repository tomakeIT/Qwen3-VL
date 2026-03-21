import atexit
import logging
import os
import threading

import transformers


class WandbCheckpointUploadCallback(transformers.TrainerCallback):
    """Upload checkpoints to W&B artifacts on every save."""

    def __init__(self, training_args):
        self.enabled = bool(training_args.wandb_upload_checkpoints)
        self.artifact_name = training_args.wandb_checkpoint_artifact_name
        self.keep_last_n = max(1, int(training_args.wandb_checkpoint_keep_last_n or 1))
        self.upload_async = bool(training_args.wandb_checkpoint_upload_async)
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._stop_event = threading.Event()
        self._latest_checkpoint = None
        self._worker = None
        if self.upload_async:
            self._worker = threading.Thread(
                target=self._upload_worker,
                name="wandb-checkpoint-uploader",
                daemon=True,
            )
            self._worker.start()
            atexit.register(self._shutdown_worker)

    def on_save(self, args, state, control, **kwargs):
        if not self.enabled or not state.is_world_process_zero:
            return control

        if not self._wandb_ready(args):
            return control

        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(checkpoint_dir):
            return control

        if self.upload_async:
            with self._lock:
                self._latest_checkpoint = checkpoint_dir
            self._event.set()
        else:
            self._upload_checkpoint(checkpoint_dir)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if not self.enabled or not state.is_world_process_zero:
            return control

        if not self._wandb_ready(args):
            return control

        last_checkpoint = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(last_checkpoint):
            if self.upload_async:
                with self._lock:
                    self._latest_checkpoint = last_checkpoint
                self._event.set()
                self._shutdown_worker()
            else:
                self._upload_checkpoint(last_checkpoint)
        return control

    def _upload_worker(self):
        while True:
            self._event.wait(timeout=1.0)
            self._event.clear()
            with self._lock:
                checkpoint_dir = self._latest_checkpoint
                self._latest_checkpoint = None
            if checkpoint_dir:
                self._upload_checkpoint(checkpoint_dir)
            if self._stop_event.is_set():
                with self._lock:
                    final_checkpoint = self._latest_checkpoint
                    self._latest_checkpoint = None
                if final_checkpoint:
                    self._upload_checkpoint(final_checkpoint)
                break

    def _shutdown_worker(self):
        if not self._worker:
            return
        self._stop_event.set()
        self._event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=600)

    def _wandb_ready(self, args):
        report_to = args.report_to
        if isinstance(report_to, str):
            report_to = [report_to]
        if not report_to or "wandb" not in report_to:
            return False
        try:
            import wandb

            return wandb.run is not None
        except Exception:
            return False

    def _upload_checkpoint(self, checkpoint_dir):
        try:
            import wandb

            run = wandb.run
            if run is None:
                return

            artifact_name = self.artifact_name or f"{run.id}-checkpoints"
            step = self._extract_step(checkpoint_dir)

            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                metadata={"checkpoint_dir": os.path.basename(checkpoint_dir), "step": step},
            )
            artifact.add_dir(checkpoint_dir, name=os.path.basename(checkpoint_dir))

            aliases = ["latest"]
            if step is not None:
                aliases.append(f"step-{step}")
            run.log_artifact(artifact, aliases=aliases)
            self._cleanup_old_artifacts(run, artifact_name)
            logging.info("Uploaded checkpoint to W&B: %s", checkpoint_dir)
        except Exception as e:
            logging.warning("Failed to upload checkpoint to W&B: %s", e)

    def _cleanup_old_artifacts(self, run, artifact_name):
        if self.keep_last_n <= 0:
            return
        try:
            import wandb

            api = wandb.Api()
            path = f"{run.entity}/{run.project}/{artifact_name}"
            versions = list(api.artifact_versions(type_name="model", name=path))
            versions.sort(key=lambda a: int(a.version.lstrip("v")), reverse=True)
            for old_artifact in versions[self.keep_last_n :]:
                old_artifact.delete(delete_aliases=True)
        except Exception as e:
            logging.warning("W&B artifact cleanup skipped: %s", e)

    @staticmethod
    def _extract_step(checkpoint_dir):
        base = os.path.basename(checkpoint_dir)
        if not base.startswith("checkpoint-"):
            return None
        step_str = base.split("checkpoint-")[-1]
        return int(step_str) if step_str.isdigit() else None
