import os
import json
import random
import argparse
import yaml
import logging
from types import SimpleNamespace
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.prompt import build_prompt_with_reference_multiview
from utils.data_formatting import (
    build_qwen_data_sample,
    compute_delta_progress_label_int,
)
from utils.utils import abs_to_rel_path, compute_mean_pixel_diff, list_image_files, list_subdirs, dict_to_namespace
from utils.frame_sampling import sample_reference_frames_from_demo

# 配置 logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)



def sample_pair_indices(
    T: int,
    max_delta_t: int,
    min_delta_t: int = 1,
    peak_distance: int = 3,
    rise_factor: float = 1.3,
    decay_factor: float = 0.8,
) -> Optional[Tuple[int, int]]:
    """
    在给定长度为 T 的帧序列里，按山峰分布随机采样 (i, j)
    
    山峰分布：在 peak_distance 位置达到最大权重，左侧上升，右侧衰减
    
    Args:
        T: 序列长度
        max_delta_t: 最大时间差
        min_delta_t: 最小时间差（硬限制）
        peak_distance: 峰值位置（权重最大的距离）
        rise_factor: 上升因子（峰值左侧，>1 表示上升速度）
        decay_factor: 衰减因子（峰值右侧，0 < decay_factor < 1），越小衰减越快
    
    Returns:
        (i, j) 或 None
    """
    if T < 2:
        return None
    
    i = random.randint(0, T - 2)
    
    # 计算所有可能的 delta_t 及其权重（山峰分布）
    candidates: List[Tuple[int, float]] = []
    
    # 计算权重函数
    def calculate_weight(dt: int) -> float:
        if dt < peak_distance:
            # 上升阶段：从 min_delta_t 到 peak_distance
            weight = rise_factor ** (dt - min_delta_t)
        elif dt == peak_distance:
            # 峰值位置：权重为 1.0
            weight = 1.0
        else:
            # 衰减阶段：从 peak_distance 到 max_delta_t
            weight = decay_factor ** (dt - peak_distance)
        return weight
    
    # 正向采样
    max_forward = min(max_delta_t, T - 1 - i)
    for dt in range(min_delta_t, max_forward + 1):
        weight = calculate_weight(dt)
        candidates.append((dt, weight))
    
    # 反向采样
    max_backward = min(max_delta_t, i)
    for dt in range(min_delta_t, max_backward + 1):
        weight = calculate_weight(dt)
        candidates.append((-dt, weight))
    
    if not candidates:
        return None
    
    # 按权重采样（加权随机选择）
    deltas, weights = zip(*candidates)
    total_weight = sum(weights)
    if total_weight == 0:
        return None
    
    r = random.uniform(0, total_weight)
    cumsum = 0
    for delta_t, weight in candidates:
        cumsum += weight
        if r <= cumsum:
            j = i + delta_t
            return i, j
    
    # 如果没选到（理论上不会发生），返回第一个
    delta_t = candidates[0][0]
    j = i + delta_t
    return i, j


def sample_num_ref_frames(mean_ref_frames: int, min_frames: int = 3, max_frames: int = 10, std: float = 2.0) -> int:
    """以 mean_ref_frames 为均值做一次高斯采样，得到本次实际使用的 reference time steps 数量"""
    n = int(round(random.gauss(mean_ref_frames, std)))
    n = max(min_frames, min(max_frames, n))
    return n


def sample_reference_multiview_frames(
    root: str,
    task_name: str,
    valid_task_demos: Dict[str, List[str]],
    cur_demo_name: str,
    num_ref_frames: int,
    ref_jitter: int,
    reference_views: List[str],
) -> Tuple[List[str], List[int]]:
    """在同一个 task 中随机选一个 demo 作为 reference demo，然后从指定的视角中采样若干 time steps"""
    task_path = os.path.join(root, task_name)
    demos = valid_task_demos.get(task_name, [])
    candidates = [d for d in demos if d != cur_demo_name]
    if not candidates:
        candidates = demos if demos else [cur_demo_name]

    ref_demo_path = None
    for demo_name in random.sample(candidates, len(candidates)):
        demo_path = os.path.join(task_path, demo_name)
        view_names = list_subdirs(demo_path)
        if all(v in view_names for v in reference_views):
            ref_demo_path = demo_path
            break

    if ref_demo_path is None:
        fallback_path = os.path.join(task_path, cur_demo_name)
        view_names = list_subdirs(fallback_path)
        if all(v in view_names for v in reference_views):
            ref_demo_path = fallback_path
        else:
            return [], []

    return sample_reference_frames_from_demo(
        reference_demo_path=ref_demo_path,
        reference_views=reference_views,
        num_ref_frames=num_ref_frames,
        ref_jitter=ref_jitter,
    )




# ==================== 数据集构建 ====================

class DatasetBuilder:
    """数据集构建核心类"""
    
    def __init__(self, config: SimpleNamespace):
        self.config = config
        self.root = config.root
        self.required_views = config.sampling.required_views
    
    def generate_samples_for_task_split(
        self,
        split_name: str,
        task_name: str,
        demo_names: List[str],
        task_desc_map: Dict[str, str],
        all_task_names: List[str],
        valid_task_demos_for_ref: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        """为某个 task 在给定 demo 列表上生成一个 split (train/eval) 的样本"""
        samples: List[Dict[str, Any]] = []
        task_path = os.path.join(self.root, task_name)

        task_desc = task_desc_map[task_name]
        logger.info(f"Processing task ({split_name}): {task_name}")

        # for each demo in task
        for demo_name in demo_names:
            demo_path = os.path.join(task_path, demo_name)

            # get all frames for all views
            view_to_frames: Dict[str, List[str]] = {}
            for v in self.required_views:
                v_path = os.path.join(demo_path, v)
                frames = list_image_files(v_path)
                view_to_frames[v] = frames

            T = min(len(frames) for frames in view_to_frames.values())
            logger.info(f"  {task_name}/{demo_name} multi-view with T={T} (views: {self.required_views}).")
            if T < 2:
                continue

            ref_img_paths: List[str] = []
            ref_prog_ints: List[int] = []

            # for each pair in demo
            for pair_idx in range(self.config.sampling.pairs_per_demo):
                if self.config.reference.resample_every > 0 and (pair_idx % self.config.reference.resample_every == 0):
                    num_ref_frames_this = sample_num_ref_frames(
                        self.config.reference.frames,
                        self.config.reference.frames_min,
                        self.config.reference.frames_max,
                        self.config.reference.frames_std,
                    )
                    ref_img_paths, ref_prog_ints = sample_reference_multiview_frames(
                        root=self.root,
                        task_name=task_name,
                        valid_task_demos=valid_task_demos_for_ref,
                        cur_demo_name=demo_name,
                        num_ref_frames=num_ref_frames_this,
                        ref_jitter=self.config.reference.jitter,
                        reference_views=self.config.reference.views,
                    )

                pair = sample_pair_indices(
                    T,
                    self.config.sampling.max_delta_t,
                    self.config.sampling.min_delta_t,
                    self.config.sampling.peak_distance,
                    self.config.sampling.rise_factor,
                    self.config.sampling.decay_factor,
                )
                if pair is None:
                    continue
                i, j = pair

                # 直接使用采样到的 (i, j) 对，不再生成四元组
                target_paths_t1: List[str] = []
                target_paths_t2: List[str] = []
                per_view_diffs: List[float] = []

                for v in self.required_views:
                    v_path = os.path.join(demo_path, v)
                    frames_v = view_to_frames[v]
                    frame_i_name = frames_v[i]
                    frame_j_name = frames_v[j]
                    img_abs_1 = os.path.join(v_path, frame_i_name)
                    img_abs_2 = os.path.join(v_path, frame_j_name)
                    target_paths_t1.append(img_abs_1)
                    target_paths_t2.append(img_abs_2)

                    if self.config.filtering.static_diff_threshold > 0:
                        diff = compute_mean_pixel_diff(img_abs_1, img_abs_2)
                        per_view_diffs.append(diff)

                delta_progress_int = compute_delta_progress_label_int(i, j, T)
                if self.config.filtering.static_diff_threshold > 0 and per_view_diffs:
                    if all(d < self.config.filtering.static_diff_threshold for d in per_view_diffs):
                        delta_progress_int = 0

                if random.random() < self.config.filtering.mismatch_prob and len(all_task_names) > 1:
                    mismatch_candidates = [t for t in all_task_names if t != task_name]
                    used_task_desc = task_desc_map[random.choice(mismatch_candidates)]
                    delta_progress_int = 0
                else:
                    used_task_desc = task_desc

                images, human_str = build_prompt_with_reference_multiview(
                    ref_img_paths=ref_img_paths,
                    ref_progress_ints=ref_prog_ints,
                    target_img_paths_t1=target_paths_t1,
                    target_img_paths_t2=target_paths_t2,
                    reference_view_names=self.config.reference.views,
                    target_view_names=self.required_views,
                    task_desc=used_task_desc
                )
                
                assistant_answer = f"{delta_progress_int:+d}"
                images_rel = [abs_to_rel_path(self.root, img) for img in images]
                data_sample = build_qwen_data_sample(images_rel, human_str, assistant_answer)
                samples.append(data_sample)

        logger.info(f"Total {split_name} samples for task {task_name}: {len(samples)}")
        return samples


    def split_tasks(self, task_desc_map: Dict[str, str]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """划分训练和验证任务"""
        train_ratio = self.config.split.train_ratio
        min_demos_per_task = self.config.split.min_demos_per_task
        selected_tasks = self.config.split.selected_tasks
        
        # 获取所有任务目录
        task_dirs = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        
        # 根据 selected_tasks 过滤任务
        if selected_tasks is None:
            tasks_to_process = task_dirs
            logger.info(f"Processing all tasks (total: {len(tasks_to_process)})")
        else:
            tasks_to_process = selected_tasks
            logger.info(f"Processing selected tasks: {tasks_to_process} (total: {len(tasks_to_process)})")
        
        train_tasks: Dict[str, List[str]] = {}
        eval_tasks: Dict[str, List[str]] = {}
        
        for task_name in tasks_to_process:
            task_path = os.path.join(self.root, task_name)
            demo_names = list_subdirs(task_path)
            num_demos = len(demo_names)
            num_train_demos = int(num_demos * train_ratio)
            sorted_demos = sorted(demo_names)
            train_tasks[task_name] = sorted_demos[:num_train_demos]
            eval_tasks[task_name] = sorted_demos[num_train_demos:]
            logger.info(
                f"Task {task_name}: {num_demos} demos total, "
                f"using first {num_train_demos} ({train_ratio*100:.0f}%) for training, "
                f"leaving {num_demos - num_train_demos} ({(1-train_ratio)*100:.0f}%) for validation."
            )
        return train_tasks, eval_tasks


def _process_single_task_worker(
    worker_idx: int,
    config: SimpleNamespace,
    task_name: str,
    train_demo_names: List[str],
    eval_demo_names: List[str],
    task_desc_map: Dict[str, str],
    all_task_names: List[str],
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """多进程 worker：对单个 task 生成 train / eval 两个 split 的样本"""
    random.seed(config.seed + worker_idx)
    builder = DatasetBuilder(config)
    valid_task_demos_for_ref = {task_name: train_demo_names}

    train_samples = builder.generate_samples_for_task_split(
        split_name="train",
        task_name=task_name,
        demo_names=train_demo_names,
        task_desc_map=task_desc_map,
        all_task_names=all_task_names,
        valid_task_demos_for_ref=valid_task_demos_for_ref,
    )

    eval_samples = builder.generate_samples_for_task_split(
        split_name="eval",
        task_name=task_name,
        demo_names=eval_demo_names,
        task_desc_map=task_desc_map,
        all_task_names=all_task_names,
        valid_task_demos_for_ref=valid_task_demos_for_ref,
    )

    return task_name, train_samples, eval_samples


# ==================== 主流程 ====================

def main(args):
    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    
    if args.root:
        config_dict["root"] = args.root
    
    config = dict_to_namespace(config_dict)
    random.seed(config.seed)

    desc_path = os.path.join(config.root, "task_descriptions.json")
    with open(desc_path, "r", encoding="utf-8") as f:
        task_desc_map: Dict[str, str] = json.load(f)

    builder = DatasetBuilder(config)
    train_tasks, eval_tasks = builder.split_tasks(task_desc_map)

    all_task_names = list(train_tasks.keys())
    logger.info(f"Processing {len(train_tasks)} tasks with >={config.split.min_demos_per_task} demos")
    logger.info("-" * 60)

    # 多进程模式
    if config.per_task:
        os.makedirs(config.train_output_dir, exist_ok=True)
        if config.eval_output_dir:
            os.makedirs(config.eval_output_dir, exist_ok=True)

        num_workers = config.num_workers or os.cpu_count() or 1
        logger.info(f"Using per-task multiprocessing with {num_workers} workers for {len(train_tasks)} tasks.")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _process_single_task_worker,
                    worker_idx,
                    config,
                    task_name,
                train_tasks[task_name],
                eval_tasks[task_name],
                    task_desc_map,
                    all_task_names,
                )
                for worker_idx, task_name in enumerate(sorted(train_tasks.keys()))
            ]

            for fut in as_completed(futures):
                task_name, train_samples, eval_samples = fut.result()
                train_json_path = os.path.join(config.train_output_dir, f"{task_name}.json")
                with open(train_json_path, "w", encoding="utf-8") as f:
                    json.dump(train_samples, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved train JSON for task {task_name} to {train_json_path}")

                if config.eval_output_dir and eval_samples:
                    eval_json_path = os.path.join(config.eval_output_dir, f"{task_name}_eval.json")
                    with open(eval_json_path, "w", encoding="utf-8") as f:
                        json.dump(eval_samples, f, ensure_ascii=False, indent=2)
                    logger.info(f"Saved eval JSON for task {task_name} to {eval_json_path}")
        return

    # 单进程模式
    train_samples_all: List[Dict[str, Any]] = []
    eval_samples_all: List[Dict[str, Any]] = []

    for idx, task_name in enumerate(sorted(train_tasks.keys())):
        logger.info(f"(sequential) Generating samples for task {idx+1}/{len(train_tasks)}: {task_name}")
        train_demos = train_tasks[task_name]
        eval_demos = eval_tasks.get(task_name, [])

        train_samples = builder.generate_samples_for_task_split(
            split_name="train",
            task_name=task_name,
            demo_names=train_demos,
            task_desc_map=task_desc_map,
            all_task_names=all_task_names,
            valid_task_demos_for_ref={task_name: train_demos},
        )
        train_samples_all.extend(train_samples)

        eval_samples = builder.generate_samples_for_task_split(
            split_name="eval",
            task_name=task_name,
            demo_names=eval_demos,
            task_desc_map=task_desc_map,
            all_task_names=all_task_names,
            valid_task_demos_for_ref={task_name: train_demos},
        )
        eval_samples_all.extend(eval_samples)

    logger.info(f"Total train samples (all tasks): {len(train_samples_all)}")
    with open(config.train_output, "w", encoding="utf-8") as f:
        json.dump(train_samples_all, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved train set to {config.train_output}")

    if config.eval_output:
        logger.info(f"Total eval samples (all tasks): {len(eval_samples_all)}")
        with open(config.eval_output, "w", encoding="utf-8") as f:
            json.dump(eval_samples_all, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved eval set to {config.eval_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Qwen-style JSON dataset for VLAC-style delta-progress critic")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")
    parser.add_argument("--root", type=str, help="Root of dataset (overrides YAML config).")

    args = parser.parse_args()
    main(args)
