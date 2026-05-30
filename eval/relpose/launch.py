import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import math
import cv2
import numpy as np
import torch
import argparse

from copy import deepcopy
from eval.relpose.metadata import dataset_metadata
from eval.relpose.utils import *

from accelerate import PartialState
from add_ckpt_path import add_path_to_dust3r

from tqdm import tqdm
import time

def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        help="path to the model weights",
        default="",
    )

    parser.add_argument("--device", type=str, default="cuda", help="pytorch device")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument(
        "--no_crop", type=bool, default=True, help="whether to crop input data"
    )

    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="sintel",
        choices=list(dataset_metadata.keys()),
    )
    parser.add_argument("--size", type=int, default="224")

    parser.add_argument(
        "--model_update_type",
        type=str,
        default="cut3r",
        help="model type for state update strategy: cut3r, ttt3r, ttt3r_joint, etc.",
    )
    # DDD3R unified parameters (paper notation)
    parser.add_argument("--alpha", type=float, default=0.5, help="DDD3R constant dampening rate")
    parser.add_argument("--alpha_perp", type=float, default=0.5, help="DDD3R novel component coefficient")
    parser.add_argument("--alpha_parallel", type=float, default=0.05, help="DDD3R drift component coefficient")
    parser.add_argument("--beta_ema", type=float, default=0.95, help="DDD3R EMA decay for drift direction")
    parser.add_argument("--gamma", type=float, default=0.0, help="DDD3R steep adaptive exponent (0=fixed, >0=drift-adaptive)")
    parser.add_argument("--brake_tau", type=float, default=2.0, help="DDD3R brake temperature")
    parser.add_argument("--warmup_t0", type=int, default=0, help="DDD3R: no drift suppression for first T0 frames")
    parser.add_argument("--warmup_window", type=int, default=0, help="DDD3R: linear ramp window after T0")
    # Keep for abandoned methods
    parser.add_argument("--spectral_temperature", type=float, default=1.0, help="Layer 2 SIASU temperature")
    parser.add_argument("--geo_gate_tau", type=float, default=2.0, help="Layer 3 geo gate temperature")
    parser.add_argument("--geo_gate_freq_cutoff", type=int, default=4, help="Layer 3 geo gate freq cutoff denominator")
    # Backward-compat aliases (hidden)
    parser.add_argument("--random_gate_p", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ortho_alpha_novel", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ortho_alpha_drift", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ortho_beta", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ortho_gamma", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ortho_adaptive", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--momentum_tau", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ortho_warmup_t0", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ortho_warmup_window", type=int, default=None, help=argparse.SUPPRESS)

    parser.add_argument(
        "--pose_eval_stride", default=1, type=int, help="stride for pose evaluation"
    )
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument(
        "--full_seq",
        action="store_true",
        default=False,
        help="use full sequence for pose evaluation",
    )
    parser.add_argument(
        "--seq_list",
        nargs="+",
        default=None,
        help="list of sequences for pose evaluation",
    )

    parser.add_argument("--revisit", type=int, default=1)
    parser.add_argument("--freeze_state", action="store_true", default=False)
    parser.add_argument("--solve_pose", action="store_true", default=False)
    return parser


def _resolve_ddd3r_aliases(args):
    """Map old CLI arg names to new ones (backward compat)."""
    if args.random_gate_p is not None and args.alpha == 0.5:
        args.alpha = args.random_gate_p
    if args.ortho_alpha_novel is not None:
        args.alpha_perp = args.ortho_alpha_novel
    if args.ortho_alpha_drift is not None:
        args.alpha_parallel = args.ortho_alpha_drift
    if args.ortho_beta is not None:
        args.beta_ema = args.ortho_beta
    if args.ortho_gamma is not None:
        args.gamma = args.ortho_gamma
    if args.momentum_tau is not None:
        args.brake_tau = args.momentum_tau
    if args.ortho_warmup_t0 is not None:
        args.warmup_t0 = args.ortho_warmup_t0
    if args.ortho_warmup_window is not None:
        args.warmup_window = args.ortho_warmup_window
    # Legacy: ortho_adaptive="steep" → set gamma if not already set
    if args.ortho_adaptive == 'steep' and args.gamma == 0.0:
        args.gamma = args.ortho_gamma if args.ortho_gamma is not None else 2.0
    return args


def eval_pose_estimation(args, model, save_dir=None):
    metadata = dataset_metadata.get(args.eval_dataset)
    img_path = metadata["img_path"]
    mask_path = metadata["mask_path"]

    ate_mean, rpe_trans_mean, rpe_rot_mean = eval_pose_estimation_dist(
        args, model, save_dir=save_dir, img_path=img_path, mask_path=mask_path
    )
    return ate_mean, rpe_trans_mean, rpe_rot_mean


def eval_pose_estimation_dist(args, model, img_path, save_dir=None, mask_path=None):
    from dust3r.inference import inference, inference_recurrent, inference_recurrent_lighter

    metadata = dataset_metadata.get(args.eval_dataset)
    anno_path = metadata.get("anno_path", None)

    seq_list = args.seq_list
    if seq_list is None:
        if metadata.get("full_seq", False):
            args.full_seq = True
        else:
            seq_list = metadata.get("seq_list", [])
        if args.full_seq:
            seq_list = os.listdir(img_path)
            seq_list = [
                seq for seq in seq_list if os.path.isdir(os.path.join(img_path, seq))
            ]
        seq_list = sorted(seq_list)

    if save_dir is None:
        save_dir = args.output_dir

    distributed_state = PartialState()
    model.to(distributed_state.device)
    device = distributed_state.device

    with distributed_state.split_between_processes(seq_list) as seqs:
        ate_list = []
        rpe_trans_list = []
        rpe_rot_list = []
        load_img_size = args.size
        error_log_path = f"{save_dir}/_error_log_{distributed_state.process_index}.txt"  # Unique log file per process
        bug = False
        for seq in tqdm(seqs):
            try:
                dir_path = metadata["dir_path_func"](img_path, seq)

                # Handle skip_condition
                skip_condition = metadata.get("skip_condition", None)
                if skip_condition is not None and skip_condition(save_dir, seq):
                    continue

                mask_path_seq_func = metadata.get(
                    "mask_path_seq_func", lambda mask_path, seq: None
                )
                mask_path_seq = mask_path_seq_func(mask_path, seq)

                filelist = [
                    os.path.join(dir_path, name) for name in os.listdir(dir_path)
                ]
                filelist.sort()
                filelist = filelist[:: args.pose_eval_stride]

                views = prepare_input(
                    filelist,
                    [True for _ in filelist],
                    size=load_img_size,
                    crop=not args.no_crop,
                    revisit=args.revisit,
                    update=not args.freeze_state,
                )

                start = time.time()
                outputs, _ = inference_recurrent_lighter(views, model, device)
                end = time.time()
                fps = len(filelist) / (end - start)
                print(f"Finished pose estimation for {args.eval_dataset} {seq: <16}, FPS: {fps:.2f}")

                (
                    colors,
                    pts3ds_self,
                    pts3ds_other,
                    conf_self,
                    conf_other,
                    cam_dict,
                    pr_poses,
                ) = prepare_output(
                    outputs, revisit=args.revisit, solve_pose=args.solve_pose
                )

                pred_traj = get_tum_poses(pr_poses)
                os.makedirs(f"{save_dir}/{seq}", exist_ok=True)
                save_tum_poses(pr_poses, f"{save_dir}/{seq}/pred_traj.txt")
                save_focals(cam_dict, f"{save_dir}/{seq}/pred_focal.txt")
                save_intrinsics(cam_dict, f"{save_dir}/{seq}/pred_intrinsics.txt")
                # save_depth_maps(pts3ds_self,f'{save_dir}/{seq}', conf_self=conf_self)
                # save_conf_maps(conf_self,f'{save_dir}/{seq}')
                # save_rgb_imgs(colors,f'{save_dir}/{seq}')

                gt_traj_file = metadata["gt_traj_func"](img_path, anno_path, seq)
                traj_format = metadata.get("traj_format", None)

                if args.eval_dataset == "sintel":
                    gt_traj = load_traj(
                        gt_traj_file=gt_traj_file, stride=args.pose_eval_stride
                    )
                elif traj_format is not None:
                    gt_traj = load_traj(
                        gt_traj_file=gt_traj_file,
                        traj_format=traj_format,
                        stride=args.pose_eval_stride,
                    )
                else:
                    gt_traj = None

                if gt_traj is not None:
                    ate, rpe_trans, rpe_rot = eval_metrics(
                        pred_traj,
                        gt_traj,
                        seq=seq,
                        filename=f"{save_dir}/{seq}_eval_metric.txt",
                    )
                    plot_trajectory(
                        pred_traj, gt_traj, title=seq, filename=f"{save_dir}/{seq}.png"
                    )
                else:
                    ate, rpe_trans, rpe_rot = 0, 0, 0
                    bug = True

                ate_list.append(ate)
                rpe_trans_list.append(rpe_trans)
                rpe_rot_list.append(rpe_rot)

                # Write to error log after each sequence
                with open(error_log_path, "a") as f:
                    f.write(
                        f"{args.eval_dataset}-{seq: <16} | ATE: {ate:.5f}, RPE trans: {rpe_trans:.5f}, RPE rot: {rpe_rot:.5f}\n"
                    )
                    f.write(f"{ate:.5f}\n")
                    f.write(f"{rpe_trans:.5f}\n")
                    f.write(f"{rpe_rot:.5f}\n")

            except Exception as e:
                if "out of memory" in str(e):
                    # Handle OOM
                    torch.cuda.empty_cache()  # Clear the CUDA memory
                    with open(error_log_path, "a") as f:
                        f.write(
                            f"OOM error in sequence {seq}, skipping this sequence.\n"
                        )
                    print(f"OOM error in sequence {seq}, skipping...")
                elif "Degenerate covariance rank" in str(
                    e
                ) or "Eigenvalues did not converge" in str(e):
                    # Handle Degenerate covariance rank exception and Eigenvalues did not converge exception
                    with open(error_log_path, "a") as f:
                        f.write(f"Exception in sequence {seq}: {str(e)}\n")
                    print(f"Traj evaluation error in sequence {seq}, skipping.")
                else:
                    raise e  # Rethrow if it's not an expected exception

    distributed_state.wait_for_everyone()

    results = process_directory(save_dir)
    avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results)

    # Write the averages to the error log (only on the main process)
    if distributed_state.is_main_process:
        with open(f"{save_dir}/_error_log.txt", "a") as f:
            # Copy the error log from each process to the main error log
            for i in range(distributed_state.num_processes):
                if not os.path.exists(f"{save_dir}/_error_log_{i}.txt"):
                    break
                with open(f"{save_dir}/_error_log_{i}.txt", "r") as f_sub:
                    f.write(f_sub.read())
            f.write(
                f"Average ATE: {avg_ate:.5f}, Average RPE trans: {avg_rpe_trans:.5f}, Average RPE rot: {avg_rpe_rot:.5f}\n"
            )

    return avg_ate, avg_rpe_trans, avg_rpe_rot


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    _resolve_ddd3r_aliases(args)
    add_path_to_dust3r(args.weights)
    from dust3r.utils.image import load_images_for_eval as load_images
    from dust3r.post_process import estimate_focal_knowing_depth
    from dust3r.model import ARCroco3DStereo
    from dust3r.utils.camera import pose_encoding_to_camera
    from dust3r.utils.geometry import weighted_procrustes, geotrf

    args.full_seq = False
    args.no_crop = False

    def recover_cam_params(pts3ds_self, pts3ds_other, conf_self, conf_other):
        B, H, W, _ = pts3ds_self.shape
        pp = (
            torch.tensor([W // 2, H // 2], device=pts3ds_self.device)
            .float()
            .repeat(B, 1)
            .reshape(B, 1, 2)
        )
        focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

        pts3ds_self = pts3ds_self.reshape(B, -1, 3)
        pts3ds_other = pts3ds_other.reshape(B, -1, 3)
        conf_self = conf_self.reshape(B, -1)
        conf_other = conf_other.reshape(B, -1)
        # weighted procrustes
        c2w = weighted_procrustes(
            pts3ds_self,
            pts3ds_other,
            torch.log(conf_self) * torch.log(conf_other),
            use_weights=True,
            return_T=True,
        )
        return c2w, focal, pp.reshape(B, 2)

    def prepare_input(
        img_paths,
        img_mask,
        size,
        raymaps=None,
        raymap_mask=None,
        revisit=1,
        update=True,
        crop=True,
    ):
        images = load_images(img_paths, size=size, crop=crop, verbose=False)
        views = []
        if raymaps is None and raymap_mask is None:
            num_views = len(images)

            for i in range(num_views):
                view = {
                    "img": images[i]["img"],
                    "ray_map": torch.full(
                        (
                            images[i]["img"].shape[0],
                            6,
                            images[i]["img"].shape[-2],
                            images[i]["img"].shape[-1],
                        ),
                        torch.nan,
                    ),
                    "true_shape": torch.from_numpy(images[i]["true_shape"]),
                    "idx": i,
                    "instance": str(i),
                    "camera_pose": torch.from_numpy(
                        np.eye(4).astype(np.float32)
                    ).unsqueeze(0),
                    "img_mask": torch.tensor(True).unsqueeze(0),
                    "ray_mask": torch.tensor(False).unsqueeze(0),
                    "update": torch.tensor(True).unsqueeze(0),
                    "reset": torch.tensor(False).unsqueeze(0),
                }
                views.append(view)
        else:

            num_views = len(images) + len(raymaps)
            assert len(img_mask) == len(raymap_mask) == num_views
            assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)

            j = 0
            k = 0
            for i in range(num_views):
                view = {
                    "img": (
                        images[j]["img"]
                        if img_mask[i]
                        else torch.full_like(images[0]["img"], torch.nan)
                    ),
                    "ray_map": (
                        raymaps[k]
                        if raymap_mask[i]
                        else torch.full_like(raymaps[0], torch.nan)
                    ),
                    "true_shape": (
                        torch.from_numpy(images[j]["true_shape"])
                        if img_mask[i]
                        else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))
                    ),
                    "idx": i,
                    "instance": str(i),
                    "camera_pose": torch.from_numpy(
                        np.eye(4).astype(np.float32)
                    ).unsqueeze(0),
                    "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                    "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                    "update": torch.tensor(img_mask[i]).unsqueeze(0),
                    "reset": torch.tensor(False).unsqueeze(0),
                }
                if img_mask[i]:
                    j += 1
                if raymap_mask[i]:
                    k += 1
                views.append(view)
            assert j == len(images) and k == len(raymaps)

        if revisit > 1:
            # repeat input for 'revisit' times
            new_views = []
            for r in range(revisit):
                for i in range(len(views)):
                    new_view = deepcopy(views[i])
                    new_view["idx"] = r * len(views) + i
                    new_view["instance"] = str(r * len(views) + i)
                    if r > 0:
                        if not update:
                            new_view["update"] = torch.tensor(False).unsqueeze(0)
                    new_views.append(new_view)
            return new_views
        return views

    def prepare_output(outputs, revisit=1, solve_pose=False):
        valid_length = len(outputs["pred"]) // revisit
        outputs["pred"] = outputs["pred"][-valid_length:]
        outputs["views"] = outputs["views"][-valid_length:]

        if solve_pose:
            pts3ds_self = [
                output["pts3d_in_self_view"].cpu() for output in outputs["pred"]
            ]
            pts3ds_other = [
                output["pts3d_in_other_view"].cpu() for output in outputs["pred"]
            ]
            conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
            conf_other = [output["conf"].cpu() for output in outputs["pred"]]
            pr_poses, focal, pp = recover_cam_params(
                torch.cat(pts3ds_self, 0),
                torch.cat(pts3ds_other, 0),
                torch.cat(conf_self, 0),
                torch.cat(conf_other, 0),
            )
            pts3ds_self = torch.cat(pts3ds_self, 0)
        else:

            pts3ds_self = [
                output["pts3d_in_self_view"].cpu() for output in outputs["pred"]
            ]
            pts3ds_other = [
                output["pts3d_in_other_view"].cpu() for output in outputs["pred"]
            ]
            conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
            conf_other = [output["conf"].cpu() for output in outputs["pred"]]
            pts3ds_self = torch.cat(pts3ds_self, 0)
            pr_poses = [
                pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
                for pred in outputs["pred"]
            ]
            pr_poses = torch.cat(pr_poses, 0)

            B, H, W, _ = pts3ds_self.shape
            pp = (
                torch.tensor([W // 2, H // 2], device=pts3ds_self.device)
                .float()
                .repeat(B, 1)
                .reshape(B, 2)
            )
            focal = estimate_focal_knowing_depth(
                pts3ds_self, pp, focal_mode="weiszfeld"
            )

        colors = [0.5 * (output["rgb"][0] + 1.0) for output in outputs["pred"]]
        cam_dict = {
            "focal": focal.cpu().numpy(),
            "pp": pp.cpu().numpy(),
        }
        return (
            colors,
            pts3ds_self,
            pts3ds_other,
            conf_self,
            conf_other,
            cam_dict,
            pr_poses,
        )

    model = ARCroco3DStereo.from_pretrained(args.weights)

    # set model type and DDD3R hyperparameters
    model.config.model_update_type = args.model_update_type
    model.config.alpha = args.alpha
    model.config.alpha_perp = args.alpha_perp
    model.config.alpha_parallel = args.alpha_parallel
    model.config.beta_ema = args.beta_ema
    model.config.gamma = args.gamma
    model.config.brake_tau = args.brake_tau
    model.config.warmup_t0 = args.warmup_t0
    model.config.warmup_window = args.warmup_window
    # Keep for abandoned methods
    model.config.spectral_temperature = args.spectral_temperature
    model.config.geo_gate_tau = args.geo_gate_tau
    model.config.geo_gate_freq_cutoff = args.geo_gate_freq_cutoff

    eval_pose_estimation(args, model, save_dir=args.output_dir)


def apply_ddd3r_config(args, model):
    """Apply DDD3R hyperparameters from args to model config.
    Shared by relpose and mv_recon evaluation pipelines.
    """
    model.config.model_update_type = args.model_update_type
    model.config.alpha = getattr(args, 'alpha', 0.5)
    model.config.alpha_perp = getattr(args, 'alpha_perp', 0.5)
    model.config.alpha_parallel = getattr(args, 'alpha_parallel', 0.05)
    model.config.beta_ema = getattr(args, 'beta_ema', 0.95)
    model.config.gamma = getattr(args, 'gamma', 0.0)
    model.config.brake_tau = getattr(args, 'brake_tau', 2.0)
    model.config.warmup_t0 = getattr(args, 'warmup_t0', 0)
    model.config.warmup_window = getattr(args, 'warmup_window', 0)
    model.config.spectral_temperature = getattr(args, 'spectral_temperature', 1.0)
    model.config.geo_gate_tau = getattr(args, 'geo_gate_tau', 2.0)
    model.config.geo_gate_freq_cutoff = getattr(args, 'geo_gate_freq_cutoff', 4)
    # Auto-gamma (for mv_recon experiments)
    model.config.auto_gamma = getattr(args, 'auto_gamma', '')
    model.config.auto_gamma_warmup = getattr(args, 'auto_gamma_warmup', 30)
    model.config.auto_gamma_max = getattr(args, 'auto_gamma_max', 3.0)
    model.config.auto_gamma_k = getattr(args, 'auto_gamma_k', 10.0)
    model.config.auto_gamma_lo = getattr(args, 'auto_gamma_lo', 0.3)
    model.config.auto_gamma_hi = getattr(args, 'auto_gamma_hi', 0.6)
