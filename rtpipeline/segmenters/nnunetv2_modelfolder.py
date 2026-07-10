from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np


def _parse_predict_args(args: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-chk", default="checkpoint_final.pth")
    parser.add_argument("-device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("-step_size", type=float, default=0.5)
    parser.add_argument("-npp", type=int, default=0)
    parser.add_argument("-nps", type=int, default=0)
    parser.add_argument("--disable_tta", action="store_true")
    parser.add_argument("--save_probabilities", action="store_true")
    parser.add_argument("--continue_prediction", "--c", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--disable_progress_bar", action="store_true")
    parser.add_argument("-prev_stage_predictions", default=None)
    parsed, unknown = parser.parse_known_args(list(args or []))
    if unknown:
        raise ValueError(f"Unsupported nnUNetv2 modelfolder predict_args: {' '.join(unknown)}")
    return parsed


def _folds_as_list(folds: object) -> list[int | str] | None:
    if folds is None:
        return None
    if isinstance(folds, str):
        if folds.strip().lower() == "all":
            return ["all"]
        return [int(token) for token in folds.replace(",", " ").split()]
    if isinstance(folds, (list, tuple, set)):
        out: list[int | str] = []
        for fold in folds:
            out.append(fold if str(fold) == "all" else int(fold))
        return out
    return [int(folds)]


def _serial_preprocessing_iterator_fromfiles(
    list_of_lists: list[list[str]],
    list_of_segs_from_prev_stage_files: list[str] | None,
    output_filenames_truncated: list[str] | None,
    plans_manager,
    dataset_json: dict,
    configuration_manager,
    verbose: bool = False,
):
    import torch
    from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot

    label_manager = plans_manager.get_label_manager(dataset_json)
    preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
    for idx, files in enumerate(list_of_lists):
        seg_prev_stage = (
            list_of_segs_from_prev_stage_files[idx]
            if list_of_segs_from_prev_stage_files is not None
            else None
        )
        data, seg, data_properties = preprocessor.run_case(
            files,
            seg_prev_stage,
            plans_manager,
            configuration_manager,
            dataset_json,
        )
        if seg_prev_stage is not None:
            seg_onehot = convert_labelmap_to_one_hot(seg[0], label_manager.foreground_labels, data.dtype)
            data = np.vstack((data, seg_onehot))
        data = torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format)
        yield {
            "data": data,
            "data_properties": data_properties,
            "ofile": output_filenames_truncated[idx] if output_filenames_truncated is not None else None,
        }


def _predict_from_data_iterator_serial(predictor, data_iterator, save_probabilities: bool = False):
    import os as _os
    import torch
    from nnunetv2.inference.export_prediction import (
        convert_predicted_logits_to_segmentation_with_correct_shape,
        export_prediction_from_logits,
    )
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian
    from nnunetv2.utilities.helpers import empty_cache

    ret = []
    for preprocessed in data_iterator:
        data = preprocessed["data"]
        if isinstance(data, str):
            delfile = data
            data = torch.from_numpy(np.load(data))
            _os.remove(delfile)

        ofile = preprocessed["ofile"]
        if ofile is not None:
            print(f"\nPredicting {_os.path.basename(ofile)}:")
        else:
            print(f"\nPredicting image of shape {data.shape}:")
        print(f"perform_everything_on_device: {predictor.perform_everything_on_device}")

        properties = preprocessed["data_properties"]
        prediction = predictor.predict_logits_from_preprocessed_data(data).cpu().detach().numpy()
        class_ids, class_counts = np.unique(np.argmax(prediction, axis=0), return_counts=True)
        print(
            "nnUNet v2 logits argmax voxel counts: "
            + ", ".join(f"class_{int(cls)}={int(count)}" for cls, count in zip(class_ids, class_counts))
        )
        if ofile is not None:
            print("exporting prediction")
            ret.append(
                export_prediction_from_logits(
                    prediction,
                    properties,
                    predictor.configuration_manager,
                    predictor.plans_manager,
                    predictor.dataset_json,
                    ofile,
                    save_probabilities,
                )
            )
            print(f"done with {_os.path.basename(ofile)}")
        else:
            print("resampling prediction")
            ret.append(
                convert_predicted_logits_to_segmentation_with_correct_shape(
                    prediction,
                    predictor.plans_manager,
                    predictor.configuration_manager,
                    predictor.label_manager,
                    properties,
                    save_probabilities,
                )
            )
            print(f"\nDone with image of shape {data.shape}:")

    compute_gaussian.cache_clear()
    empty_cache(predictor.device)
    return ret


def _predict_from_files_serial(
    predictor,
    input_dir: str,
    output_dir: str,
    *,
    save_probabilities: bool,
    overwrite: bool,
    folder_with_segs_from_prev_stage: str | None,
) -> None:
    if predictor.configuration_manager.previous_stage_name is not None:
        assert folder_with_segs_from_prev_stage is not None, (
            "The requested configuration is a cascaded network. It requires the segmentations of the previous "
            f"stage ({predictor.configuration_manager.previous_stage_name}) as input."
        )
    list_of_lists, output_filename_truncated, seg_from_prev_stage_files = (
        predictor._manage_input_and_output_lists(
            input_dir,
            output_dir,
            folder_with_segs_from_prev_stage,
            overwrite,
            0,
            1,
            save_probabilities,
        )
    )
    if len(list_of_lists) == 0:
        return
    data_iterator = _serial_preprocessing_iterator_fromfiles(
        list_of_lists,
        seg_from_prev_stage_files,
        output_filename_truncated,
        predictor.plans_manager,
        predictor.dataset_json,
        predictor.configuration_manager,
        predictor.verbose_preprocessing,
    )
    _predict_from_data_iterator_serial(predictor, data_iterator, save_probabilities)


@contextmanager
def _patched_trainer_lookup(extra_trainers: Sequence[str] | None) -> Iterator[None]:
    names = {name for name in (extra_trainers or []) if name}
    if not names:
        yield
        return

    from nnunetv2.inference import predict_from_raw_data
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    original = predict_from_raw_data.recursive_find_python_class

    def patched(folder: str, class_name: str, current_module: str):
        found = original(folder, class_name, current_module)
        if found is not None:
            return found
        if class_name in names:
            return type(class_name, (nnUNetTrainer,), {})
        return None

    predict_from_raw_data.recursive_find_python_class = patched
    try:
        yield
    finally:
        predict_from_raw_data.recursive_find_python_class = original


@contextmanager
def _temporary_env(overrides: Mapping[str, str]) -> Iterator[None]:
    """Apply ``overrides`` to os.environ, restoring the prior values (or absence)
    of each affected key on exit.

    Without this, a reused pool worker process leaks environment mutations from
    one call into the next (e.g. a stale ``nnUNet_results``/``nnUNet_def_n_proc``
    from a previous model's config bleeding into a later prediction).
    """
    prior = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_modelfolder_prediction(
    *,
    model_folder: Path,
    input_dir: Path,
    output_dir: Path,
    folds: object,
    predict_args: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    trainer_shims: Sequence[str] | None = None,
) -> None:
    """Run nnU-Net v2 modelfolder inference with optional inference-only trainer shims."""
    overrides = {"nnUNet_def_n_proc": os.environ.get("nnUNet_def_n_proc", "1")}
    if env:
        overrides.update({str(key): str(value) for key, value in env.items()})

    with _temporary_env(overrides):
        args = _parse_predict_args(predict_args)

        import torch
        from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        output_dir.mkdir(parents=True, exist_ok=True)
        maybe_mkdir_p(str(output_dir))

        if args.device == "cpu":
            import multiprocessing

            torch.set_num_threads(multiprocessing.cpu_count())
            device = torch.device("cpu")
        elif args.device == "cuda":
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
            device = torch.device("cuda")
        else:
            device = torch.device("mps")

        predictor = nnUNetPredictor(
            tile_step_size=args.step_size,
            use_gaussian=True,
            use_mirroring=not args.disable_tta,
            perform_everything_on_device=True,
            device=device,
            verbose=args.verbose,
            allow_tqdm=not args.disable_progress_bar,
            verbose_preprocessing=args.verbose,
        )
        with _patched_trainer_lookup(trainer_shims):
            predictor.initialize_from_trained_model_folder(str(model_folder), _folds_as_list(folds), args.chk)
        if args.npp <= 0 and args.nps <= 0:
            print("Running nnUNet v2 modelfolder inference serially (pin_memory disabled).")
            _predict_from_files_serial(
                predictor,
                str(input_dir),
                str(output_dir),
                save_probabilities=args.save_probabilities,
                overwrite=not args.continue_prediction,
                folder_with_segs_from_prev_stage=args.prev_stage_predictions,
            )
        else:
            predictor.predict_from_files(
                str(input_dir),
                str(output_dir),
                save_probabilities=args.save_probabilities,
                overwrite=not args.continue_prediction,
                num_processes_preprocessing=max(1, args.npp),
                num_processes_segmentation_export=max(1, args.nps),
                folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                num_parts=1,
                part_id=0,
            )
