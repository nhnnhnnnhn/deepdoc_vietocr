#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
import copy
import time
import os
import re
from dataclasses import dataclass

from huggingface_hub import snapshot_download

from utils.file_utils import get_project_base_directory
from utils.settings import PARALLEL_DEVICES
from .operators import *  # noqa: F403
from . import operators
import math
import numpy as np
import cv2
import onnxruntime as ort
import torch
import matplotlib.pyplot as plt
from PIL import Image
import yaml

from .postprocess import build_post_process

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

loaded_models = {}

_GPU_PROVIDERS = (
    "CUDAExecutionProvider",
    "MIGraphXExecutionProvider",
    "ROCMExecutionProvider",
)


def _find_gpu_onnx_provider():
    available = set(ort.get_available_providers())
    for ep in _GPU_PROVIDERS:
        if ep in available:
            return ep
    return None


@dataclass(frozen=True)
class RuntimeDevice:
    requested: str
    mode: str
    device_id: int | None
    device_id_explicit: bool
    onnx_provider: str
    torch_device: str
    fallback_reason: str = ""

    @property
    def is_gpu(self):
        return self.onnx_provider in _GPU_PROVIDERS

    @property
    def is_cuda(self):
        return self.is_gpu

    @property
    def cache_key(self):
        if self.is_gpu:
            return f"gpu:{self.device_id}"
        return "cpu"

    def describe(self):
        if self.is_gpu:
            ep_short = self.onnx_provider.replace("ExecutionProvider", "")
            return f"cuda:{self.device_id} (ONNX Runtime {ep_short}, VietOCR {self.torch_device})"
        if self.fallback_reason:
            return f"cpu ({self.fallback_reason})"
        return "cpu"


def _parse_device(device):
    requested = "auto" if device is None else str(device).strip().lower()
    if requested in ("auto", "cpu"):
        return requested, None, False, requested

    match = re.fullmatch(r"(?:cuda|rocm)(?::(\d+))?", requested)
    if match:
        device_id = int(match.group(1)) if match.group(1) is not None else None
        return "cuda", device_id, device_id is not None, requested

    raise ValueError("Unsupported device '{}'. Use auto, cpu, cuda, cuda:<id>, rocm, or rocm:<id>.".format(device))


def _cuda_unavailable_reason(device_id):
    reasons = []
    try:
        torch_cuda = torch.cuda.is_available()
        torch_device_count = torch.cuda.device_count() if torch_cuda else 0
    except Exception as exc:
        torch_cuda = False
        torch_device_count = 0
        reasons.append(f"PyTorch CUDA/ROCm check failed: {exc}")

    if not torch_cuda:
        reasons.append("PyTorch CUDA/ROCm is not available")
    elif torch_device_count <= device_id:
        reasons.append(f"GPU device {device_id} is not available; PyTorch sees {torch_device_count} device(s)")
    else:
        is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
        if not is_rocm:
            try:
                major, minor = torch.cuda.get_device_capability(device_id)
                device_sm = major * 10 + minor
                arch_list = [a for a in getattr(torch.cuda, "get_arch_list", list)() if re.fullmatch(r"sm_\d+", a)]
                if arch_list:
                    min_supported = min(int(a[3:]) for a in arch_list)
                    if device_sm < min_supported:
                        reasons.append(
                            f"GPU SM {major}.{minor} is below the minimum SM "
                            f"{min_supported // 10}.{min_supported % 10} required by this PyTorch build "
                            f"(supported: {arch_list})"
                        )
            except Exception as exc:
                logging.warning("CUDA compute capability check failed for device %d: %s", device_id, exc)

    gpu_provider = _find_gpu_onnx_provider()
    if gpu_provider is None:
        ort_providers = ort.get_available_providers()
        reasons.append(f"No GPU ONNX Runtime provider available; providers: {ort_providers}")

    return "; ".join(reasons)


def resolve_device(device="auto", device_id: int | None = None):
    if isinstance(device, RuntimeDevice):
        return device

    mode, parsed_device_id, explicit_device_id, requested = _parse_device(device)
    selected_device_id = parsed_device_id if parsed_device_id is not None else (device_id if device_id is not None else 0)

    if mode == "cpu":
        return RuntimeDevice(
            requested=requested,
            mode=mode,
            device_id=None,
            device_id_explicit=explicit_device_id,
            onnx_provider="CPUExecutionProvider",
            torch_device="cpu",
        )

    unavailable_reason = _cuda_unavailable_reason(selected_device_id)
    if not unavailable_reason:
        gpu_provider = _find_gpu_onnx_provider()
        return RuntimeDevice(
            requested=requested,
            mode=mode,
            device_id=selected_device_id,
            device_id_explicit=explicit_device_id,
            onnx_provider=gpu_provider,
            torch_device=f"cuda:{selected_device_id}",
        )

    if mode == "cuda":
        raise RuntimeError(
            "GPU device was requested but cannot be used: {}".format(unavailable_reason)
        )

    logging.warning("GPU auto-selection is unavailable; falling back to CPU: %s", unavailable_reason)
    return RuntimeDevice(
        requested=requested,
        mode=mode,
        device_id=None,
        device_id_explicit=explicit_device_id,
        onnx_provider="CPUExecutionProvider",
        torch_device="cpu",
        fallback_reason=unavailable_reason,
    )


def _preload_onnx_cuda_dlls():
    preload_dlls = getattr(ort, "preload_dlls", None)
    if not callable(preload_dlls):
        return

    try:
        preload_dlls()
    except Exception as exc:
        logging.warning("onnxruntime.preload_dlls() failed: %s", exc)


def _load_vietocr_config(name):
    config_dir = os.path.join(get_project_base_directory(), "vietocr", "config")
    base_file = os.path.join(config_dir, "base.yml")
    model_file = os.path.join(config_dir, f"{name}.yml")

    with open(base_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(model_file, encoding="utf-8") as f:
        model_config = yaml.safe_load(f)

    config.update(model_config)
    return Cfg(config)

def transform(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(
        op_param_list, list), ('operator config should be a list')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = getattr(operators, op_name)(**param)
        ops.append(op)
    return ops


def load_model(model_dir, nm, device_id: int | None = None, device="auto"):
    runtime_device = resolve_device(device, device_id)
    model_file_path = os.path.join(model_dir, nm + ".onnx")
    model_cached_tag = f"{model_file_path}:{runtime_device.cache_key}"

    global loaded_models
    loaded_model = loaded_models.get(model_cached_tag)
    if loaded_model:
        logging.info(f"load_model {model_file_path} reuses cached model")
        return loaded_model

    if not os.path.exists(model_file_path):
        raise ValueError("not find model file path {}".format(
            model_file_path))

    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 2

    # https://github.com/microsoft/onnxruntime/issues/9509#issuecomment-951546580
    # Shrink GPU memory after execution
    run_options = ort.RunOptions()
    if runtime_device.is_gpu:
        ep = runtime_device.onnx_provider

        if ep == "CUDAExecutionProvider":
            _preload_onnx_cuda_dlls()
            provider_options = {
                "device_id": runtime_device.device_id,
                "gpu_mem_limit": 512 * 1024 * 1024,
                "arena_extend_strategy": "kNextPowerOfTwo",
            }
        elif ep == "MIGraphXExecutionProvider":
            provider_options = {
                "device_id": str(runtime_device.device_id),
            }
        else:
            provider_options = {
                "device_id": runtime_device.device_id,
                "gpu_mem_limit": 512 * 1024 * 1024,
                "arena_extend_strategy": "kNextPowerOfTwo",
            }

        sess = ort.InferenceSession(
            model_file_path,
            options=options,
            providers=[ep],
            provider_options=[provider_options],
        )

        if ep == "CUDAExecutionProvider":
            run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:" + str(runtime_device.device_id))

        logging.info(
            "load_model %s uses %s device %s; providers=%s",
            model_file_path,
            ep.replace("ExecutionProvider", ""),
            runtime_device.device_id,
            sess.get_providers(),
        )
    else:
        sess = ort.InferenceSession(
            model_file_path,
            options=options,
            providers=['CPUExecutionProvider'])
        run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu")
        logging.info("load_model %s uses CPU; providers=%s", model_file_path, sess.get_providers())
    loaded_model = (sess, run_options)
    loaded_models[model_cached_tag] = loaded_model
    return loaded_model

class TextRecognizer:
    def __init__(self, model_dir=None, device_id: int | None = None, device="auto"):
        runtime_device = resolve_device(device, device_id)
        
        #seq2seq
        config = _load_vietocr_config('vgg-seq2seq')
        config['weights'] = os.path.join(get_project_base_directory(), "vietocr", "weight", "vgg_seq2seq.pth")

        #transformer
        #config = Cfg.load_config_from_name('vgg_transformer')
        #config['weights'] = r"vietocr\weight\vgg_transformer.pth" 

        config['cnn']['pretrained'] = False
        config['device'] = runtime_device.torch_device
        self.device = runtime_device.torch_device
        self.detector = Predictor(config)
        logging.info("VietOCR recognizer uses %s", self.device)

    def __call__(self, img_list):
        results = []
        for img in img_list:
            # Ensure PIL Image
            if isinstance(img, np.ndarray):
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            text = self.detector.predict(img)
            results.append((text, 1.0))
        return results, 0.0


class TextDetector:
    def __init__(self, model_dir, device_id: int | None = None, device="auto"):
        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': 960,
                'limit_type': "max",
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        postprocess_params = {"name": "DBPostProcess", "thresh": 0.3, "box_thresh": 0.5, "max_candidates": 1000,
                              "unclip_ratio": 1.5, "use_dilation": False, "score_mode": "fast", "box_type": "quad"}

        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.run_options = load_model(model_dir, 'det', device_id, device=device)
        self.input_tensor = self.predictor.get_inputs()[0]

        img_h, img_w = self.input_tensor.shape[2:]
        if isinstance(img_h, str) or isinstance(img_w, str):
            pass
        elif img_h is not None and img_w is not None and img_h > 0 and img_w > 0:
            pre_process_list[0] = {
                'DetResizeForTest': {
                    'image_shape': [img_h, img_w]
                }
            }
        self.preprocess_op = create_operators(pre_process_list)

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if isinstance(box, list):
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if isinstance(box, list):
                box = np.array(box)
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}

        st = time.time()
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()
        input_dict = {}
        input_dict[self.input_tensor.name] = img
        for i in range(100000):
            try:
                outputs = self.predictor.run(None, input_dict, self.run_options)
                break
            except Exception as e:
                if i >= 3:
                    raise e
                time.sleep(5)

        post_result = self.postprocess_op({"maps": outputs[0]}, shape_list)
        dt_boxes = post_result[0]['points']
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)

        return dt_boxes, time.time() - st


class OCR:
    def __init__(self, model_dir=None, device="auto"):
        """
        If you have trouble downloading HuggingFace models, -_^ this might help!!

        For Linux:
        export HF_ENDPOINT=https://hf-mirror.com

        For Windows:
        Good luck
        ^_-

        """
        self.runtime_device = resolve_device(device)
        self.device_request = device

        if not model_dir:
            model_dir = os.path.join(
                    get_project_base_directory(),
                    "onnx")

        try:
            self._init_models(model_dir)
        except ValueError as exc:
            if "not find model file path" not in str(exc):
                raise
            model_dir = snapshot_download(repo_id="InfiniFlow/deepdoc",
                                          local_dir=os.path.join(get_project_base_directory(), "onnx"),
                                          local_dir_use_symlinks=False)
            self._init_models(model_dir)

        self.drop_score = 0.5
        self.crop_image_res_index = 0

    def _device_ids(self):
        runtime_device = self.runtime_device
        if not runtime_device.is_cuda:
            return [0]

        if runtime_device.mode == "auto" and not runtime_device.device_id_explicit and PARALLEL_DEVICES and PARALLEL_DEVICES > 1:
            return list(range(PARALLEL_DEVICES))

        return [runtime_device.device_id]

    def _device_spec(self, device_id):
        if not self.runtime_device.is_cuda:
            return "cpu"
        return f"cuda:{device_id}"

    def _init_models(self, model_dir):
        self.text_detector = []
        self.text_recognizer = []
        for device_id in self._device_ids():
            device_spec = self._device_spec(device_id)
            self.text_detector.append(TextDetector(model_dir, device_id, device=device_spec))
            self.text_recognizer.append(TextRecognizer(model_dir, device_id, device=device_spec))

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        assert len(points) == 4, "shape of points must be 4*2"
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def sorted_boxes(self, dt_boxes):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                        (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes

    def detect(self, img, device_id: int | None = None):
        if device_id is None:
            device_id = 0

        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}

        if img is None:
            return None, None, time_dict

        start = time.time()
        dt_boxes, elapse = self.text_detector[device_id](img)
        time_dict['det'] = elapse

        if dt_boxes is None:
            end = time.time()
            time_dict['all'] = end - start
            return None, None, time_dict

        return zip(self.sorted_boxes(dt_boxes), [
                   ("", 0) for _ in range(len(dt_boxes))])

    def recognize(self, ori_im, box, device_id: int | None = None):
        if device_id is None:
            device_id = 0

        img_crop = self.get_rotate_crop_image(ori_im, box)

        rec_res, elapse = self.text_recognizer[device_id]([img_crop])
        text, score = rec_res[0]
        if score < self.drop_score:
            return ""
        return text

    def recognize_batch(self, img_list, device_id: int | None = None):
        if device_id is None:
            device_id = 0
        rec_res, elapse = self.text_recognizer[device_id](img_list)
        texts = []
        for i in range(len(rec_res)):
            text, score = rec_res[i]
            if score < self.drop_score:
                text = ""
            texts.append(text)
        return texts

    def __call__(self, img, device_id = 0, cls=True):
        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}
        if device_id is None:
            device_id = 0

        if img is None:
            return None, None, time_dict

        start = time.time()
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector[device_id](img)
        time_dict['det'] = elapse

        if dt_boxes is None:
            end = time.time()
            time_dict['all'] = end - start
            return None, None, time_dict

        img_crop_list = []

        dt_boxes = self.sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        rec_res, elapse = self.text_recognizer[device_id](img_crop_list)

        time_dict['rec'] = elapse

        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        end = time.time()
        time_dict['all'] = end - start

        # for bno in range(len(img_crop_list)):
        #    print(f"{bno}, {rec_res[bno]}")

        return list(zip([a.tolist() for a in filter_boxes], filter_rec_res))
