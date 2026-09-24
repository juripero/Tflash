# Imports
import json
import os
import random
import sys
import time
from typing import Sequence, Mapping, Any, Union


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Return a sequence or mapping result item by index."""
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def get_comfyui_path() -> str:
    """Return the configured ComfyUI path, preferring COMFYUI_PATH when set."""
    comfyui_path = os.environ.get("COMFYUI_PATH")
    if comfyui_path:
        return comfyui_path
    return find_path("ComfyUI")


def find_path(name: str, path: str = None) -> str:
    """Recursively search parent folders until the named entry is found."""
    if path is None:
        path = os.getcwd()

    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    parent_directory = os.path.dirname(path)
    if parent_directory == path:
        return None

    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """Add the ComfyUI checkout to sys.path."""
    comfyui_path = get_comfyui_path()
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        if comfyui_path in sys.path:
            sys.path.remove(comfyui_path)
        sys.path.insert(0, comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """Load ComfyUI extra model paths configuration when available."""
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")
    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


def bootstrap_comfyui_runtime() -> None:
    """Mirror the allocator-related ComfyUI startup steps before torch import."""
    add_comfyui_directory_to_sys_path()

    import comfy.options

    comfy.options.enable_args_parsing()

    from comfy.cli_args import args

    if os.name == "nt":
        os.environ["MIMALLOC_PURGE_DELAY"] = "0"

    if args.default_device is not None:
        default_dev = args.default_device
        devices = list(range(32))
        devices.remove(default_dev)
        devices.insert(0, default_dev)
        devices = ",".join(map(str, devices))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)
        os.environ["HIP_VISIBLE_DEVICES"] = str(devices)

    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
        os.environ["HIP_VISIBLE_DEVICES"] = str(args.cuda_device)
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.cuda_device)

    if args.oneapi_device_selector is not None:
        os.environ["ONEAPI_DEVICE_SELECTOR"] = args.oneapi_device_selector

    if args.deterministic and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import cuda_malloc

    if "rocm" in cuda_malloc.get_torch_version_noimport():
        os.environ["OCL_SET_SVM_SIZE"] = "262144"


def cleanup_comfyui_runtime(unload_models: bool | None = None) -> None:
    """Best-effort cleanup for embedded or repeated generated-script execution."""
    import gc

    def run_cleanup_hook(name: str, should_run: bool = True) -> None:
        if not should_run or not hasattr(model_management, name):
            return
        cleanup_fn = getattr(model_management, name)
        try:
            cleanup_fn()
        except Exception as exc:
            warnings.warn(
                f"ComfyUI cleanup hook {name} failed during teardown: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    should_unload = unload_models
    if should_unload is None:
        should_unload = os.environ.get(
            "COMFYUI_TOPYTHON_UNLOAD_MODELS", ""
        ).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    try:
        import comfy.model_management as model_management
    except ModuleNotFoundError:
        gc.collect()
        return

    run_cleanup_hook("cleanup_models_gc")
    run_cleanup_hook("unload_all_models", should_run=should_unload)
    run_cleanup_hook("soft_empty_cache")
    gc.collect()


def import_custom_nodes() -> None:
    """Initialize ComfyUI custom nodes in the exporter runtime."""
    comfyui_path = get_comfyui_path()
    if comfyui_path and comfyui_path not in sys.path:
        sys.path.insert(0, comfyui_path)

    import asyncio
    import execution
    from nodes import init_extra_nodes

    if comfyui_path in sys.path:
        sys.path.remove(comfyui_path)
    sys.path.insert(0, comfyui_path)

    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)
    asyncio.run(init_extra_nodes())


# Workflow data
def build_workflow() -> dict[str, Any]:
    return {
        "1": {
            "inputs": {
                "unet_name": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
                "weight_dtype": "default",
                "world_size": 2,
                "devices": "auto",
            },
            "class_type": "MiniMaxH3SPUNETLoader",
            "_meta": {"title": "MiniMax H3 Multi-GPU Loader (Ulysses SP)"},
        },
        "2": {
            "inputs": {
                "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                "type": "minimax",
                "device": "default",
            },
            "class_type": "CLIPLoader",
            "_meta": {"title": "Load CLIP"},
        },
        "3": {
            "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"},
            "class_type": "VAELoader",
            "_meta": {"title": "Load VAE"},
        },
        "4": {
            "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"},
            "class_type": "VAELoader",
            "_meta": {"title": "Load VAE"},
        },
        "5": {
            "inputs": {"image": "example.png"},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Image"},
        },
        "6": {
            "inputs": {
                "prompt": "A red-suited superhero boy stands on a rainy city "
                "rooftop at night, cape fluttering, looking at the "
                "camera and smiling. He says: Let's go! with "
                "bright energy. Cinematic orchestral score with "
                "soft rain ambience and distant thunder.",
                "width": 1024,
                "height": 768,
                "length": 124,
                "clip": ["2", 0],
                "vae": ["3", 0],
                "first_frame": ["5", 0],
            },
            "class_type": "MiniMaxH3ImageToVideo",
            "_meta": {"title": "MiniMax H3 Image to Video"},
        },
        "7": {
            "inputs": {"noise_seed": 963657690677393},
            "class_type": "RandomNoise",
            "_meta": {"title": "RandomNoise"},
        },
        "8": {
            "inputs": {"sampler_name": "res_multistep"},
            "class_type": "KSamplerSelect",
            "_meta": {"title": "KSamplerSelect"},
        },
        "9": {
            "inputs": {"model": ["1", 0], "conditioning": ["6", 0]},
            "class_type": "BasicGuider",
            "_meta": {"title": "Basic Guider"},
        },
        "10": {
            "inputs": {
                "scheduler": "simple",
                "steps": 8,
                "denoise": 1,
                "model": ["1", 0],
            },
            "class_type": "BasicScheduler",
            "_meta": {"title": "BasicScheduler"},
        },
        "11": {
            "inputs": {
                "noise": ["7", 0],
                "guider": ["9", 0],
                "sampler": ["8", 0],
                "sigmas": ["10", 0],
                "latent_image": ["6", 1],
            },
            "class_type": "SamplerCustomAdvanced",
            "_meta": {"title": "SamplerCustomAdvanced"},
        },
        "12": {
            "inputs": {"samples": ["11", 0], "vae": ["3", 0]},
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"},
        },
        "13": {
            "inputs": {"samples": ["11", 0], "vae": ["4", 0]},
            "class_type": "VAEDecodeAudio",
            "_meta": {"title": "VAE Decode Audio"},
        },
        "14": {
            "inputs": {
                "fps": 24,
                "bit_depth": 8,
                "images": ["12", 0],
                "audio": ["13", 0],
            },
            "class_type": "CreateVideo",
            "_meta": {"title": "Create Video"},
        },
        "15": {
            "inputs": {
                "filename_prefix": "video/h3_multigpu",
                "format": "auto",
                "codec": "h264",
                "codec.encoding": "auto",
                "video": ["14", 0],
            },
            "class_type": "SaveVideo",
            "_meta": {"title": "Save Video"},
        },
    }


def build_extra_pnginfo() -> dict[str, Any] | None:
    return {
        "workflow": {
            "id": "92791889-151e-4b95-b47d-a6dbaaa91fcf",
            "revision": 0,
            "last_node_id": 15,
            "last_link_id": 18,
            "nodes": [
                {
                    "id": 2,
                    "type": "CLIPLoader",
                    "pos": [40, 220],
                    "size": [340, 110],
                    "flags": {},
                    "order": 0,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "CLIP", "type": "CLIP", "links": [3]}],
                    "properties": {"Node name for S&R": "CLIPLoader"},
                    "widgets_values": [
                        "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                        "minimax",
                        "default",
                    ],
                    "widgets_values_named": {
                        "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                        "type": "minimax",
                        "device": "default",
                    },
                },
                {
                    "id": 3,
                    "type": "VAELoader",
                    "pos": [40, 380],
                    "size": [340, 80],
                    "flags": {},
                    "order": 1,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "VAE", "type": "VAE", "links": [4, 14]}],
                    "properties": {"Node name for S&R": "VAELoader"},
                    "widgets_values": ["minimax_h3_video_vae_fp16.safetensors"],
                    "widgets_values_named": {
                        "vae_name": "minimax_h3_video_vae_fp16.safetensors"
                    },
                },
                {
                    "id": 4,
                    "type": "VAELoader",
                    "pos": [40, 510],
                    "size": [340, 80],
                    "flags": {},
                    "order": 2,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "VAE", "type": "VAE", "links": [15]}],
                    "properties": {"Node name for S&R": "VAELoader"},
                    "widgets_values": ["minimax_h3_audio_vae_fp32.safetensors"],
                    "widgets_values_named": {
                        "vae_name": "minimax_h3_audio_vae_fp32.safetensors"
                    },
                },
                {
                    "id": 5,
                    "type": "LoadImage",
                    "pos": [40, 640],
                    "size": [340, 314],
                    "flags": {},
                    "order": 3,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [
                        {"name": "IMAGE", "type": "IMAGE", "links": [5]},
                        {"name": "MASK", "type": "MASK", "links": None},
                    ],
                    "properties": {"Node name for S&R": "LoadImage"},
                    "widgets_values": ["example.png", "image"],
                    "widgets_values_named": {"image": "example.png", "upload": "image"},
                },
                {
                    "id": 7,
                    "type": "RandomNoise",
                    "pos": [440, 40],
                    "size": [300, 82],
                    "flags": {},
                    "order": 4,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "NOISE", "type": "NOISE", "links": [7]}],
                    "properties": {"Node name for S&R": "RandomNoise"},
                    "widgets_values": [963657690677393, "randomize"],
                    "widgets_values_named": {
                        "noise_seed": 963657690677393,
                        "control_after_generate": "randomize",
                    },
                },
                {
                    "id": 8,
                    "type": "KSamplerSelect",
                    "pos": [780, 40],
                    "size": [300, 80],
                    "flags": {},
                    "order": 5,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "SAMPLER", "type": "SAMPLER", "links": [8]}],
                    "properties": {"Node name for S&R": "KSamplerSelect"},
                    "widgets_values": ["res_multistep"],
                    "widgets_values_named": {"sampler_name": "res_multistep"},
                },
                {
                    "id": 9,
                    "type": "BasicGuider",
                    "pos": [900, 220],
                    "size": [260, 80],
                    "flags": {},
                    "order": 9,
                    "mode": 0,
                    "inputs": [
                        {"name": "model", "type": "MODEL", "link": 1},
                        {"name": "conditioning", "type": "CONDITIONING", "link": 6},
                    ],
                    "outputs": [{"name": "GUIDER", "type": "GUIDER", "links": [13]}],
                    "properties": {"Node name for S&R": "BasicGuider"},
                },
                {
                    "id": 11,
                    "type": "SamplerCustomAdvanced",
                    "pos": [1240, 220],
                    "size": [300, 150],
                    "flags": {},
                    "order": 10,
                    "mode": 0,
                    "inputs": [
                        {"name": "noise", "type": "NOISE", "link": 7},
                        {"name": "guider", "type": "GUIDER", "link": 13},
                        {"name": "sampler", "type": "SAMPLER", "link": 8},
                        {"name": "sigmas", "type": "SIGMAS", "link": 9},
                        {"name": "latent_image", "type": "LATENT", "link": 10},
                    ],
                    "outputs": [
                        {"name": "output", "type": "LATENT", "links": [11, 12]},
                        {"name": "denoised_output", "type": "LATENT", "links": None},
                    ],
                    "properties": {"Node name for S&R": "SamplerCustomAdvanced"},
                },
                {
                    "id": 12,
                    "type": "VAEDecode",
                    "pos": [1600, 160],
                    "size": [240, 80],
                    "flags": {},
                    "order": 11,
                    "mode": 0,
                    "inputs": [
                        {"name": "samples", "type": "LATENT", "link": 11},
                        {"name": "vae", "type": "VAE", "link": 14},
                    ],
                    "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [16]}],
                    "properties": {"Node name for S&R": "VAEDecode"},
                },
                {
                    "id": 13,
                    "type": "VAEDecodeAudio",
                    "pos": [1600, 300],
                    "size": [240, 80],
                    "flags": {},
                    "order": 12,
                    "mode": 0,
                    "inputs": [
                        {"name": "samples", "type": "LATENT", "link": 12},
                        {"name": "vae", "type": "VAE", "link": 15},
                    ],
                    "outputs": [{"name": "AUDIO", "type": "AUDIO", "links": [17]}],
                    "properties": {"Node name for S&R": "VAEDecodeAudio"},
                },
                {
                    "id": 14,
                    "type": "CreateVideo",
                    "pos": [1900, 220],
                    "size": [280, 102],
                    "flags": {},
                    "order": 13,
                    "mode": 0,
                    "inputs": [
                        {"name": "images", "type": "IMAGE", "link": 16},
                        {"name": "audio", "shape": 7, "type": "AUDIO", "link": 17},
                    ],
                    "outputs": [{"name": "VIDEO", "type": "VIDEO", "links": [18]}],
                    "properties": {"Node name for S&R": "CreateVideo"},
                    "widgets_values": [24, 8],
                    "widgets_values_named": {"fps": 24, "bit_depth": 8},
                },
                {
                    "id": 10,
                    "type": "BasicScheduler",
                    "pos": [900, 360],
                    "size": [300, 130],
                    "flags": {},
                    "order": 8,
                    "mode": 0,
                    "inputs": [{"name": "model", "type": "MODEL", "link": 2}],
                    "outputs": [{"name": "SIGMAS", "type": "SIGMAS", "links": [9]}],
                    "properties": {"Node name for S&R": "BasicScheduler"},
                    "widgets_values": ["simple", 8, 1],
                    "widgets_values_named": {
                        "scheduler": "simple",
                        "steps": 8,
                        "denoise": 1,
                    },
                },
                {
                    "id": 1,
                    "type": "MiniMaxH3SPUNETLoader",
                    "pos": [40, 40],
                    "size": [340, 130],
                    "flags": {},
                    "order": 6,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "MODEL", "type": "MODEL", "links": [1, 2]}],
                    "properties": {"Node name for S&R": "MiniMaxH3SPUNETLoader"},
                    "widgets_values": [
                        "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
                        "default",
                        2,
                        "auto",
                    ],
                    "widgets_values_named": {
                        "unet_name": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
                        "weight_dtype": "default",
                        "world_size": 2,
                        "devices": "auto",
                    },
                },
                {
                    "id": 6,
                    "type": "MiniMaxH3ImageToVideo",
                    "pos": [440, 220],
                    "size": [400, 340],
                    "flags": {},
                    "order": 7,
                    "mode": 0,
                    "inputs": [
                        {"name": "clip", "type": "CLIP", "link": 3},
                        {"name": "vae", "type": "VAE", "link": 4},
                        {"name": "first_frame", "shape": 7, "type": "IMAGE", "link": 5},
                        {
                            "name": "last_frame",
                            "shape": 7,
                            "type": "IMAGE",
                            "link": None,
                        },
                    ],
                    "outputs": [
                        {"name": "positive", "type": "CONDITIONING", "links": [6]},
                        {"name": "LATENT", "type": "LATENT", "links": [10]},
                    ],
                    "properties": {"Node name for S&R": "MiniMaxH3ImageToVideo"},
                    "widgets_values": [
                        "A red-suited superhero boy stands "
                        "on a rainy city rooftop at night, "
                        "cape fluttering, looking at the "
                        "camera and smiling. He says: "
                        "Let's go! with bright energy. "
                        "Cinematic orchestral score with "
                        "soft rain ambience and distant "
                        "thunder.",
                        1024,
                        768,
                        124,
                    ],
                    "widgets_values_named": {
                        "prompt": "A red-suited "
                        "superhero boy "
                        "stands on a rainy "
                        "city rooftop at "
                        "night, cape "
                        "fluttering, "
                        "looking at the "
                        "camera and "
                        "smiling. He says: "
                        "Let's go! with "
                        "bright energy. "
                        "Cinematic "
                        "orchestral score "
                        "with soft rain "
                        "ambience and "
                        "distant thunder.",
                        "width": 1024,
                        "height": 768,
                        "length": 124,
                    },
                },
                {
                    "id": 15,
                    "type": "SaveVideo",
                    "pos": [2220, 220],
                    "size": [320, 130],
                    "flags": {},
                    "order": 14,
                    "mode": 0,
                    "inputs": [{"name": "video", "type": "VIDEO", "link": 18}],
                    "outputs": [{"name": "video", "type": "VIDEO", "links": None}],
                    "properties": {"Node name for S&R": "SaveVideo"},
                    "widgets_values": ["video/h3_multigpu", "auto", "h264", "auto"],
                    "widgets_values_named": {
                        "filename_prefix": "video/h3_multigpu",
                        "format": "auto",
                        "codec": "h264",
                        "codec.encoding": "auto",
                    },
                },
            ],
            "links": [
                [1, 1, 0, 9, 0, "MODEL"],
                [2, 1, 0, 10, 0, "MODEL"],
                [3, 2, 0, 6, 0, "CLIP"],
                [4, 3, 0, 6, 1, "VAE"],
                [5, 5, 0, 6, 2, "IMAGE"],
                [6, 6, 0, 9, 1, "CONDITIONING"],
                [7, 7, 0, 11, 0, "NOISE"],
                [8, 8, 0, 11, 2, "SAMPLER"],
                [9, 10, 0, 11, 3, "SIGMAS"],
                [10, 6, 1, 11, 4, "LATENT"],
                [11, 11, 0, 12, 0, "LATENT"],
                [12, 11, 0, 13, 0, "LATENT"],
                [13, 9, 0, 11, 1, "GUIDER"],
                [14, 3, 0, 12, 1, "VAE"],
                [15, 4, 0, 13, 1, "VAE"],
                [16, 12, 0, 14, 0, "IMAGE"],
                [17, 13, 0, 14, 1, "AUDIO"],
                [18, 14, 0, 15, 0, "VIDEO"],
            ],
            "groups": [],
            "config": {},
            "extra": {
                "ds": {
                    "scale": 0.7000000000000003,
                    "offset": [57.32560397072022, 316.5695286668337],
                },
                "frontendVersion": "1.52.7",
                "info": "buqi-minimax-h3-multigpu example: MiniMax-H3 "
                "image-to-video with 2-GPU Ulysses sequence "
                "parallelism. Replace LoadImage with your "
                "first frame, edit the prompt, then Queue. "
                "Start ComfyUI with: python main.py "
                "--cuda-device 0,1 --highvram",
            },
            "version": 0.4,
        }
    }


workflow = build_workflow()
prompt = json.loads(json.dumps(workflow))
extra_pnginfo = build_extra_pnginfo()


# Workflow execution
def main(unload_models: bool | None = None):
    bootstrap_comfyui_runtime()
    add_extra_model_paths()
    import_custom_nodes()

    # Node imports
    from nodes import CLIPLoader, LoadImage, NODE_CLASS_MAPPINGS, VAEDecode, VAELoader

    import torch

    try:
        with torch.inference_mode():
            minimaxh3spunetloader = NODE_CLASS_MAPPINGS["MiniMaxH3SPUNETLoader"]()
            minimaxh3spunetloader_1 = minimaxh3spunetloader.EXECUTE_NORMALIZED(
                unet_name="minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
                weight_dtype="default",
                world_size=2,
                devices="auto",
            )
            cliploader = CLIPLoader()
            cliploader_2 = cliploader.load_clip(
                clip_name="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                type="minimax",
                device="default",
            )
            vaeloader = VAELoader()
            vaeloader_3 = vaeloader.load_vae(
                vae_name="minimax_h3_video_vae_fp16.safetensors"
            )
            vaeloader_4 = vaeloader.load_vae(
                vae_name="minimax_h3_audio_vae_fp32.safetensors"
            )
            loadimage = LoadImage()
            loadimage_5 = loadimage.load_image(image="example.png")
            randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
            node_7_noise_seed = prompt["7"]["inputs"]["noise_seed"] = random.randint(
                1, 2**64
            )
            randomnoise_7 = randomnoise.EXECUTE_NORMALIZED(noise_seed=node_7_noise_seed)
            ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
            ksamplerselect_8 = ksamplerselect.EXECUTE_NORMALIZED(
                sampler_name="res_multistep"
            )
            minimaxh3imagetovideo = NODE_CLASS_MAPPINGS["MiniMaxH3ImageToVideo"]()
            basicguider = NODE_CLASS_MAPPINGS["BasicGuider"]()
            basicscheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]()
            samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
            vaedecode = VAEDecode()
            vaedecodeaudio = NODE_CLASS_MAPPINGS["VAEDecodeAudio"]()
            createvideo = NODE_CLASS_MAPPINGS["CreateVideo"]()
            savevideo = NODE_CLASS_MAPPINGS["SaveVideo"]()

            start_time = time.perf_counter()

            for q in range(1):
                minimaxh3imagetovideo_6 = minimaxh3imagetovideo.EXECUTE_NORMALIZED(
                    prompt="A red-suited superhero boy stands on a rainy city rooftop at night, cape fluttering, looking at the camera and smiling. He says: Let's go! with bright energy. Cinematic orchestral score with soft rain ambience and distant thunder.",
                    width=1024,
                    height=768,
                    length=124,
                    clip=get_value_at_index(cliploader_2, 0),
                    vae=get_value_at_index(vaeloader_3, 0),
                    first_frame=get_value_at_index(loadimage_5, 0),
                )
                basicguider_9 = basicguider.EXECUTE_NORMALIZED(
                    model=get_value_at_index(minimaxh3spunetloader_1, 0),
                    conditioning=get_value_at_index(minimaxh3imagetovideo_6, 0),
                )
                basicscheduler_10 = basicscheduler.EXECUTE_NORMALIZED(
                    scheduler="simple",
                    steps=8,
                    denoise=1,
                    model=get_value_at_index(minimaxh3spunetloader_1, 0),
                )
                samplercustomadvanced_11 = samplercustomadvanced.EXECUTE_NORMALIZED(
                    noise=get_value_at_index(randomnoise_7, 0),
                    guider=get_value_at_index(basicguider_9, 0),
                    sampler=get_value_at_index(ksamplerselect_8, 0),
                    sigmas=get_value_at_index(basicscheduler_10, 0),
                    latent_image=get_value_at_index(minimaxh3imagetovideo_6, 1),
                )
                vaedecode_12 = vaedecode.decode(
                    samples=get_value_at_index(samplercustomadvanced_11, 0),
                    vae=get_value_at_index(vaeloader_3, 0),
                )
                vaedecodeaudio_13 = vaedecodeaudio.EXECUTE_NORMALIZED(
                    samples=get_value_at_index(samplercustomadvanced_11, 0),
                    vae=get_value_at_index(vaeloader_4, 0),
                )
                createvideo_14 = createvideo.EXECUTE_NORMALIZED(
                    fps=24,
                    bit_depth=8,
                    images=get_value_at_index(vaedecode_12, 0),
                    audio=get_value_at_index(vaedecodeaudio_13, 0),
                )
                #savevideo_15 = savevideo.EXECUTE_NORMALIZED(
                #    filename_prefix="video/h3_multigpu",
                #    format="auto",
                #    codec="h264",
                    #**{"codec.encoding": "auto"},
                #    video=get_value_at_index(createvideo_14, 0),
                    #prompt=prompt,
                    #extra_pnginfo=extra_pnginfo,
                #)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time:.6f} seconds")

    finally:
        cleanup_comfyui_runtime(unload_models=unload_models)


# Entrypoint
if __name__ == "__main__":
    main()
