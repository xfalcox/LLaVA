from typing import Optional
import time
import subprocess
from threading import Thread
from io import BytesIO
import shutil
import tarfile
import os

from PIL import Image
import requests
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria, get_model_name_from_path
from transformers.generation.streamers import TextIteratorStreamer

from cog import BasePredictor, Input, Path, ConcatenateIterator
from train import is_url

# we don't use the huggingface hub cache, but we need to set this to a local folder
os.environ["HUGGINGFACE_HUB_CACHE"] = os.getcwd() + "/models"

# url for the weights mirror
REPLICATE_WEIGHTS_URL = "https://weights.replicate.delivery/default"

WEIGHTS_SPEC = {
    "openai/clip-vit-large-patch14-336": {
        "src": "clip-vit-large-patch14-336/ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
        "files": [
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.bin"
        ],
    },
    "liuhaotian/llava-v1.5-13b": {
        # git commit hash from huggingface
        "src": "llava-v1.5-13b/006818fc465ebda4c003c0998674d9141d8d95f8",
        "files": [
            "config.json",
            "generation_config.json",
            "pytorch_model-00001-of-00003.bin",
            "pytorch_model-00002-of-00003.bin",
            "pytorch_model-00003-of-00003.bin",
            "pytorch_model.bin.index.json",
            "special_tokens_map.json",
            "tokenizer.model",
            "tokenizer_config.json",
        ]
    },
    "liuhaotian/llava-v1.6-vicuna-7b": {
        "src": "liuhaotian--llava-v1.6-vicuna-7b/72892672a5d71c897218c761c8e06c9a0541690d",
        "files": [
            "config.json",
            "generation_config.json",
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
            "model.safetensors.index.json",
            "special_tokens_map.json",
            "tokenizer.model",
            "tokenizer_config.json"
        ]
    },
    "liuhaotian/llava-v1.6-mistral-7b": {
        "src": "liuhaotian--llava-v1.6-mistral-7b/ff05e0854965e4584a9777a96e9bf6adf0c75a67",
        "files": [
            "config.json",
            "generation_config.json",
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
            "model.safetensors.index.json",
            "special_tokens_map.json",
            "tokenizer.model",
            "tokenizer_config.json"
        ]
    },
    "liuhaotian/llava-v1.6-vicuna-13b": {
        "src": "liuhaotian--llava-v1.6-vicuna-13b/331ea260197bc1b64c917ec70cf4f95ba566ea00",
        "files": [
            "config.json",
            "generation_config.json",
            "model-00001-of-00006.safetensors",
            "model-00002-of-00006.safetensors",
            "model-00003-of-00006.safetensors",
            "model-00004-of-00006.safetensors",
            "model-00005-of-00006.safetensors",
            "model-00006-of-00006.safetensors",
            "model.safetensors.index.json",
            "special_tokens_map.json",
            "tokenizer.model",
            "tokenizer_config.json"
        ]
    },
    "liuhaotian/llava-v1.6-34b": {
        "src": "liuhaotian--llava-v1.6-34b/bcfd034baf2adae65c3230edb25777d7922a31ce",
        "files": [
            "config.json",
            "generation_config.json",
            "model-00001-of-00015.safetensors",
            "model-00002-of-00015.safetensors",
            "model-00003-of-00015.safetensors",
            "model-00004-of-00015.safetensors",
            "model-00005-of-00015.safetensors",
            "model-00006-of-00015.safetensors",
            "model-00007-of-00015.safetensors",
            "model-00008-of-00015.safetensors",
            "model-00009-of-00015.safetensors",
            "model-00010-of-00015.safetensors",
            "model-00011-of-00015.safetensors",
            "model-00012-of-00015.safetensors",
            "model-00013-of-00015.safetensors",
            "model-00014-of-00015.safetensors",
            "model-00015-of-00015.safetensors",
            "model.safetensors.index.json",
            "special_tokens_map.json",
            "tokenizer.model",
            "tokenizer_config.json"
        ]
    }
}

def download_json(url: str, dest: Path):
    res = requests.get(url, allow_redirects=True)
    if res.status_code == 200 and res.content:
        with dest.open("wb") as f:
            f.write(res.content)
    else:
        print(f"Failed to download {url}. Status code: {res.status_code}")

def download_weights(weights: list[str]):
    """Download model weights from Replicate and save to file.
    Weights and download locations are specified in DEFAULT_WEIGHTS
    """
    basedest = Path(".")
    start = time.time()
    print("downloading to: ", basedest)
    basedest.mkdir(parents=True, exist_ok=True)
    manifest = []
    for w in weights:
        ws = WEIGHTS_SPEC[w]
        for f in ws['files']:
            url = os.path.join(REPLICATE_WEIGHTS_URL, ws["src"], f)
            dest = basedest / w / f
            if not dest.exists():
                dest.parent.mkdir(exist_ok=True, parents=True)
                print("downloading url: ", url)
                manifest.append(f"{url} {dest}")
    if manifest != []:
        process = subprocess.Popen(["pget", "multifile", "-"], stdin=subprocess.PIPE, close_fds=False)
        input_bytes = "\n".join(manifest).encode('utf-8')
        process.stdin.write(input_bytes)
        process.stdin.close()
        process.wait()  # Wait for the process to complete
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, process.args)
    print("downloading took: ", time.time() - start)

def infer_conv_mode(model_name):
    if "llava" in model_name.lower():
        if 'llama-2' in model_name.lower():
            template_name = "llava_llama_2"
        elif "mistral" in model_name.lower() or "mixtral" in model_name.lower():
            if 'orca' in model_name.lower():
                template_name = "mistral_orca"
            elif 'hermes' in model_name.lower():
                template_name = "chatml_direct"
            else:
                template_name = "mistral_instruct"
        elif 'llava-v1.6-34b' in model_name.lower():
            template_name = "chatml_direct"
        elif "v1" in model_name.lower():
            if 'mmtag' in model_name.lower():
                template_name = "v1_mmtag"
            elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                template_name = "v1_mmtag"
            else:
                template_name = "llava_v1"
        elif "mpt" in model_name.lower():
            template_name = "mpt"
        else:
            if 'mmtag' in model_name.lower():
                template_name = "v0_mmtag"
            elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                template_name = "v0_mmtag"
            else:
                template_name = "llava_v0"
    elif "mpt" in model_name:
        template_name = "mpt_text"
    elif "llama-2" in model_name:
        template_name = "llama_2"
    else:
        template_name = "vicuna_v1"
    return template_name
    
class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None) -> None:
        """Load the model into memory to make running multiple predictions efficient. 

        The parameter `weights` can be set with environment variable COG_WEIGHTS or with cog predict -e [your weights here]
        """
        # download base models
        model_path = "liuhaotian/llava-v1.5-13b"
        # model_path = "liuhaotian/llava-v1.6-mistral-7b"
        download_weights(["openai/clip-vit-large-patch14-336", model_path])
        self.model_name = get_model_name_from_path(model_path)
        disable_torch_init()
        # custom weights
        if weights is not None and str(weights) != "weights":
            print(f"Loading custom LLaVA lora model: {weights}...")
            self.model_name += "-custom-lora"

            # remove folder if it already exists
            custom_weights_dir = Path("/src/custom_weights")
            if custom_weights_dir.exists():
                shutil.rmtree(custom_weights_dir)

            # download custom weights from URL
            custom_weights_dir.mkdir(parents=True, exist_ok=True)
            weights_url = str(weights)
            download_location = custom_weights_dir / "custom_weights.tar"
            subprocess.check_call(["pget", str(weights_url), str(download_location)], close_fds=False)

            # extract tar file
            custom_weights_file = tarfile.open(download_location)
            custom_weights_file.extractall(path=custom_weights_dir)

            model_base = model_path
            model_path = custom_weights_dir
        else:
            print(f"Loading base LLaVA model...")
            model_base = None

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, model_name=self.model_name, model_base=model_base, load_8bit=False, load_4bit=False)

    def predict(
        self,
        image: Path = Input(description="Input image", default=None),
        prompt: str = Input(description="Prompt to use for text generation"),
        top_p: float = Input(description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens", ge=0.0, le=1.0, default=1.0),
        temperature: float = Input(description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic", default=0.2, ge=0.0),
        max_tokens: int = Input(description="Maximum number of tokens to generate. A word is generally 2-3 tokens", default=1024, ge=0),
        history: list[str] = Input(description="List of earlier chat messages, alternating roles, starting with user input. Include <image> to specify which message to attach the image to.", default=None),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""
    
        conv_mode = infer_conv_mode(self.model_name)
        conv = conv_templates[conv_mode].copy()
    
        if image is not None:
            image_data = [load_image(str(image))]
            image_sizes = [i.size for i in image_data]
            image_tensor = process_images(image_data, self.image_processor, self.model.config)
            if type(image_tensor) is list:
                image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
            image_args = {
                "images": image_tensor,
                "image_sizes": image_sizes
            }
            num_images = len(image_data)
        else:
            image_args = {}
            num_images = 0
    
        if history is not None:
            for i, msg in enumerate(history):
                conv.append_message(conv.roles[i % 2], msg)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()

        # no image token, insert it in the first message
        if prompt.count(DEFAULT_IMAGE_TOKEN) < num_images:
            conv.messages[0] = (conv.messages[0][0], DEFAULT_IMAGE_TOKEN + '\n' + conv.messages[0][1])
            prompt = conv.get_prompt()

        if num_images != prompt.count(DEFAULT_IMAGE_TOKEN):
            # There could be multiple but the gradio demo seems to avoid it
            # raise ValueError("Number of images does not match number of <image> tokens in prompt")
            raise ValueError("There can only be one <image> token in the prompt")

        replace_token = DEFAULT_IMAGE_TOKEN
        if getattr(self.model.config, 'mm_use_im_start_end', False):
            replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
        # TODO: error if max token length reached
        num_image_tokens = prompt.count(replace_token) * self.model.get_vision_tower().num_patches

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else conv.sep2
        keywords = [stop_str]
        max_context_length = getattr(self.model.config, 'max_position_embeddings', 2048)
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=20.0)

        max_new_tokens = min(max_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)

        if max_new_tokens < 1:
            raise RuntimeError("Conversation exceeds max token length. Please start a new conversation, thanks.")

    
        with torch.inference_mode():
            thread = Thread(target=self.model.generate, kwargs=dict(
                inputs=input_ids,
                do_sample=True if temperature > 0.001 else False,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                **image_args
            ))
            thread.start()
            # workaround: second-to-last token is always " "
            # but we want to keep it if it's not the second-to-last token
            prepend_space = False
            for new_text in streamer:
                if new_text == " ":
                    prepend_space = True
                    continue
                if new_text.endswith(stop_str):
                    new_text = new_text[:-len(stop_str)].strip()
                    prepend_space = False
                elif prepend_space:
                    new_text = " " + new_text
                    prepend_space = False
                if len(new_text):
                    yield new_text
            if prepend_space:
                yield " "
            thread.join()
    

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

