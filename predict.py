import os, sys, shutil
from cog import BasePredictor, Input, Path
from typing import List
sys.path.append('/content/StoryDiffusion-hf')
os.chdir('/content/StoryDiffusion-hf')

from email.policy import default
from json import encoder
import numpy as np
import torch
import random
from PIL import Image
from tqdm.auto import tqdm
from datetime import datetime
from utils.gradio_utils import is_torch2_available
if is_torch2_available():
    from utils.gradio_utils import AttnProcessor2_0 as AttnProcessor
else:
    from utils.gradio_utils import AttnProcessor
import diffusers
from diffusers import StableDiffusionXLPipeline
from utils import PhotoMakerStableDiffusionXLPipeline
from diffusers import DDIMScheduler
import torch.nn.functional as F
from utils.gradio_utils import cal_attn_mask_xl
import copy
from huggingface_hub import hf_hub_download
from diffusers.utils import load_image
from utils.utils import get_comic
from utils.style_template import styles

DEFAULT_STYLE_NAME = "Japanese Anime"
models_dict = {"RealVision": "SG161222/RealVisXL_V4.0" , "Unstable": "stablediffusionapi/sdxl-unstable-diffusers-y"}
photomaker_path =  hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")

attn_count = 0
total_count = 0
cur_step = 0
id_length = 4
total_length = 5
cur_model_type = ""
device="cuda"
attn_procs = {}
write = False
sa32 = 0.5
sa64 = 0.5
height = 768
width = 768
sd_model_path = models_dict["Unstable"]
use_safetensors= False

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_image_path_list(folder_name):
    image_basename_list = os.listdir(folder_name)
    image_path_list = sorted([os.path.join(folder_name, basename) for basename in image_basename_list])
    return image_path_list

class SpatialAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        text_context_len (`int`, defaults to 77):
            The context length of the text features.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(self, hidden_size = None, cross_attention_dim=None,id_length = 4,device = "cuda",dtype = torch.float16):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.total_length = id_length + 1
        self.id_length = id_length
        self.id_bank = {}

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None):
        global total_count,attn_count,cur_step,mask1024,mask4096
        global sa32, sa64
        global write
        global height,width
        global num_steps
        if write:
            self.id_bank[cur_step] = [hidden_states[:self.id_length], hidden_states[self.id_length:]]
        else:
            encoder_hidden_states = torch.cat((self.id_bank[cur_step][0].to(self.device),hidden_states[:1],self.id_bank[cur_step][1].to(self.device),hidden_states[1:]))
        if cur_step <=1:
            hidden_states = self.__call2__(attn, hidden_states,None,attention_mask,temb)
        else:
            random_number = random.random()
            if cur_step <0.4 * num_steps:
                rand_num = 0.3
            else:
                rand_num = 0.1
            if random_number > rand_num:
                if not write:
                    if hidden_states.shape[1] == (height//32) * (width//32):
                        attention_mask = mask1024[mask1024.shape[0] // self.total_length * self.id_length:]
                    else:
                        attention_mask = mask4096[mask4096.shape[0] // self.total_length * self.id_length:]
                else:
                    if hidden_states.shape[1] == (height//32) * (width//32):
                        attention_mask = mask1024[:mask1024.shape[0] // self.total_length * self.id_length,:mask1024.shape[0] // self.total_length * self.id_length]
                    else:
                        attention_mask = mask4096[:mask4096.shape[0] // self.total_length * self.id_length,:mask4096.shape[0] // self.total_length * self.id_length]
                hidden_states = self.__call1__(attn, hidden_states,encoder_hidden_states,attention_mask,temb)
            else:
                hidden_states = self.__call2__(attn, hidden_states,None,attention_mask,temb)
        attn_count +=1
        if attn_count == total_count:
            attn_count = 0
            cur_step += 1
            mask1024,mask4096 = cal_attn_mask_xl(self.total_length,self.id_length,sa32,sa64,height,width, device=self.device, dtype= self.dtype)

        return hidden_states
    def __call1__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            total_batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(total_batch_size, channel, height * width).transpose(1, 2)
        total_batch_size,nums_token,channel = hidden_states.shape
        img_nums = total_batch_size//2
        hidden_states = hidden_states.view(-1,img_nums,nums_token,channel).reshape(-1,img_nums * nums_token,channel)

        batch_size, sequence_length, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1,self.id_length+1,nums_token,channel).reshape(-1,(self.id_length+1) * nums_token,channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)


        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(total_batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(total_batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    def __call2__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, channel = (
            hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1,self.id_length+1,sequence_length,channel).reshape(-1,(self.id_length+1) * sequence_length,channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

def set_attention_processor(unet, id_length, is_ipadapter = False):
    global total_count
    total_count = 0
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            if name.startswith("up_blocks") :
                attn_procs[name] = SpatialAttnProcessor2_0(id_length = id_length)
                total_count +=1
            else:    
                attn_procs[name] = AttnProcessor()
        else:
            if is_ipadapter:
                attn_procs[name] = IPAttnProcessor2_0(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1,
                    num_tokens=4,
                ).to(unet.device, dtype=torch.float16)
            else:
                attn_procs[name] = AttnProcessor()

    unet.set_attn_processor(copy.deepcopy(attn_procs))
    print("successsfully load paired self-attention")
    print(f"number of the processor : {total_count}")

def apply_style_positive(style_name: str, positive: str):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive) 

def apply_style(style_name: str, positives: list, negative: str = ""):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return [p.replace("{prompt}", positive) for positive in positives], n + ' ' + negative

def process_generation(_sd_type, _model_type, _upload_images, _num_steps, style_name, _Ip_Adapter_Strength, _style_strength_ratio, guidance_scale, seed_,  sa32_, sa64_, id_length_,  general_prompt, negative_prompt, prompt_array, G_height, G_width, _comic_type, pipe2, pipe4):
    _model_type = "Photomaker" if _model_type == "Using Ref Images" else "original"
    if _model_type == "Photomaker" and "img" not in general_prompt:
        print(f"Please add the triger word \" img \"  behind the class word you want to customize, such as: man img or woman img")
    if _upload_images is None and _model_type != "original":
        print(f"Cannot find any input face image!")
    if len(prompt_array.splitlines()) > 10:
        print(f"No more than 10 prompts in huggface demo for Speed! But found {len(prompt_array.splitlines())} prompts!")
    global sa32, sa64,id_length,total_length,attn_procs,unet,cur_model_type,device
    global num_steps
    global write
    global cur_step,attn_count
    global height,width
    height = G_height
    width = G_width
    global sd_model_path,models_dict
    sd_model_path = models_dict[_sd_type]
    num_steps =_num_steps
    use_safe_tensor = True
    if  style_name == "(No style)":
        sd_model_path = models_dict["RealVision"]
    if _model_type == "original":
        pipe = StableDiffusionXLPipeline.from_pretrained(sd_model_path, torch_dtype=torch.float16)
        pipe = pipe.to(device)
        pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        set_attention_processor(pipe.unet,id_length_,is_ipadapter = False)
    elif _model_type == "Photomaker":
        if _sd_type != "RealVision" and style_name != "(No style)":
            pipe = pipe2.to(device)
            pipe.id_encoder.to(device)
            set_attention_processor(pipe.unet,id_length_,is_ipadapter = False)
        else:
            pipe = pipe4.to(device)
            pipe.id_encoder.to(device)
            set_attention_processor(pipe.unet,id_length_,is_ipadapter = False)
    else:
        raise NotImplementedError("You should choice between original and Photomaker!",f"But you choice {_model_type}")

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
    cur_model_type = _sd_type+"-"+_model_type+""+str(id_length_)
    if _model_type != "original":
        input_id_images = []
        for img in _upload_images:
            print(img)
            input_id_images.append(load_image(img))
    prompts = prompt_array.splitlines()
    start_merge_step = int(float(_style_strength_ratio) / 100 * _num_steps)
    if start_merge_step > 30:
        start_merge_step = 30
    print(f"start_merge_step:{start_merge_step}")
    generator = torch.Generator(device="cuda").manual_seed(seed_)
    sa32, sa64 =  sa32_, sa64_
    id_length = id_length_
    clipped_prompts = prompts[:]
    prompts = [general_prompt + "," + prompt if "[NC]" not in prompt else prompt.replace("[NC]","")  for prompt in clipped_prompts]
    prompts = [prompt.rpartition('#')[0] if "#" in prompt else prompt for prompt in prompts]
    print(prompts)
    id_prompts = prompts[:id_length]
    real_prompts = prompts[id_length:]
    torch.cuda.empty_cache()
    write = True
    cur_step = 0

    attn_count = 0
    id_prompts, negative_prompt = apply_style(style_name, id_prompts, negative_prompt)
    setup_seed(seed_)
    total_results = []
    if _model_type == "original":
        id_images = pipe(id_prompts, num_inference_steps=_num_steps, guidance_scale=guidance_scale,  height = height, width = width,negative_prompt = negative_prompt,generator = generator).images
    elif _model_type == "Photomaker":
        id_images = pipe(id_prompts,input_id_images=input_id_images, num_inference_steps=_num_steps, guidance_scale=guidance_scale, start_merge_step = start_merge_step, height = height, width = width,negative_prompt = negative_prompt,generator = generator).images
    else: 
        raise NotImplementedError("You should choice between original and Photomaker!",f"But you choice {_model_type}")
    total_results = id_images + total_results
    yield total_results
    real_images = []
    write = False
    for real_prompt in real_prompts:
        setup_seed(seed_)
        cur_step = 0
        real_prompt = apply_style_positive(style_name, real_prompt)
        if _model_type == "original":   
            real_images.append(pipe(real_prompt,  num_inference_steps=_num_steps, guidance_scale=guidance_scale,  height = height, width = width,negative_prompt = negative_prompt,generator = generator).images[0])
        elif _model_type == "Photomaker":      
            real_images.append(pipe(real_prompt, input_id_images=input_id_images, num_inference_steps=_num_steps, guidance_scale=guidance_scale,  start_merge_step = start_merge_step, height = height, width = width,negative_prompt = negative_prompt,generator = generator).images[0])
        else:
            raise NotImplementedError("You should choice between original and Photomaker!",f"But you choice {_model_type}")
        total_results = [real_images[-1]] + total_results
        yield total_results
    if _comic_type != "No typesetting":
        captions= prompt_array.splitlines()
        captions = [caption.replace("[NC]","") for caption in captions]
        captions = [caption.split('#')[-1] if "#" in caption else caption for caption in captions]
        from PIL import ImageFont
        total_results = get_comic(id_images + real_images, _comic_type,captions= captions,font=ImageFont.truetype("./fonts/Inkfree.ttf", int(45))) + total_results
    if _model_type == "Photomaker":
        pipe = pipe2.to("cpu")
        pipe.id_encoder.to("cpu")
        set_attention_processor(pipe.unet,id_length_,is_ipadapter = False)
    yield total_results

def array2string(arr):
    stringtmp = ""
    for i,part in enumerate(arr):
        if i != len(arr)-1:
            stringtmp += part +"\n"
        else:
            stringtmp += part

    return stringtmp

def save_images_and_collect_paths(images):
    image_objects = []
    for index, image in enumerate(images):
        image_objects.append(image)
    return image_objects

def remove_duplicates_and_save(images, save_path):
    unique_images_paths = []
    unique_images = set()
    index = 0
    for row in images:
        for img in row:
            img_data = img.tobytes()
            if img_data not in unique_images:
                unique_images.add(img_data)
                img_path = os.path.join(save_path, f"unique_{index}.png")
                img.save(img_path)
                unique_images_paths.append(img_path)
                index += 1
    return unique_images_paths

class Predictor(BasePredictor):
    def setup(self) -> None:
        self.pipe2 = PhotoMakerStableDiffusionXLPipeline.from_pretrained(models_dict["Unstable"], torch_dtype=torch.float16, use_safetensors=use_safetensors)
        self.pipe2 = self.pipe2.to("cpu")
        self.pipe2.load_photomaker_adapter(os.path.dirname(photomaker_path), subfolder="", weight_name=os.path.basename(photomaker_path), trigger_word="img")
        self.pipe2 = self.pipe2.to("cpu")
        self.pipe2.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        self.pipe2.fuse_lora()
        self.pipe4 = PhotoMakerStableDiffusionXLPipeline.from_pretrained(models_dict["RealVision"], torch_dtype=torch.float16, use_safetensors=True)
        self.pipe4 = self.pipe4.to("cpu")
        self.pipe4.load_photomaker_adapter(os.path.dirname(photomaker_path), subfolder="", weight_name=os.path.basename(photomaker_path), trigger_word="img")
        self.pipe4 = self.pipe4.to("cpu")
        self.pipe4.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        self.pipe4.fuse_lora()
    def predict(
        self,
        input_image: Path = Input(),
        style: str = Input(default="Japanese Anime", description="Style template: '(No style)', 'Japanese Anime', 'Cinematic', 'Disney Character', 'Photographic', 'Comic book', 'Line art'"),
        style_strength_ratio: int = Input(default=20, description="Style strength of Ref Image (%)"),
        model_type: str = Input(choices=["Only Using Textual Description", "Using Ref Images"], default="Using Ref Images", description="Control type of the Character"),
        general_prompt: str = Input(default="a woman img, wearing a white T-shirt, blue loose hair", description="Textual Description for Character"),
        negative_prompt: str = Input(default="bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, cartoon, cg, 3d, unreal, animate, amputation, disconnected limbs", description="Negative Prompt"),
        prompt_array: str = Input(default="wake up in the bed\nhave breakfast\nis on the road, go to company\nwork in the company\nTake a walk next to the company at noon\nlying in bed at night", description="Comic Description (each line corresponds to a frame)."),
        Ip_Adapter_Strength: float = Input(default=0.5, description="Ip Adapter Strength"),
        sa32_: float = Input(default=0.5, description="The degree of Paired Attention at 32 x 32 self-attention layers"),
        sa64_: float = Input(default=0.5, description="The degree of Paired Attention at 64 x 64 self-attention layers"),
        id_length_: int = Input(default=3, description="Number of id images in total images"),
        seed_: int = Input(default=1, description="Seed"),
        num_steps: int = Input(default=50, description="Number of sample steps"),
        G_height: int = Input(default=768, description="Height"),
        G_width: int = Input(default=768, description="Width"),
        comic_type: str = Input(choices=["Classic Comic Style", "No typesetting"], default="Classic Comic Style", description="Typesetting Style"),
        guidance_scale: int = Input(default=5, description="Guidance scale"),
    ) -> List[Path]:
        input_image = Image.open(input_image)
        input_image.save("/content/StoryDiffusion-hf/examples/taylor/1.jpeg")
        files=get_image_path_list('/content/StoryDiffusion-hf/examples/taylor')
        images = process_generation("Unstable", model_type, files, num_steps,style, Ip_Adapter_Strength, style_strength_ratio, guidance_scale, seed_, sa32_, sa64_, id_length_, general_prompt, negative_prompt, prompt_array, G_height, G_width, comic_type, self.pipe2, self.pipe4)
        image_objects = save_images_and_collect_paths(images)
        save_path = "/content"
        unique_images_paths = remove_duplicates_and_save(image_objects, save_path)
        final_paths = [Path(path) for path in unique_images_paths]
        return final_paths