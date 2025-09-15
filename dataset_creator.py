import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
import math
import time
import argparse
import sys

import os
os.environ['GOOGLE_API_KEY'] = 'AIzaSyDX8Ts8aMRtUbCak76q573t1ZMu_za0320 '

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def create_smart_prompt(base_prompt, max_words, task_type="description"):
    """
    Intelligently modify prompts to encourage concise responses
    """
    if max_words <= 0:
        return base_prompt
    
    # Smart prompt engineering based on task type
    if task_type == "description":
        if max_words <= 10:
            return f"{base_prompt} Give a very brief, {max_words}-word maximum description."
        elif max_words <= 25:
            return f"{base_prompt} Be concise. Limit to {max_words} words maximum."
        elif max_words <= 50:
            return f"{base_prompt} Keep it brief, under {max_words} words."
        else:
            return f"{base_prompt} Be concise and descriptive."
    
    elif task_type == "vpr":
        if max_words <= 15:
            return f"{base_prompt} List only the most distinctive features in {max_words} words max."
        elif max_words <= 30:
            return f"{base_prompt} Focus on key identifying features. {max_words} words maximum."
        else:
            return f"{base_prompt} Emphasize distinctive elements for place recognition."
    
    return base_prompt

def run_intern_vl(image_path, max_response_words=0):
    # If you set `load_in_8bit=True`, you will need two 80GB GPUs.
    # If you set `load_in_8bit=False`, you will need at least three 80GB GPUs.
    path = 'OpenGVLab/InternVL3-8B'
    #device_map = split_model('InternVL3-8B')
    device_map = split_model(path)
    device_map = 'cuda:0'  # Use a single GPU for simplicity in this example
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    # set the max number of tiles in `max_num`
    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
    
    # Adjust generation config based on word limit
    if max_response_words > 0:
        # Rough estimate: 1 word ≈ 1.5 tokens
        max_tokens = max(10, int(max_response_words * 1.5))
        generation_config = dict(max_new_tokens=max_tokens, do_sample=True)
    else:
        generation_config = dict(max_new_tokens=1024, do_sample=True)

    # single-image single-round conversation
    image_paths = [image_path] * 1

    pixel_values = [load_image(image_path, max_num=12).to(torch.bfloat16).cuda() for image_path in image_paths]
    pixel_values = torch.cat(pixel_values, dim=0)

    base_question = 'describe all objects in this image from left to right in one line, including their attributes and colors, ignore dynamic objects like people and cars. in your response, use the format: object1, object2, object3, ...'
    
    # Apply smart prompt engineering
    question = create_smart_prompt(base_question, max_response_words, "description")
    
    for i in range(1):
        t1 = time.time()
        response = model.chat(tokenizer, pixel_values, question, generation_config)
        t2 = time.time()
        print(f'Inference time: {t2 - t1:.2f} seconds')
        print(f'User: {question}\nAssistant: {response}')
        
        # Post-process response if needed
        if max_response_words > 0:
            words = response.split()
            if len(words) > max_response_words:
                print(f"\n⚠️  Response truncated from {len(words)} to {max_response_words} words:")
                truncated_response = ' '.join(words[:max_response_words])
                print(f"Truncated: {truncated_response}")

def run_gemini(image_path, max_response_words=0):
    from google.genai import types
    from google import genai

    client = genai.Client()

    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    base_text = 'describe from left to right all distinctive features in one line for visual place recognition'
    
    # Apply smart prompt engineering
    text = create_smart_prompt(base_text, max_response_words, "vpr")

    t1 = time.time()
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',
            ),
            text,
        ]
    )
    t2 = time.time()
    print(f'Inference time: {t2 - t1:.2f} seconds')

    # Post-process response if needed
    description = response.text
    if max_response_words > 0:
        words = description.split()
        if len(words) > max_response_words:
            print(f"\n⚠️  Response truncated from {len(words)} to {max_response_words} words:")
            truncated_description = ' '.join(words[:max_response_words])
            print(f"Truncated: {truncated_description}")
            description = truncated_description
    
    print(description)

def main():
    parser = argparse.ArgumentParser(description="Text-based Visual Place Recognition Dataset Creator")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file")
    parser.add_argument("--model", type=str, choices=["internvl", "gemini"], default="gemini", 
                       help="Model to use for image description")
    parser.add_argument("--max_words", type=int, default=0, 
                       help="Maximum number of words in response (0 = no limit)")
    parser.add_argument("--output_file", type=str, default=None, 
                       help="File to save the response (optional)")
    
    args = parser.parse_args()
    
    print(f"Processing image: {args.image_path}")
    print(f"Model: {args.model}")
    print(f"Max words: {args.max_words if args.max_words > 0 else 'No limit'}")
    print("-" * 50)
    
    if args.model == "internvl":
        run_intern_vl(args.image_path, args.max_words)
    elif args.model == "gemini":
        run_gemini(args.image_path, args.max_words)
    
    print("\n✅ Processing completed!")

if __name__ == '__main__':
    # For backward compatibility, you can still use the old way
    if len(sys.argv) == 1:  # No command line arguments
        image_path = 'images\@0543256.96@4178906.70@10@S@037.75645@-122.50893@L-2IR8dif5bpYQIeb5o7oQ@@0@@@@201502@@.jpg'
        run_gemini(image_path, max_response_words=20)
    else:
        main()