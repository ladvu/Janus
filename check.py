from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop, Normalize

import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import os
import PIL.Image
from trainer import JanusWarpper
from peft import LoraConfig

# specify the path to the model
model_path = "/root/cvg/project/wcy/project/Janus_IFT/checkpoints/janus_7b"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

transform = Compose([
            Resize(384),
            CenterCrop(384),
            ToTensor(),
            Normalize(0.5, 0.5) 
        ])
img = PIL.Image.open("/root/cvg/project/wcy/project/Janus_IFT/flickr30k/flickr30k-images/1002674143.jpg")
lora_config = LoraConfig(  
        r=8,  
        lora_alpha=32,  
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05,  
        bias="none",
    )
janus = JanusWarpper(vl_gpt, vl_chat_processor, lora_config, feature_extractor_weights_path="../checkpoints/inceptionv3/inceptionv3.pth")
batch = {
    "text" : ["A small girl in the grass plays with fingerpaints in front of a white canvas with a rainbow on it."],
    "img" : transform(img).unsqueeze(0).cuda().to(dtype=torch.bfloat16)
}

loss, logits = janus.forward(batch)
probs = torch.softmax(logits , dim=-1)
prediction = torch.multinomial(probs[0], num_samples=1).unsqueeze(0)
dec = vl_gpt.gen_vision_model.decode_code(prediction.to(dtype=torch.int), shape=[1, 8, 24, 24])
dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
dec = np.clip((dec + 1) / 2 * 255, 0, 255)
visual_img = np.zeros((1, 384, 384, 3), dtype=np.uint8)
visual_img[:, :, :] = dec
os.makedirs('checksamples', exist_ok=True)
for i in range(1):
    save_path = os.path.join('checksamples', "img_{}.jpg".format(i))
    PIL.Image.fromarray(visual_img[i]).save(save_path)

print(loss)