import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from trainer import FlickrWarpper, JanusWarpper
from peft import LoraConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import gradio as gr
import numpy as np
import os
from PIL import Image

dataroot = "/root/cvg/project/wcy/project/Janus_IFT/flickr30k"
model_path = "/root/cvg/project/wcy/project/Janus_IFT/checkpoints/janus_7b"
finetune_model_path = "/root/cvg/project/wcy/project/Janus_IFT/checkpoints/finetune_7b/janus-flickr-19135.pth"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
lora_config = LoraConfig(  
    r=8,  
    lora_alpha=32,  
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05,  
    bias="none"  
)  
pl_module = JanusWarpper(vl_gpt, vl_chat_processor, lora_config, lr=0.00004, feature_extractor_weights_path="../checkpoints/inceptionv3/inceptionv3.pth")
pl_module.load_checkpoint(
    torch.load(finetune_model_path, map_location='cpu')
)
mmgpt = pl_module.model
vl_chat_processor = pl_module.tokenizer

@torch.inference_mode()
def generate(
    prompt: str,
    temperature: float = 1,
    cfg_weight: float = 5,
    seed:int = 0
):
    torch.cuda.empty_cache()
    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    parallel_size = 1
    prompt = pl_module.text_preprocess(prompt)
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state
        
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)


    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec
    output_img = []
    for i in range(parallel_size):
        output_img.append(Image.fromarray(visual_img[i]))
    return output_img



if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown(value="# Text-to-Image Generation")
        with gr.Row():
            cfg_weight_input = gr.Slider(minimum=1, maximum=10, value=5, step=0.5, label="CFG Weight")
            t2i_temperature = gr.Slider(minimum=0, maximum=1, value=1.0, step=0.05, label="temperature")

        prompt_input = gr.Textbox(label="Prompt. (Prompt in more detail can help produce better images!)")
        seed_input = gr.Number(label="Seed (Optional)", precision=0, value=12345)
        generation_button = gr.Button("Generate Images")

        image_output = gr.Gallery(label="Generated Images", columns=2, rows=2, height=300)

        examples_t2i = gr.Examples(
            label="Text to image generation examples.",
            examples=[
                "Master shifu racoon wearing drip attire as a street gangster.",
                "The face of a beautiful girl",
                "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                "A glass of red wine on a reflective surface.",
                "A cute and adorable baby fox with big brown eyes, autumn leaves in the background enchanting,immortal,fluffy, shiny mane,Petals,fairyism,unreal engine 5 and Octane Render,highly detailed, photorealistic, cinematic, natural colors.",
                "The image features an intricately designed eye set against a circular backdrop adorned with ornate swirl patterns that evoke both realism and surrealism. At the center of attention is a strikingly vivid blue iris surrounded by delicate veins radiating outward from the pupil to create depth and intensity. The eyelashes are long and dark, casting subtle shadows on the skin around them which appears smooth yet slightly textured as if aged or weathered over time.\n\nAbove the eye, there's a stone-like structure resembling part of classical architecture, adding layers of mystery and timeless elegance to the composition. This architectural element contrasts sharply but harmoniously with the organic curves surrounding it. Below the eye lies another decorative motif reminiscent of baroque artistry, further enhancing the overall sense of eternity encapsulated within each meticulously crafted detail. \n\nOverall, the atmosphere exudes a mysterious aura intertwined seamlessly with elements suggesting timelessness, achieved through the juxtaposition of realistic textures and surreal artistic flourishes. Each component\u2014from the intricate designs framing the eye to the ancient-looking stone piece above\u2014contributes uniquely towards creating a visually captivating tableau imbued with enigmatic allure.",
            ],
            inputs=prompt_input,
        )

        generation_button.click(
            fn=generate,
            inputs=[prompt_input, t2i_temperature, cfg_weight_input, seed_input],
            outputs=image_output
        )
    demo.launch(share=True)
#
