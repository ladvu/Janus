import pytorch_lightning as PL
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop, Normalize
from PIL import Image
import pandas as pd
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import PIL.Image
from torchmetrics.image.fid import FrechetInceptionDistance


def sample(x_list):
    ind = random.randint(0, len(x_list) - 1)
    return x_list[ind]

def collate_fn(batch):
    return dict(
        text = [b["text"] for b in batch],
        img = torch.stack([b["img"] for b in batch], dim = 0)
    )

@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 16,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    step:int = 0,
    mode:str = "train"
):
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
    save_dir = os.path.join('generated_samples', mode)
    os.makedirs(save_dir, exist_ok=True)
    for i in range(parallel_size):
        save_path = os.path.join(save_dir, "{}_img_{}.jpg".format(step, i))
        PIL.Image.fromarray(visual_img[i]).save(save_path)
    return visual_img

class FlickrDataset(Dataset):
    def __init__(self,root:str, df:pd.DataFrame,split = "train", resolution:int = 384):
        super().__init__()
        self.df = df[df["split"] == split]
        self.root = root
        self.transform = Compose([
            Resize(resolution),
            CenterCrop(resolution),
            ToTensor(),
            Normalize([0.48145466, 0.4578275, 0.40821073], 
                      [0.26862954, 0.26130258, 0.27577711]) 
        ])

    def __getitem__(self, index):
        record = self.df.iloc[index]
        text = sample(eval(record["raw"]))
        img_path = record["filename"]
        pil_img = Image.open(os.path.join(self.root,"flickr30k-images", img_path))
        img_tensor = self.transform(pil_img)
        return dict(text = text, img = img_tensor)

    def __len__(self):
        return len(self.df)



class FlickrWarpper(PL.LightningDataModule):
    def __init__(self, root:str, resolution:int=384, batch_size:int = 4):
        super().__init__()
        df = os.path.join(root, "flickr_annotations_30k.csv")
        self.df = pd.read_csv(df)
        self.root = root
        self.resolution = resolution
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
            FlickrDataset(self.root, self.df, "train", self.resolution),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=32,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            FlickrDataset(self.root, self.df, "val", self.resolution),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=32,
            collate_fn=collate_fn
        )
    
    def test_dataloader(self):
        return DataLoader(
            FlickrDataset(self.root, self.df, "test", self.resolution),
            batch_size=1,
            shuffle=True,
            num_workers=32,
            collate_fn=collate_fn
        )

class JanusWarpper(PL.LightningModule):
    def __init__(self,
                 model:MultiModalityCausalLM,
                 tokenizer:VLChatProcessor,
                 config:LoraConfig,
                 img_token_num:int = 576,
                 lr:float = 0.003,
                 drop_p:float = 0.1,
                 feature_extractor_weights_path = None):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "tokenizer", "config"])
        self.drop_p = drop_p
        self.img_token_num = img_token_num
        self.lr = lr
        self.model = model
        self.model.language_model.model = get_peft_model(model.language_model.model, config)
        # self.model.language_model.requires_grad_(False)
        self.tokenizer = tokenizer
        # freeze all other 
        self.model.vision_model.requires_grad_(False)
        self.model.gen_vision_model.requires_grad_(False)
        self.model.aligner.requires_grad_(False)
        self.model.gen_aligner.requires_grad_(False)
        # finetune these two model
        self.model.gen_head.requires_grad_(True)
        self.model.gen_embed.requires_grad_(True)

        self.fid_comp = FrechetInceptionDistance(feature=2048,
                                       feature_extractor_weights_path=feature_extractor_weights_path)

    @torch.no_grad() 
    def encode(self, img):
        b = img.shape[0]
        return self.model.gen_vision_model.encode(img)[2][2].reshape(b, -1)

    def text_preprocess(self, text):
        conversation = [
            {
                "role": "User",
                "content": text
            },
            {"role": "Assistant", "content": ""},
        ]

        sft_format = self.tokenizer.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.tokenizer.sft_format,
            system_prompt="",
        )
        prompt = sft_format + self.tokenizer.image_start_tag + self.tokenizer.image_end_tag
        return prompt

    @torch.no_grad()
    def tokenize(self, batch):
        b = batch['img'].shape[0]
        img_token = self.encode(batch['img'])
        # Process text for each item in batch
        text_tokens = []
        random_p = torch.rand(size=(b,))
        for drop_p, text in zip(random_p, batch['text']):
            text = self.text_preprocess(text)
            ids = self.tokenizer.tokenizer.encode(text)
            ids = torch.LongTensor(ids)
            if self.training and drop_p < self.drop_p:
                ids[1:-1] = self.tokenizer.pad_id
            text_tokens.append(ids)
        return img_token, text_tokens

    def forward(self, batch):
        b = batch['img'].shape[0]
        img_token, text_token = self.tokenize(batch)
        img_embeds = self.model.prepare_gen_img_embeds(img_token) 
        # 
        lengths = list([len(t) for t in text_token])
        max_len = max(lengths) + self.img_token_num
        embed_dim = img_embeds.shape[-1]
        inputs_embeds = torch.zeros(b, max_len, embed_dim, device = img_embeds.device, dtype=img_embeds.dtype)
        attention_mask = torch.zeros(b, max_len, device = img_embeds.device, dtype=torch.bool)
        for i in range(b):
            text_embeds = self.model.language_model.get_input_embeddings()(text_token[i].to(img_embeds.device))
            text_seqlen = text_embeds.shape[0]
            inputs_embeds[i, 0:text_seqlen - 1] = text_embeds[:-1]
            inputs_embeds[i, text_seqlen - 1: text_seqlen - 1 +self.img_token_num] = img_embeds[i]
            inputs_embeds[i, text_seqlen - 1 + self.img_token_num] = text_embeds[-1]
            attention_mask[i, :text_seqlen + self.img_token_num] = True
        position_ids = torch.arange(0, max_len).unsqueeze(0).to(device=img_embeds.device, dtype = torch.long)
        outputs = self.model.language_model.model(inputs_embeds=inputs_embeds,
                                                  position_ids=position_ids,
                                                  attention_mask=attention_mask,
                                                  use_cache=False,
                                                  past_key_values= None)
        hidden_states = outputs.last_hidden_state

        # gather from img tokens
        img_hidden_states = []
        for i in range(b):
            img_start = lengths[i] - 2
            img_end = img_start + self.img_token_num
            img_hidden_states.append(hidden_states[i, img_start:img_end, :])
        img_hidden_states = torch.stack(img_hidden_states, dim=0)
        logits = self.model.gen_head(img_hidden_states)
        b, s = img_token.shape
        ce_loss = F.cross_entropy(logits.reshape(b * s, -1), img_token.reshape(-1), reduction='mean')
        return ce_loss, logits
        
    def training_step(self, batch, batch_idx):
        self.model.train()
        ce_loss, _ = self.forward(batch) 
        self.log("train/loss", ce_loss, prog_bar=True, on_step=True, on_epoch=True)
        return ce_loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        ce_loss, logits = self.forward(batch) 
        probs = torch.softmax(logits , dim=-1)
        b, s = probs.shape[:2]
        prediction = torch.multinomial(probs.reshape(b * s, -1), num_samples=1).reshape(b, s)
        dec = self.model.gen_vision_model.decode_code(prediction.to(dtype=torch.int), shape=[b, 8, 24, 24])
        dec = torch.clamp((dec + 1) / 2 * 255, 0, 255).to(dtype=torch.uint8)
        real = torch.clamp((batch['img'] + 1) / 2 * 255, 0, 255).to(dtype = torch.uint8)
        self.log("val/loss", ce_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.fid_comp.update(dec, real=False)
        self.fid_comp.update(real, real=True)
        return ce_loss

    def on_validation_end(self):
        fid = self.fid_comp.compute()
        self.logger.experiment.add_scalar("val/fid", fid.item(), self.global_step)
        self.fid_comp.reset()
        self.save_checkpoint()
        return super().on_validation_end()

    def test_step(self, batch, batch_idx):

        text = batch['text'][0]
        text = self.text_preprocess(text)
        pred_img = generate(self.model, self.tokenizer, text, parallel_size=1, step=batch_idx, mode=self.output_dir)
        pred_img = torch.from_numpy(pred_img).permute(0,3,1,2).to(batch['img'].device)
        real = torch.clamp((batch['img'] + 1) / 2 * 255, 0, 255).to(dtype = torch.uint8)
        self.fid_comp.update(pred_img, real=False)
        self.fid_comp.update(real, real=True)

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir
    
    def on_test_end(self):
        fid = self.fid_comp.compute()
        self.fid_comp.reset()
        print(f"fid = {fid}")
        return super().on_test_end()

    def load_checkpoint(self, checkpoint):
        self.model.language_model.model.load_state_dict(checkpoint["model"], strict=False)
        self.model.gen_head.load_state_dict(checkpoint["gen_head"])
        self.model.gen_embed.load_state_dict(checkpoint["gen_embed"])

    def save_checkpoint(self):
        checkpoint = {}
        state_dict = {} 
        for name, param in self.model.language_model.named_parameters():
            if param.requires_grad:
                state_dict[name] = param
        checkpoint["model"] = state_dict
        checkpoint["gen_head"] = self.model.gen_head.state_dict()
        checkpoint["gen_embed"] = self.model.gen_embed.state_dict()
        torch.save(checkpoint, f"exp/janus-flickr-{self.global_step}.pth")

    def configure_optimizers(self):
        params = [
            {
                "params" : [p for p in self.model.language_model.model.parameters() if p.requires_grad],
                "lr": self.lr,
                "name" : "model"
            },
            {
                "params" : self.model.gen_head.parameters(),
                "lr": self.lr,
                "name": "gen_head",
            },
            {
                "params": self.model.gen_embed.parameters(),
                "lr": self.lr,
                "name": "gen_embed"
            }
        ]
        optim = torch.optim.AdamW(params)
        return {
            "optimizer" : optim
        }

        
if __name__ == "__main__":
    from pytorch_lightning import Trainer
    dataroot = "/root/cvg/project/wcy/project/Janus_IFT/flickr30k"
    datamodule = FlickrWarpper(dataroot)
    model_path = "/root/cvg/project/wcy/project/Janus_IFT/checkpoint"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    lora_config = LoraConfig(  
        task_type="CAUSAL_LM",
        r=8,  
        lora_alpha=32,  
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05,  
        bias="none"  
    )  

    pl_module = JanusWarpper(vl_gpt, vl_chat_processor, lora_config)

    trainer = Trainer(precision="bf16",
                      fast_dev_run=True,
                      devices=[0]
                      )
    trainer.fit(pl_module, datamodule)