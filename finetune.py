import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from trainer import FlickrWarpper, JanusWarpper
from peft import LoraConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == "__main__":
    dataroot = "/root/cvg/project/wcy/project/Janus_IFT/flickr30k"
    model_path = "/root/cvg/project/wcy/project/Janus_IFT/checkpoints/janus_1_3b"
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

    pl_module = JanusWarpper(vl_gpt, vl_chat_processor, lora_config, lr=0.00004)
    datamodule = FlickrWarpper(dataroot, batch_size=12)

    trainer = Trainer(
                      precision="bf16",
                      callbacks=[
                          ModelCheckpoint(monitor='val/loss', dirpath='exp', filename='janus-flickr-{step:05d}-{val_loss:.2f}')
                      ],
                      max_epochs=2,
                      max_steps=4,
                      logger=True,
                      val_check_interval=2,
                      enable_checkpointing=True,
                      accumulate_grad_batches=1,
                      )
    trainer.fit(pl_module, datamodule)