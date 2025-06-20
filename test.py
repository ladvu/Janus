import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from trainer import FlickrWarpper, JanusWarpper
from peft import LoraConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == "__main__":
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
    pl_module.set_output_dir("original")
    # uncomment this line to test original model in fid
    pl_module.load_checkpoint(
        torch.load(finetune_model_path, map_location='cpu')
    )
    pl_module.set_output_dir("finetune")
    datamodule = FlickrWarpper(dataroot, batch_size=12)

    trainer = Trainer(
                      precision="bf16",
                      devices=1,
                      num_nodes=1
                      )
    # trainer.fit(pl_module, datamodule)
    trainer.test(
        pl_module, datamodule, 
    )