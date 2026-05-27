import sys, os
sys.path.insert(0, '/root/root/autodl-tmp/emotion_qwen_feature_extractor')

import torch, json, logging
from transformers import AutoModel, AutoProcessor
from qwen_vl_utils import process_vision_info
from feature_utils import setup_logger

logger, _ = setup_logger('/root/root/autodl-tmp/emotion_qwen_feature_extractor/logs', 'probe', 'INFO')
device, dtype = "cuda:0", torch.bfloat16

# ========= 1. Load full model =========
for attn in ["sdpa", "eager"]:
    try:
        model = AutoModel.from_pretrained(
            "/root/autodl-tmp/data/Emotion-Qwen-pretrained",
            torch_dtype=dtype, attn_implementation=attn,
            device_map=device, trust_remote_code=True
        )
        logger.info(f"Loaded with {attn}")
        break
    except Exception as e:
        logger.warning(f"{attn} failed: {e}")

model.eval()
processor = AutoProcessor.from_pretrained("/root/autodl-tmp/data/Emotion-Qwen-pretrained", trust_remote_code=True)

# ========= 2. Identify vision encoder attributes =========
logger.info("=== Top-level children ===")
for name, child in model.named_children():
    logger.info(f"  {name}: {type(child).__name__}")

# ========= 3. Pick a sample video =========
with open("/root/autodl-fs/benchmark/CA-MER/video-aligned.json") as f:
    data = json.load(f)
entry = data[0]
video_path = f"/root/autodl-fs/benchmark/CA-MER/video-aligned/{entry['file_name']}.avi"

# ========= 4. Preprocess using official flow =========
messages = [{"role":"user","content":[
    {"type":"video","video":f"file://{video_path}","max_pixels":768*28*28,"fps":2.0},
    {"type":"text","text":"Analyze the emotion."}
]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
img_in, vid_in = process_vision_info(messages)
inputs = processor(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to(device)

# ========= 5. Inspect pixel_values =========
pv_keys = [k for k in inputs.keys() if 'pixel' in k]
gw_keys = [k for k in inputs.keys() if 'grid' in k]
logger.info(f"pixel keys: {pv_keys}, grid keys: {gw_keys}")
pv = inputs[pv_keys[0]] if pv_keys else inputs['pixel_values']
gw = inputs[gw_keys[0]] if gw_keys else inputs['video_grid_thw']
logger.info(f"pixel_values shape: {pv.shape}, grid_thw: {gw}")

# ========= 6. Hook vision encoder during a generate call =========
class HookProbe:
    def __init__(self): self.inputs = []; self.outputs = []
    def hook_fn(self, m, inp, out): self.inputs.append(inp); self.outputs.append(out)

# 精准 hook 模型顶层子模块
hook_targets = ["vpm", "generalcompressor", "emotioncompressor", "gatenet"]
probes = {}
for tgt in hook_targets:
    if hasattr(model, tgt):
        module = getattr(model, tgt)
        hp = HookProbe()
        handle = module.register_forward_hook(hp.hook_fn)
        probes[tgt] = (hp, handle)
        logger.info(f"  Hooked: model.{tgt} ({type(module).__name__})")
    else:
        logger.warning(f"  ⚠ model.{tgt} 不存在")

# Run generate (with tiny max_new_tokens) to avoid wasting time
with torch.no_grad():
    gen_ids = model.generate(**inputs, max_new_tokens=10)

# ========= 6. Report hook data =========
for name, (hp, handle) in probes.items():
    logger.info(f"=== model.{name} ===")
    if hp.inputs:
        inp = hp.inputs[0]
        logger.info(f"  输入参数数量: {len(inp)}")
        for i, a in enumerate(inp):
            if isinstance(a, torch.Tensor):
                logger.info(f"    arg[{i}]: Tensor shape={a.shape} dtype={a.dtype}")
            elif isinstance(a, (tuple, list)):
                logger.info(f"    arg[{i}]: {type(a).__name__} len={len(a)}")
                for j, sub in enumerate(a):
                    if isinstance(sub, torch.Tensor):
                        logger.info(f"      [{j}]: Tensor shape={sub.shape} dtype={sub.dtype}")
            else:
                logger.info(f"    arg[{i}]: {type(a).__name__}")
    else:
        logger.warning("  ❌ 未捕获到输入")

    if hp.outputs:
        out = hp.outputs[0]
        if isinstance(out, torch.Tensor):
            logger.info(f"  输出: Tensor shape={out.shape} dtype={out.dtype}")
        elif isinstance(out, (tuple, list)):
            logger.info(f"  输出: {type(out).__name__} len={len(out)}")
            for j, o in enumerate(out):
                if isinstance(o, torch.Tensor):
                    logger.info(f"    [{j}]: Tensor shape={o.shape} dtype={o.dtype}")
    else:
        logger.warning("  ❌ 未捕获到输出")

    handle.remove()

logger.info("探测完成。")
