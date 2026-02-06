# Inside __init__ method
model_name = 'coca_ViT-L-14'
        
         # --- CHANGE THIS PART ---
        # Comment out the online name:
        # pretrained = 'mscoco_finetuned_laion2B-s13B-b90k'
        
        # Use your local file path instead:
pretrained = "models/coca_l14.bin" 
        # ------------------------
logger.info(f"Loading Phase 2 Model: {model_name}...")
        
try:
            # We remove 'cache_dir' because we are giving a direct file path now
    self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained,
            device=self.device
)
except Exception as e:
    # Note: 'logger' and 'self' must be defined for this to work
    if 'logger' in globals():
        logger.error(f"Failed to load model: {e}")
    raise 