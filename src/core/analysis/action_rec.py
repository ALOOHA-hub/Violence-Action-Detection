import torch
import open_clip
import torch.nn.functional as F
from PIL import Image
from src.utils.logger import logger
from src.utils.config_loader import cfg

class ActionRecognizer:
    def __init__(self):
        self.device = cfg['system']['device']
        
        # 1. Load the Model (CoCa - Contrastive Captioner)
        # This model is 'Zero-Shot'. It knows what 'fighting' looks like without training.
        model_name = 'coca_ViT-L-14'
        pretrained = 'mscoco_finetuned_laion2B-s13B-b90k'
        
        logger.info(f"Loading Phase 2 Model: {model_name}...")
        
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, 
                pretrained=pretrained,
                device=self.device
            )
            self.model.eval() # Freeze the model (Save memory)
            logger.info("Phase 2 Model loaded successfully.")
            
        except Exception as e:
            logger.critical(f"Failed to load OpenCLIP: {e}")
            raise e

        # 2. Prepare the Prompts
        # We read the list ["punching", "walking"...] from config
        self.prompts = cfg['action']['prompts']
        self.text_embeddings = self._encode_text(self.prompts)
        logger.info(f"Monitoring actions: {self.prompts}")

    def _encode_text(self, text_list):
        """
        Converts text strings into mathematical vectors.
        Run only ONCE at startup to save performance.
        """
        with torch.no_grad():
            tokenizer = open_clip.get_tokenizer('coca_ViT-L-14')
            text_tokens = tokenizer(text_list).to(self.device)
            
            # Get the vectors
            text_features = self.model.encode_text(text_tokens)
            # Normalize them (Required for Cosine Similarity)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        return text_features

    def get_action_score(self, frame_list):
        """
        Input: List of 16 Numpy Images (The Video Clip)
        Output: Dictionary { "punching": 0.85, "walking": 0.10, ... }
        """
        if not frame_list:
            return None

        # 1. Preprocess Images (Numpy -> Tensor)
        try:
            # Stack all 16 frames into one batch
            images = [self.preprocess(Image.fromarray(f)).unsqueeze(0) for f in frame_list]
            image_batch = torch.cat(images).to(self.device)
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return None
        
        # 2. Run the AI
        with torch.no_grad():
            image_features = self.model.encode_image(image_batch)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # 3. Calculate Similarity (The Dot Product)
            # This compares every frame against every text prompt
            # Result shape: [16 frames, num_prompts]
            similarity = (100.0 * image_features @ self.text_embeddings.T)
            
            # 4. Softmax to get percentages (0.0 to 1.0)
            probs = F.softmax(similarity, dim=-1)
            
            # 5. Aggregate (Take the average score across the 16 frames)
            avg_scores = probs.mean(dim=0).cpu().numpy()
            
        # Map scores back to text labels
        result = {prompt: score for prompt, score in zip(self.prompts, avg_scores)}
        return result