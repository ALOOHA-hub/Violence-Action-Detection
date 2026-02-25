import torch
import open_clip
import torch.nn.functional as F
from PIL import Image
from src.utils.logger import logger
from src.utils.config_loader import cfg 

class ActionRecognizer:
    """
    Phase 2: The "Muscle" (Action Recognition).
    Responsibility: 
    1. Takes a short video clip (8 - 16 frames) of a specific person.
    2. Uses OpenCLIP to classify the action (e.g., "punching", "falling").
    """
    def __init__(self):
        self.device = cfg['system']['device']
        
        # 1. Load Settings from Config
        # We use .get() to avoid crashing if the key is missing
        # This will look for "coca_ViT-L-14"
        model_name = cfg['action'].get('model_name', 'coca_ViT-L-14')
        
        # This will look for "models/coca_l14.bin"
        pretrained = cfg['action'].get('weights_path', 'models/coca_l14.bin')

        logger.info(f"Loading Phase 2 Model: {model_name}...")
        logger.info(f"Using weights from: {pretrained}")
        
        try:
            # 2. Initialize Model
            # We point 'pretrained' to your local .bin file
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, 
                pretrained=pretrained,
                device=self.device
            )
            
            # Convert model to FP16 (Half Precision) if using GPU
            if self.device == 'cuda':
                self.model = self.model.half()
                
            self.model.eval() # Freeze the model (Save memory)
            logger.info("Phase 2 Model loaded successfully.")
            
        except Exception as e:
            logger.critical(f"Failed to load OpenCLIP: {e}")
            raise e

        # 3. Prepare the Prompts
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

            # Convert to FP16 if using GPU
            if self.device == 'cuda':
                image_batch = image_batch.half()
                
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return None
        
        # 2. Run the AI
        with torch.no_grad():
            image_features = self.model.encode_image(image_batch)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # 3. Calculate Similarity (The Dot Product)
            raw_similarity = image_features @ self.text_embeddings.T   
            scaled_similarity = 100.0 * raw_similarity
            # 4. Softmax to get percentages (0.0 to 1.0)
            probs = F.softmax(scaled_similarity, dim=-1)            
            # 5. Aggregate (Take the average score across the 16 frames)
            avg_scores = probs.mean(dim=0).cpu().numpy()
            max_raw_sim = raw_similarity.mean(dim=0).max().item()
            
        # Map scores back to text labels
        result = {prompt: score for prompt, score in zip(self.prompts, avg_scores)}
        # --- THE OSR GATEKEEPER ---
        # If the highest raw similarity is too low, the model is guessing blindly.
        # We override the result and force it into a safe "unknown" state.
        osr_thresh = cfg['action'].get('osr_threshold', 0.25)
        if max_raw_sim < osr_thresh:
           # Return a dummy result that forces the State Machine into IDLE
            return {"unknown_benign_activity": 1.0}

        return result