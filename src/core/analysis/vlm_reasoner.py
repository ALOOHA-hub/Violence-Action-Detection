import torch
import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from src.utils.logger import logger

class IncidentReasoner:
    def __init__(self):
        logger.info("Initializing Phase 3: VLM CoVT Reasoner...")
        self.model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # SWE Principle: Strict Memory Budgeting for 4GB VRAM
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map="auto",
                quantization_config=quantization_config
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model.eval() # Lock weights
            logger.info("VLM loaded and ready for incident reports.")
        except Exception as e:
            logger.critical(f"Failed to load VLM: {e}")
            raise e

    def analyze_incident(self, video_path):
        """
        Takes a 5-second incident MP4 and applies Chain-of-Visual-Thought (CoVT)
        to generate a structured security report.
        """
        logger.info(f"VLM analyzing incident footage: {video_path}")
        
        # --- THE CoVT PROMPT ARCHITECTURE ---
        # We force the model to analyze spatial and temporal features BEFORE concluding.
        system_prompt = (
            "You are an expert CCTV security analyst. "
            "Analyze the provided video using Chain-of-Visual-Thought reasoning. "
            "You must follow these exact steps in your mind before answering: "
            "1. Grounding: Identify the subjects and their spatial relationship. "
            "2. Temporal Tracking: Describe how their posture and actions change over the video. "
            "3. Intent Analysis: Determine if the action is hostile (assault), an accident (falling), or benign. "
            "Finally, output your conclusion STRICTLY as a JSON object."
        )
        
        user_prompt = (
            "Analyze this security footage. Return a JSON object with these exact keys:\n"
            "- 'thought_process': A brief 2-sentence summary of your CoVT analysis.\n"
            "- 'threat_level': 'CRITICAL', 'WARNING', or 'SAFE'.\n"
            "- 'weapon_detected': true or false.\n"
            "- 'final_classification': A 3-word description (e.g., 'Physical Assault', 'Accidental Fall', 'Friendly Interaction')."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 360, # Keep resolution low to save VRAM
                    "fps": 2.0,              # Sample 2 frames per second
                },
                {"type": "text", "text": user_prompt},
            ]}
        ]

        try:
            # 1. Prepare inputs using Qwen's specific vision utility
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            # 2. Run Inference (Heavy Computation)
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=150)
                
            # 3. Clean the output (strip the prompt from the response)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # Clean up potential markdown formatting from the LLM
            output_text = output_text.replace("```json", "").replace("```", "").strip()
            
            return json.loads(output_text)

        except Exception as e:
            logger.error(f"VLM Analysis Failed: {e}")
            return {"error": str(e), "threat_level": "UNKNOWN"}