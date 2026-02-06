from src.utils.config_loader import cfg
from src.pipelines.rapid_flow import RapidPipeline

if __name__ == "__main__":
    # Get video path from config
    video_path = cfg['paths']['input_source']
    
    # Initialize and Run
    app = RapidPipeline()
    app.run(video_path)