"""
DETR Layout Detection Model Server for Ray Serve Cluster
File: detr_layout_server.py

Deploy to cluster:
    serve deploy detr_config.yaml

Or programmatically:
    python detr_layout_server.py
"""
import ray
from ray import serve
from transformers import AutoImageProcessor, DetrForSegmentation
from typing import Dict, Any
from fastapi import Request
import torch
from PIL import Image
import base64
import io
import os


@serve.deployment()
class DETRLayoutDetector:
    def __init__(
        self, 
        model_name: str = "cmarkea/detr-layout-detection", 
        threshold: float = 0.4,
        device: str = "auto"
    ):
        """
        Initialize DETR Layout Detection model.
        
        Args:
            model_name: HuggingFace model name
            threshold: Confidence threshold for detections
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.threshold = threshold
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"[DETRLayoutDetector] Loading model '{model_name}' on {self.device}")
        
        # Check for HuggingFace token
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if token:
            print("[DETRLayoutDetector] Using HuggingFace token from environment")
        
        # Load image processor and model
        self.img_processor = AutoImageProcessor.from_pretrained(
            model_name,
            use_auth_token=token if token else None
        )
        self.model = DetrForSegmentation.from_pretrained(
            model_name,
            use_auth_token=token if token else None
        )
        self.model.to(self.device)
        self.model.eval()
        
        print(f"[DETRLayoutDetector] Model loaded successfully")
        print(f"[DETRLayoutDetector] Available labels: {list(self.model.config.id2label.values())}")

    def _decode_image(self, image_input: str) -> Image.Image:
        """
        Decode image from various formats:
        - Base64 string
        - Base64 with data URI prefix (data:image/jpeg;base64,...)
        """
        if image_input.startswith(('http://', 'https://')):
            raise ValueError("URL image loading not implemented. Please send base64.")
        
        # Remove data URI prefix if present
        if image_input.startswith('data:image'):
            image_input = image_input.split(',', 1)[1]
        
        # Decode base64
        try:
            image_bytes = base64.b64decode(image_input)
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to decode image: {str(e)}")

    def detect_layout(self, image: Image.Image, threshold: float = None) -> Dict[str, Any]:
        """
        Perform layout detection on an image.
        
        Args:
            image: PIL Image
            threshold: Optional threshold override
            
        Returns:
            Dict containing bounding boxes and segmentation masks
        """
        if threshold is None:
            threshold = self.threshold
        
        # Preprocess image
        inputs = self.img_processor(image, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.inference_mode():
            outputs = self.model(**inputs)
        
        # Post-process results
        target_size = [image.size[::-1]]  # (height, width)
        
        # Get bounding boxes
        bbox_results = self.img_processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_size
        )[0]
        
        # Get segmentation masks
        segmentation_results = self.img_processor.post_process_segmentation(
            outputs,
            threshold=threshold,
            target_sizes=target_size
        )[0]
        
        # Format results
        # Format results
        detections = []
        for i, (score, label, box) in enumerate(zip(
            bbox_results["scores"],
            bbox_results["labels"],
            bbox_results["boxes"]
        )):
            detections.append({
                "label": self.model.config.id2label[label.item()],
                "score": score.item(),
                "box": {
                    "xmin": box[0].item(),
                    "ymin": box[1].item(),
                    "xmax": box[2].item(),
                    "ymax": box[3].item()
                }
            })

        # segmentation results is a dict, extract mask shape
        if "masks" in segmentation_results:
            segmentation_shape = list(segmentation_results["masks"].shape)
        else:
            segmentation_shape = None

        return {
            "detections": detections,
            "num_detections": len(detections),
            "image_size": {"width": image.width, "height": image.height},
            "segmentation_shape": segmentation_shape
        }


    async def __call__(self, request: Request) -> Dict[str, Any]:
        """
        Handle incoming HTTP requests.
        
        Expected JSON format:
        {
            "inputs": "<base64_image>",  # or "image": "<base64_image>"
            "threshold": 0.4  # optional
        }
        """
        try:
            content_type = request.headers.get("content-type", "")
            
            if "application/json" not in content_type:
                return {
                    "success": False,
                    "error": "Content-Type must be application/json"
                }
            
            data = await request.json()
            
            # Extract image
            image_data = data.get("inputs") or data.get("image")
            if not image_data:
                return {
                    "success": False,
                    "error": "Missing 'inputs' or 'image' field in request"
                }
            
            # Decode image
            image = self._decode_image(image_data)
            
            # Get threshold (optional override)
            threshold = data.get("threshold", self.threshold)
            
            # Perform detection
            result = self.detect_layout(image, threshold=threshold)
            
            return {
                "success": True,
                "result": result["detections"],
                "metadata": {
                    "num_detections": result["num_detections"],
                    "image_size": result["image_size"],
                    "threshold": threshold
                }
            }
            
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }

    async def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "model": "cmarkea/detr-layout-detection",
            "device": self.device,
            "threshold": self.threshold
        }


# Deployment binding
deployment = DETRLayoutDetector.bind(
    model_name="cmarkea/detr-layout-detection",
    threshold=0.4,
    device="auto"
)


# Programmatic deployment (alternative to YAML)
def deploy_to_cluster(ray_address: str = "auto"):
    """
    Deploy to Ray cluster programmatically.
    
    Args:
        ray_address: Ray cluster address ('auto' for existing cluster, 'ray://host:port' for remote)
    """
    print(f"Connecting to Ray cluster at: {ray_address}")
    ray.init(address=ray_address, ignore_reinit_error=True)
    
    print("Deploying DETR Layout Detection service...")
    serve.run(
        deployment,
        name="detr-layout-detection",
        route_prefix="/layout-detection"
    )
    
    print("\n" + "="*60)
    print("✓ Deployment successful!")
    print("="*60)
    print(f"\nService URL: http://<cluster-head-ip>:8000/layout-detection")
    print("\nTest with:")
    print("  curl -X POST http://<cluster-head-ip>:8000/layout-detection \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"inputs\": \"<base64_image>\"}'")
    print("\nTo check status:")
    print("  serve status")
    print("\nTo view dashboard:")
    print("  http://<cluster-head-ip>:8265")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    # Get Ray address from command line or default to 'auto'
    ray_address = sys.argv[1] if len(sys.argv) > 1 else "auto"
    
    deploy_to_cluster(ray_address)
    
    # Keep running
    print("\nService is running. Press Ctrl+C to stop...")
    try:
        import time
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nShutting down...")
        serve.shutdown()
        ray.shutdown()