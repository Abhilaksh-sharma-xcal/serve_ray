import io
import os
import fitz
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageSegmentation
from ray import serve

# ============================================================
# Ray Serve Deployment
# ============================================================
@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0}  # change to 1 GPU if needed
)
class LayoutDetector:
    def __init__(self):
        print("🚀 Loading DETR layout detection model...")
        self.processor = AutoImageProcessor.from_pretrained("cmarkea/detr-layout-detection")
        self.model = AutoModelForImageSegmentation.from_pretrained("cmarkea/detr-layout-detection")
        self.model.eval()
        print("✅ Model loaded")

    # --------------------------------------------------------
    # Helper: Run layout detection on a single image
    # --------------------------------------------------------
    def _detect_page(self, page, img):
        inputs = self.processor(images=img, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([img.size[::-1]])  # height, width

        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=0.5
        )[0]

        layout_elements = []
        page_rect = page.rect

        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            x_min, y_min, x_max, y_max = box.tolist()

            # Convert image coord → PDF coord
            pdf_x_min = x_min * page_rect.width / img.width
            pdf_y_min = y_min * page_rect.height / img.height
            pdf_x_max = x_max * page_rect.width / img.width
            pdf_y_max = y_max * page_rect.height / img.height

            rect = fitz.Rect(pdf_x_min, pdf_y_min, pdf_x_max, pdf_y_max)
            text = page.get_text("text", clip=rect).strip()

            layout_elements.append({
                "label": self.model.config.id2label[label.item()],
                "confidence": float(score.item()),
                "box": [x_min, y_min, x_max, y_max],
                "text": text,
            })

        layout_elements.sort(key=lambda x: x["box"][1])

        return layout_elements

    # --------------------------------------------------------
    # Main API: detect layout from PDF bytes
    # --------------------------------------------------------
    async def __call__(self, request):
        """
        Accepts:
          - PDF file uploaded as multipart/form-data
          - OR JSON: { "pdf_path": "/path/to/file.pdf" }
        """
        if request.headers.get("content-type", "").startswith("multipart/form-data"):
            form = await request.form()
            pdf_bytes = await form["file"].read()
            pdf_path = None
        else:
            data = await request.json()
            pdf_path = data.get("pdf_path")
            pdf_bytes = None

        # Load PDF
        if pdf_path:
            doc = fitz.open(pdf_path)
        else:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        results = []

        for page_num, page in enumerate(doc):
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            layout = self._detect_page(page, img)

            results.append({
                "page": page_num,
                "image_size": img.size,
                "layout_elements": layout
            })

        doc.close()
        return {"success": True, "pages": results}


# ============================================================
# Serve App Handle
# ============================================================
app = LayoutDetector.bind()
