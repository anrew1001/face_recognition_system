"""Debug SCRFD model outputs to understand the structure."""
import logging
import numpy as np

from core.config import AppConfig
from recognition import registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
config = AppConfig.from_yaml("config/recognition.yaml")
model = registry.get("scrfd_10g")
model.load()

# Create test image
image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# Get raw outputs
import cv2
input_tensor = model._preprocess_image(image)
input_name = model._session.get_inputs()[0].name
outputs = model._session.run(None, {input_name: input_tensor})

logger.info(f"\n{'='*70}")
logger.info("SCRFD Model Outputs")
logger.info(f"{'='*70}")
logger.info(f"Total number of outputs: {len(outputs)}")
logger.info("")

for i, out in enumerate(outputs):
    logger.info(f"Output {i}:")
    logger.info(f"  Shape: {out.shape}")
    logger.info(f"  Dtype: {out.dtype}")
    logger.info(f"  Min: {out.min():.4f}, Max: {out.max():.4f}, Mean: {out.mean():.4f}")
    logger.info("")

# Analyze patterns
logger.info(f"{'='*70}")
logger.info("Pattern Analysis")
logger.info(f"{'='*70}")

# Group by shape similarity
from collections import defaultdict
shape_groups = defaultdict(list)
for i, out in enumerate(outputs):
    shape_groups[out.shape].append(i)

logger.info("Outputs grouped by shape:")
for shape, indices in sorted(shape_groups.items()):
    logger.info(f"  Shape {shape}: outputs {indices}")

logger.info("")
logger.info("Hypothesis:")
logger.info("  SCRFD typically outputs:")
logger.info("    - 3 scales (8, 16, 32) × 3 types (score, bbox, kps) = 9 tensors")
logger.info("    - OR different format depending on model variant")
