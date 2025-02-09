import logging
import os

format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=format,
    handlers=[
        logging.FileHandler("logs/logging.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)