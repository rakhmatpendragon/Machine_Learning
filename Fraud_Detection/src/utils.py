from src.logger import get_logger

logger = get_logger(__name__)

def banner(title: str) -> None:
    width = 70
    logger.info("")
    logger.info("=" * width)
    logger.info(" %s", title.upper())
    logger.info("=" * width)