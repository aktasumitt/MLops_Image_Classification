from datetime import datetime
import logging
import os


LOG_FILE_NAME=datetime.now().strftime("%d-%m-%Y_%H_%M_%S")

log_dir="logs"

os.makedirs(log_dir,exist_ok=True)
LOG_FILE_PATH=os.path.join(log_dir,LOG_FILE_NAME)


logging.basicConfig(filename=LOG_FILE_PATH,
                    level=logging.INFO,
                    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")

logger=logging.getLogger("Logger")

