from src.logging.logger import logger
import sys

class ExceptionNetwork(Exception):
    def __init__(self,error,error_details:sys):
        super.__init__(error)
        
        self.error_message=error
        _,_,exc_tb=error_details.exc_info()
        
        self.line_no=exc_tb.tb_lineno
        self.file_name=exc_tb.tb_frame.f_code.co_filename
        
        self.error_message=f"error occured on [{self.file_name} ] file name, [{self.line_no}]. line , error_message [{self.error_message}]"
        logger.error(self.error_message)

    def __str__(self):
        return self.error_message
        
        
        
    

