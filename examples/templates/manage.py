import time
import sys
import logging
from execute import execute_mape

logging.basicConfig(level=logging.INFO, format='[CustomManage] %(message)s')

def run_mape_loop():
    logging.info("Starting Custom MAPE Loop...")
    while True:
        # Run every 5 seconds for testing
        time.sleep(5)
        logging.info("--- Triggering MAPE Cycle ---")
        execute_mape(trigger="local")

if __name__ == "__main__":
    # The wrapper calls this file. 
    # We can perform a simple loop or listen to commands.
    run_mape_loop()