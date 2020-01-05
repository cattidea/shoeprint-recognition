import os
import subprocess
import time
import webbrowser

from config_parser.config import CONFIG

docs_dir = CONFIG.docs.docs_dir
port = CONFIG.docs.port
host = 'localhost'


def docs_dev():
    shell = os.name == "nt"
    p = subprocess.Popen(["python", "-m", "http.server",
                          str(port)], shell=shell, cwd=docs_dir)
    webbrowser.open(f'http://{host}:{port}')
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            p.terminate()
            break
