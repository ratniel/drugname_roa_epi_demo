import os
import subprocess

os.environ['LD_LIBRARY_PATH'] = '/usr/lib64-nvidia'

def run_command(command):
    result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

run_command('sudo apt-get install -y lshw')
run_command('curl -fsSL https://ollama.com/install.sh | sh')
run_command('pip install ollama instructor -Uqq')
run_command('nohup ollama serve & disown')

# automatically unload model after each generation
os.environ["OLLAMA_KEEP_ALIVE"] = "0"
