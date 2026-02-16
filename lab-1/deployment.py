from subprocess import Popen

import modal

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install("setuptools<81")
    .pip_install(
        "streamlit",
        "ftfy",
        "regex",
        "tqdm",
        "torch",
        "torchvision",
        "git+https://github.com/openai/CLIP.git",
    )
    .add_local_file("index.py", "/app/index.py")
    .add_local_file("search.py", "/app/search.py")
)

app = modal.App("minuscoogle", image=image)


@app.function(
    volumes={"/assets": modal.Volume.from_name("minuscoogle")},
    scaledown_window=1200,
    env=dict(PYTHONPATH="/app"),
)
@modal.concurrent(max_inputs=16)
@modal.web_server(8000)
def run():
    Popen(
        [
            "streamlit",
            "run",
            "/app/search.py",
            "--server.port",
            "8000",
            "--server.enableCORS",
            "false",
            "--server.enableXsrfProtection",
            "false",
        ]
    )
