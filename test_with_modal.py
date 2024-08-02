from modal import Image, gpu, App, Mount

image = Image.debian_slim().pip_install_from_requirements("dev-requirements.txt")
app = App(name="test-auto_hookpoint")

@app.function(
    image=image
)
def test_modal():
    return "Hello, World!"

@app.local_entrypoint()
def main():
    test_modal.remote()