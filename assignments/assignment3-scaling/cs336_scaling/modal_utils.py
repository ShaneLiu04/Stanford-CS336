from pathlib import Path, PurePosixPath

import modal

SUNET_ID = "brunborg"

(DATA_PATH := Path("data")).mkdir(exist_ok=True)

app = modal.App(f"scaling-{SUNET_ID}")
furu_volume = modal.Volume.from_name(
    f"scaling-{SUNET_ID}", create_if_missing=True, version=2
)
output_volume = modal.Volume.from_name(
    f"scaling-output-{SUNET_ID}", create_if_missing=True, version=2
)


def build_image(
    *, include_tests: bool = False, include_tailscale: bool = False
) -> modal.Image:
    image = modal.Image.debian_slim()
    if include_tailscale:
        image = image.apt_install("curl").run_commands(
            "curl -fsSL https://tailscale.com/install.sh | sh"
        )
    image = image.uv_sync()
    if include_tests:
        image = image.add_local_dir("tests", remote_path="/root/tests")
    if include_tailscale:
        image = (
            image.add_local_file("entrypoint.sh", "/root/entrypoint.sh", copy=True)
            .run_commands("chmod +x /root/entrypoint.sh")
            .entrypoint(["/root/entrypoint.sh"])
        )
    image = image.add_local_python_source("cs336_scaling")
    return image


VOLUME_MOUNTS: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    "/root/furu": furu_volume,
    "/root/output": output_volume,
}

MODAL_SECRETS = [
    modal.Secret.from_name("my-secret", required_keys=["HF_TOKEN", "WANDB_API_KEY"]),
    modal.Secret.from_name(
        "tailscale-auth",
        required_keys=["TAILSCALE_AUTHKEY", "TAILSCALE_URL", "INTERNAL_API_KEY"],
    ),
    modal.Secret.from_dict({"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90"}),
]
