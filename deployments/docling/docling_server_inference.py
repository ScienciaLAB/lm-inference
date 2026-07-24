import modal
from docling_serve.app import create_app
import time

from lm_inference_utils import get_cost_per_second

app = modal.App("docling-serve-modal")

image = (
    modal.Image.from_registry("quay.io/docling-project/docling-serve-cu128:v1.7.0")
    .run_commands("docling-tools models download --all")
    .add_local_file("lm_inference_utils.py", "/root/lm_inference_utils.py")
)


@app.function(
    image=image,
    timeout=7200,
    gpu=["A100-40GB", "A10", "L4", "T4"],
    scaledown_window=60,
    cpu=4.0,
    memory=16 * 1024,
    env={"DOCLING_SERVE_MAX_SYNC_WAIT": "7100"},
)
@modal.concurrent(max_inputs=16)
@modal.asgi_app()
def docling_serve_fastapi_app_with_lifespan():
    from fastapi import Request, Response

    web_app = create_app()

    GPU_TYPE = "A100_40GB"
    COST_PER_SEC = get_cost_per_second(GPU_TYPE)

    @web_app.middleware("http")
    async def cost_middleware(request: Request, call_next):
        start_time = time.perf_counter()
        response: Response = await call_next(request)
        duration = time.perf_counter() - start_time

        total_cost = duration * COST_PER_SEC

        print(
            f"[COST_LOG] Path={request.url.path} "
            f"Duration={duration:.2f}s "
            f"Cost=${total_cost:.6f}"
        )

        # cost info in headers
        response.headers["X-Docling-Compute-Time"] = f"{duration:.3f}"
        response.headers["X-Docling-Approx-Cost-USD"] = f"{total_cost:.6f}"
        return response

    return web_app
