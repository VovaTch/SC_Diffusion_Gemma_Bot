"""Host FastAPI policy server bridging the JBWAPI S0 container to S1/S2."""

from starcraft_diffusion_gemma.server.app import create_app

__all__ = ["create_app"]
