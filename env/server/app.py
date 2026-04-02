# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Env Environment.

This module creates an HTTP server that exposes the EnvEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with 'uv sync'"
    ) from e

# Use absolute imports (PYTHONPATH is set to /app/env in Dockerfile)
from models import EnvAction, EnvObservation
from server.env_environment import EnvEnvironment


# Create the app with web interface and README integration
app = create_app(
    EnvEnvironment,
    EnvAction,
    EnvObservation,
    env_name="env",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()