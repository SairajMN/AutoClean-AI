"""Compatibility shim for the canonical OpenEnv FastAPI app."""

from env.app import app, startup_event


def main() -> None:
    import uvicorn

    uvicorn.run("env.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
