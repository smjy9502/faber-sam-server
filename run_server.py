"""Single-worker container/native entrypoint with sanitized server logs."""
import json
import logging
from runtime_config import Settings


class ServerErrorFormatter(logging.Formatter):
    def format(self, record):
        # Uvicorn lifespan failures can embed a traceback in msg, not only exc_info.
        return json.dumps({"event": "server_error", "code": "SERVER_RUNTIME_ERROR"})


def main():
    try:
        settings = Settings.from_env()
    except ValueError:
        print(json.dumps({"event": "startup_failed", "code": "INVALID_SERVER_CONFIGURATION"}))
        return 1
    logging.basicConfig(level=settings.log_level, format="%(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(ServerErrorFormatter())
    for name in ("uvicorn", "uvicorn.error"):
        server_logger = logging.getLogger(name)
        server_logger.handlers = [handler]
        server_logger.propagate = False
        server_logger.setLevel(logging.ERROR)
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=settings.port, workers=1,
                access_log=False, log_config=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
