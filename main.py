"""Run the API locally:  python main.py   (or: uvicorn app.app:app --reload)"""
import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "app.app:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )


if __name__ == "__main__":
    main()
