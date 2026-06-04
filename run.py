import uvicorn
from money_order_validator.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "money_order_validator.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
    )
