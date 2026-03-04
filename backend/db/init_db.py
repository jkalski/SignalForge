from backend.db.models import Base
from backend.db.session import engine


def init_db():
	"""Create database tables defined on `Base` if they don't exist."""
	Base.metadata.create_all(bind=engine)


__all__ = ["init_db"]
