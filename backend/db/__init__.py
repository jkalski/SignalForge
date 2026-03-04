from backend.db.models import Base
from backend.db.session import engine

def init_db():
    Base.metadata.create_all(bind=engine)