"""History storage. Uses Neon Postgres if DATABASE_URL is set and reachable, otherwise a local SQLite file."""
import os, datetime as dt
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, text, inspect
from sqlalchemy.orm import declarative_base, sessionmaker

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))
Base = declarative_base()


class Scan(Base):
    __tablename__ = "scans"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    filename = Column(String(255))
    thumb = Column(Text)            # small JPEG, base64 data URI
    probs = Column(Text)            # JSON list of 5 floats
    flags = Column(Text)            # JSON list of 5 bools (prob >= tuned threshold)
    seconds = Column(Float)
    extra = Column(Text)            # JSON: every model that was run on this image (id, name, probs, flags, thresholds, seconds)


def _make_engine():
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        try:
            eng = create_engine(url, pool_pre_ping=True, pool_recycle=300)
            with eng.connect() as c:
                c.execute(text("select 1"))
            return eng, "Neon (PostgreSQL)"
        except Exception as e:
            print("!! Could not connect to DATABASE_URL, falling back to SQLite:", type(e).__name__, flush=True)
    eng = create_engine("sqlite:///" + os.path.join(HERE, "history.db"), connect_args={"check_same_thread": False})
    return eng, "local SQLite file"


engine, DB_NAME = _make_engine()
Base.metadata.create_all(engine)
# older history tables (created before the model selector existed) get the new column without losing rows
if "extra" not in [c["name"] for c in inspect(engine).get_columns("scans")]:
    with engine.begin() as c:
        c.execute(text("ALTER TABLE scans ADD COLUMN extra TEXT"))
Session = sessionmaker(bind=engine)
print("History database:", DB_NAME, flush=True)
