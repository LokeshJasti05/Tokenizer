"""
models.py
---------
SQLAlchemy model for storing tokenizer session history in SQLite.
"""

import json
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

DATABASE_URL = "sqlite:///tokenizer_history.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class TokenizerSession(Base):
    __tablename__ = "tokenizer_sessions"

    id          = Column(Integer, primary_key=True, index=True)
    created_at  = Column(DateTime, default=datetime.utcnow)
    model       = Column(String(64), nullable=False, default="custom-bpe")
    messages    = Column(Text, nullable=False)   # JSON array of {role, content}
    token_count = Column(Integer, nullable=False)
    token_ids   = Column(Text, nullable=False)   # JSON array of ints
    raw_text    = Column(Text, nullable=True)     # the full prompt string

    def to_dict(self):
        return {
            "id":           self.id,
            "created_at":   self.created_at.isoformat() if self.created_at else None,
            "model":        self.model,
            "messages":     json.loads(self.messages),
            "token_count":  self.token_count,
            "token_ids":    json.loads(self.token_ids),
            "raw_text":     self.raw_text,
        }


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
