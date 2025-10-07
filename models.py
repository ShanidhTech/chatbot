from datetime import datetime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import BigInteger, String, Column, DateTime

Base = declarative_base()

class Chat(Base):
    """
    SQLAlchemy model for storing chat interactions.
    """
    __tablename__ = "chat"
    id = Column(BigInteger, primary_key=True, index=True)
    question = Column(String, nullable=False)
    answer = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=True)
