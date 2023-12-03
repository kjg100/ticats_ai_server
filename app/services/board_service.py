# built-in
from datetime import datetime
# third-party
from sqlalchemy.orm import Session
from sqlalchemy import or_

# Fast-app
from app.config.exceptions import ApiException, ExceptionCode
from app.models.board_model import Board
from app.schemas.board_schema import BoardRequestSchema


# 데이터 읽기 - ID로 게시판 불러오기
def get_board_by_id(db: Session, board_id: int):
    board = db.query(Board).filter(Board.id == board_id).first()
    if not board:
        raise ApiException(exception_code=ExceptionCode.BOARD_NOT_FOUND)
    return board


# 데이터 생성하기
def create_board(db: Session, board: BoardRequestSchema, user_id: int):
    # Board 저장
    db_board = Board(name=board.name, public=board.public, user_id=user_id)
    db.add(db_board)
    db.commit()
    db.refresh(db_board)
    return db_board


# 데이터 삭제 - id로 게시판 삭제하기
def delete_board_by_id(db: Session, board_id: int):
    db.query(Board).filter(Board.id == board_id).delete()
    db.commit()


# 데이터 수정하기 - id로 게시판 수정하기
def update_board(db: Session, board: BoardRequestSchema, board_id: int, user_id: int):
    # Board 저장
    db_board = get_board_by_id(db, board_id)

    # 사용자 검증
    if db_board.user_id != user_id:
        raise ApiException(exception_code=ExceptionCode.BOARD_CANT_UPDATE)

    db_board.name = board.name
    db_board.updated_at = datetime.now()
    db.add(db_board)
    db.commit()
    return db_board


# 데이터 삭제하기 - id로 게시판 삭제하기
def delete_board(db: Session, board_id: int, user_id: int):
    # Board 저장
    db_board = get_board_by_id(db, board_id)

    # 사용자 검증
    if db_board.user_id != user_id:
        raise ApiException(exception_code=ExceptionCode.BOARD_CANT_UPDATE)

    db.query(Board).filter(Board.id == board_id).delete()
    db.commit()


def get_board(db: Session, board_id: int, user_id: int):
    # 해당 board 불러오기
    db_board = get_board_by_id(db, board_id)

    # public 및 사용자 확인
    if db_board.user_id != user_id and db_board.public is False:
        raise ApiException(exception_code=ExceptionCode.BOARD_CANT_GET)

    return db_board


def get_board_list(db: Session, user_id: int):
    # 내 게시판, 전체 공개 게시판
    return db.query(Board).filter(or_(Board.user_id == user_id, Board.public == True)).all()
