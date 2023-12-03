# third-party
from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
# Fast-app
from app.config.config import get_db
from app.schemas.board_schema import BoardResponseSchema, BoardResponseListSchema
from app.schemas.board_schema import BoardRequestSchema
from app.services import board_service
from app.services.jwt_service import UserIdProvider

router = APIRouter(
    prefix="/api/boards",
    tags=["게시판 API"],
)


@router.post("", response_model=BoardResponseSchema, summary="게시판 Create")
def create_board(board: BoardRequestSchema, db: Session = Depends(get_db), user_id: int = Depends(UserIdProvider())):
    return board_service.create_board(db=db, board=board, user_id=user_id)


@router.put("/{board_id}", response_model=BoardResponseSchema, summary="게시판 Update")
def update_board(board: BoardRequestSchema, db: Session = Depends(get_db), board_id: int = None,
                 user_id: int = Depends(UserIdProvider())):  # todo : board_id 입력하지 않을 경우 예외 처리
    return board_service.update_board(db=db, board=board, board_id=board_id, user_id=user_id)


@router.delete("/{board_id}", status_code=204, summary="게시판 Delete")
def update_board(db: Session = Depends(get_db), board_id: int = None, user_id: int = Depends(UserIdProvider())):
    board_service.delete_board(db=db, board_id=board_id, user_id=user_id)


@router.get("/{board_id}", response_model=BoardResponseSchema, summary="게시판 Get(1개)")
def get_board(db: Session = Depends(get_db), board_id: int = None, user_id: int = Depends(UserIdProvider())):
    return board_service.get_board(db=db, board_id=board_id, user_id=user_id)


# todo : Pagination 적용 -> page=페이지 번호(default=1), size=한 페이지당 개수(default=10)
@router.get("", response_model=BoardResponseListSchema, summary="게시판 Get(리스트)")
def get_board_list(db: Session = Depends(get_db), user_id: int = Depends(UserIdProvider()), page: int = 1,
                   size: int = 10):
    return board_service.get_board_list(db=db, user_id=user_id, page=page, size=size)
