from pydantic import BaseModel, Field
from typing import Any, Generic, TypeVar
from datetime import datetime

T = TypeVar("T")

class R(BaseModel, Generic[T]):
    code: int = Field(200, description="业务状态码")
    msg:  str = Field("success", description="提示信息")
    data: T   = Field(None, description="业务数据")
    ts:   int = Field(..., description="服务器时间戳(毫秒)")
    date:   str = Field(..., description="服务器时间戳(日期)")



def response_success(data=None) -> R:
    return R(data=data, ts=int(datetime.now().timestamp()*1000), date=str(datetime.now()))

def response_fail(data=None) -> R:
    return R(code=500, data=data, ts=int(datetime.now().timestamp()*1000), date=str(datetime.now()))