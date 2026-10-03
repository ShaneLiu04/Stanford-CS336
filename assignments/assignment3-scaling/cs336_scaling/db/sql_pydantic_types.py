from typing import Any

from pydantic import TypeAdapter
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import TypeDecorator


class PydanticJSON(TypeDecorator):
    impl = JSONB
    cache_ok = False

    def __init__(self, typ: Any):
        super().__init__()
        self.ta = TypeAdapter(typ)

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        value = self.ta.validate_python(value)
        return self.ta.dump_python(value, mode="json")

    def process_result_value(self, value, dialect):
        return None if value is None else self.ta.validate_python(value)

    def coerce_compared_value(self, op, value):
        return self.impl_instance.coerce_compared_value(op, value)
