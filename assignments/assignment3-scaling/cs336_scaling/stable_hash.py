import hashlib
import json

from pydantic import BaseModel

type JsonValue = (
    list[JsonValue] | dict[str, JsonValue] | str | bool | int | float | None
)


def stable_json_hash(value: object, *, digest_size: int = 10) -> str:
    dumped = json.dumps(
        _to_typed_json(value),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.blake2s(dumped.encode(), digest_size=digest_size).hexdigest()


def _fully_qualified_type_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _to_typed_json(value: object) -> JsonValue:
    if isinstance(value, BaseModel):
        return {
            "__type__": _fully_qualified_type_name(value),
            "__value__": {
                field_name: _to_typed_json(getattr(value, field_name))
                for field_name in value.__class__.model_fields
            },
        }

    if isinstance(value, dict):
        return {str(k): _to_typed_json(v) for k, v in value.items()}

    if isinstance(value, list | tuple):
        return [_to_typed_json(item) for item in value]

    if isinstance(value, str | bool | int | float) or value is None:
        return value

    raise TypeError(f"cannot convert {_fully_qualified_type_name(value)} to typed JSON")
