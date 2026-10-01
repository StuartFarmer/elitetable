"""Bounded JSON values and Gymnasium spaces; never deserialize Python objects."""

import base64
import json
import math
import struct

import numpy as np
from gymnasium import spaces

MAX_MESSAGE = 1_048_576
MAX_ARRAY_BYTES = 262_144
MAX_SOURCE = 65_536


def dumps(value):
    data = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    if len(data) > MAX_MESSAGE:
        raise ValueError("Message exceeds 1 MiB")
    return data + b"\n"


def _invalid_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def loads(data):
    if not data or len(data) > MAX_MESSAGE + 1:
        raise ValueError("Missing or oversized message")
    return json.loads(data, parse_constant=_invalid_constant)


def frame_size(header):
    """Validate a binary channel's four-byte network-order length prefix."""
    if len(header) != 4:
        raise ValueError("Missing frame header")
    size = struct.unpack("!I", header)[0]
    if not 0 < size <= MAX_MESSAGE:
        raise ValueError("Missing or oversized message")
    return size


def pack(message):
    """Frame numeric arrays directly; retain bounded JSON for all other messages."""
    if message.get("command") == "act":
        key, tag = "observation", b"O"
    elif set(message) == {"action"}:
        key, tag = "action", b"A"
    else:
        key = None
    value = message.get(key)
    if key and isinstance(value, np.ndarray):
        _shape(value.shape, _dtype(value.dtype.str))
        body = tag + dumps([value.dtype.str, list(value.shape)]) + value.tobytes()
    else:
        body = b"J" + dumps({**message, key: encode(value)} if key else message)
    if len(body) > MAX_MESSAGE:
        raise ValueError("Message exceeds 1 MiB")
    return struct.pack("!I", len(body)) + body


def unpack(body):
    """Decode a frame body; never accept pickle or object-array payloads."""
    if not body or len(body) > MAX_MESSAGE:
        raise ValueError("Missing or oversized message")
    tag = body[:1]
    if tag == b"J":
        message = loads(body[1:])
        if not isinstance(message, dict):
            raise ValueError("Message must be an object")
        if message.get("command") == "act":
            message["observation"] = decode(message["observation"])
        elif set(message) == {"action"}:
            message["action"] = decode(message["action"])
        return message
    if tag not in (b"O", b"A"):
        raise ValueError("Unknown frame tag")
    end = body.find(b"\n", 1, 1025)
    if end < 0:
        raise ValueError("Missing array header")
    metadata = loads(body[1:end])
    if not isinstance(metadata, list) or len(metadata) != 2:
        raise ValueError("Invalid array header")
    dtype = _dtype(metadata[0])
    size = _shape(metadata[1], dtype)
    if len(body) - end - 1 != size:
        raise ValueError("Array payload does not match shape")
    value = np.frombuffer(body, dtype=dtype, offset=end + 1).reshape(metadata[1]).copy()
    return {"command": "act", "observation": value} if tag == b"O" else {"action": value}


def _dtype(name):
    dtype = np.dtype(name)
    if dtype.kind not in "biufc" or dtype.itemsize > 16:
        raise ValueError("Only ordinary numeric and boolean arrays are supported")
    return dtype


def _shape(shape, dtype):
    if not isinstance(shape, (list, tuple)) or len(shape) > 32:
        raise ValueError("Invalid array shape")
    if any(type(n) is not int or n < 0 or n > MAX_ARRAY_BYTES for n in shape):
        raise ValueError("Invalid array dimension")
    size = math.prod(shape) * dtype.itemsize
    if size > MAX_ARRAY_BYTES:
        raise ValueError("Array exceeds 256 KiB")
    return size


def encode(value, _depth=0):
    if _depth > 64:
        raise ValueError("Value nesting exceeds 64")
    if isinstance(value, np.ndarray):
        _shape(value.shape, _dtype(value.dtype.str))
        return [
            "array",
            value.dtype.str,
            list(value.shape),
            base64.b64encode(value.tobytes()).decode(),
        ]
    if isinstance(value, np.generic):
        return encode(value.item(), _depth + 1)
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else ["float", str(value)]
    if isinstance(value, (tuple, list)):
        return [
            "tuple" if isinstance(value, tuple) else "list",
            [encode(v, _depth + 1) for v in value],
        ]
    if isinstance(value, dict):
        return ["dict", [[encode(k, _depth + 1), encode(v, _depth + 1)] for k, v in value.items()]]
    raise ValueError(f"Unsupported value type: {type(value).__name__}")


def decode(value, _depth=0):
    if _depth > 64:
        raise ValueError("Value nesting exceeds 64")
    if value is None or type(value) in (bool, int, float, str):
        return value
    if not isinstance(value, list) or not value:
        raise ValueError("Invalid encoded value")
    tag = value[0]
    if tag == "array" and len(value) == 4:
        dtype = _dtype(value[1])
        size = _shape(value[2], dtype)
        if not isinstance(value[3], str) or len(value[3]) > 4 * ((size + 2) // 3):
            raise ValueError("Invalid array payload size")
        data = base64.b64decode(value[3], validate=True)
        if len(data) != size:
            raise ValueError("Array payload does not match shape")
        return np.frombuffer(data, dtype=dtype).reshape(value[2]).copy()
    if len(value) != 2:
        raise ValueError("Invalid encoded value")
    if tag == "float" and value[1] in ("nan", "inf", "-inf"):
        return float(value[1])
    if tag in ("tuple", "list") and isinstance(value[1], list):
        items = [decode(v, _depth + 1) for v in value[1]]
        return tuple(items) if tag == "tuple" else items
    if tag == "dict" and isinstance(value[1], list):
        return {decode(k, _depth + 1): decode(v, _depth + 1) for k, v in value[1]}
    raise ValueError("Unknown value tag")


def encode_space(space):
    if isinstance(space, spaces.Box):
        return {
            "type": "Box",
            "low": encode(space.low),
            "high": encode(space.high),
            "dtype": space.dtype.str,
        }
    if isinstance(space, spaces.Discrete):
        return {"type": "Discrete", "n": int(space.n), "start": int(space.start)}
    if isinstance(space, spaces.Dict):
        return {"type": "Dict", "spaces": [[k, encode_space(v)] for k, v in space.spaces.items()]}
    if isinstance(space, spaces.Tuple):
        return {"type": "Tuple", "spaces": [encode_space(v) for v in space.spaces]}
    if isinstance(space, spaces.Text):
        if space.max_length > MAX_ARRAY_BYTES:
            raise ValueError("Text space exceeds 256 KiB")
        return {
            "type": "Text",
            "min_length": space.min_length,
            "max_length": space.max_length,
            "charset": space.characters,
        }
    raise ValueError(f"Unsupported isolated space: {type(space).__name__}")


def decode_space(value):
    kind = value["type"]
    if kind == "Box":
        return spaces.Box(decode(value["low"]), decode(value["high"]), dtype=_dtype(value["dtype"]))
    if kind == "Discrete":
        return spaces.Discrete(value["n"], start=value["start"])
    if kind == "Dict":
        return spaces.Dict([(k, decode_space(v)) for k, v in value["spaces"]])
    if kind == "Tuple":
        return spaces.Tuple(tuple(decode_space(v) for v in value["spaces"]))
    if kind == "Text" and 0 <= value["min_length"] <= value["max_length"] <= MAX_ARRAY_BYTES:
        return spaces.Text(
            value["max_length"], min_length=value["min_length"], charset=value["charset"]
        )
    raise ValueError("Unsupported space definition")
