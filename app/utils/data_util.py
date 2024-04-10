from typing import Any, List
import time
from datetime import datetime
import pickle
import uuid
import os

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
MILLISECONDS_PER_DAY = 24 * 60 * 60 * 1000

CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
LLAMA_INDEX_HOME = os.path.join(PARENT_DIR, "llama_index_server")
BASE_NODES_PATH = os.path.join(PARENT_DIR, f"{LLAMA_INDEX_HOME}/llama_parse_nodes/base_nodes_parsed_json_full.pkl")
OBJECTS_PATH = os.path.join(PARENT_DIR, f"{LLAMA_INDEX_HOME}/llama_parse_nodes/objects_parsed_json_full.pkl")

def get_current_seconds():
    return int(time.time())


def get_current_milliseconds():
    return int(time.time() * 1000)


def milliseconds_to_human_readable(milliseconds):
    return time.strftime(TIME_FORMAT, time.localtime(milliseconds / 1000))


class CustomClientError(ValueError):
    msg: str

    def __init__(self, msg, *args, **kwargs):
        self.msg = msg
        super().__init__(args, kwargs)


def assert_not_none(value, msg=None):
    if value is None:
        msg = "value should not be None" if not msg else msg
        raise CustomClientError(msg)


def assert_true(value, msg=None):
    if value is not True:
        msg = "value should be true" if not msg else msg
        raise CustomClientError(msg)


def now():
    # dynamodb does not support datetime type, so we use isoformat() to convert datetime to string.
    return datetime.now().isoformat()


def get_doc_id(text: str):
    return text


def is_empty(value: Any):
    return value is None or value == "" or value == [] or value == {}


def not_empty(value: Any):
    return not is_empty(value)


def del_if_exists(data: dict, keys: List[str]):
    for key in keys:
        if key in data:
            del data[key]


def chunks(long_list, chunk_size):
    for i in range(0, len(long_list), chunk_size):
        yield long_list[i: i + chunk_size]

def get_nodes():
    with open(BASE_NODES_PATH, 'rb') as f:
        base_nodes_full = pickle.load(f)
    with open(OBJECTS_PATH, 'rb') as f:
        objects_full = pickle.load(f)

    return base_nodes_full + objects_full

def convert_to_valid_uuid(nodes: any):
    for node in nodes:
        try:
            _uuid = uuid.UUID(node.id_)
        except ValueError:
            node.id_ = str(uuid.uuid4())
    return nodes