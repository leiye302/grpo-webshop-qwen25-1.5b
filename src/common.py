"""Portable WebShop release paths and CPU-only contracts."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / '.deps/sdar'
WEBSHOP = UPSTREAM / 'agent_system/environments/env_package/webshop/webshop'
WORLD_SIZE = 4


def load(name):
    return json.loads((ROOT / 'configs' / name).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    with pending.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def devices(spec):
    ids = [x.strip() for x in spec.split(',')]
    if len(ids) != WORLD_SIZE or len(set(ids)) != WORLD_SIZE or any(not x.isdigit() for x in ids):
        raise ValueError('Choose four distinct allocated GPU indices, e.g. 0,1,2,3')
    return ids


def protocol(text):
    import re
    reasoning = re.search(r'<think>(.*?)</think>', text, re.S)
    strict = re.fullmatch(r'\s*<think>(.*?)</think>\s*<action>(.*?)</action>\s*',
                          text.replace('<|im_end|>', '').replace('<|endoftext|>', ''), re.S)
    return {'reasoning_open': '<think>' in text,
            'reasoning_nonempty_closed': bool(reasoning and reasoning[1].strip()),
            'strict_protocol': bool(strict and strict[1].strip() and strict[2].strip()
                                    and text.count('<think>') == text.count('</think>') == 1
                                    and text.count('<action>') == text.count('</action>') == 1)}
