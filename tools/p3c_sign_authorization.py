"""仅供 PI 在独立终端审阅后签署；执行代理不得调用本工具代签。"""
import argparse
import getpass
import hashlib
import hmac
import json
from pathlib import Path
import sys
import time

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()

def main():
    p = argparse.ArgumentParser(); p.add_argument('--request', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--pi-independent-review', action='store_true')
    args = p.parse_args()
    if not args.pi_independent_review or not sys.stdin.isatty():
        raise SystemExit('PI must review and sign interactively in an independent terminal')
    request = json.loads(args.request.read_text(encoding='utf8'))
    assert request['status'] == 'unsigned_request_NOT_AUTHORIZATION'
    payload = dict(request['payload']); assert payload['subcap_gpu_seconds'] == 10800
    assert payload['separate_user_approval'] and payload['PI_independent_issuance']
    if args.output.exists():
        raise SystemExit('output already exists; do not overwrite an issued approval')
    print('Approval ID:', payload['approval_id'])
    print('Scope: fixed 400 evaluations, 3 GPU-hours subcap, 160 GPU-hours total cap')
    print('Request SHA-256:', hashlib.sha256(args.request.read_bytes()).hexdigest())
    secret = getpass.getpass('PI-controlled HMAC key (at least 32 UTF-8 bytes; hidden): ').encode()
    if len(secret) < 32:
        raise SystemExit('key must contain at least 32 UTF-8 bytes')
    payload['approved_unix'] = time.time()
    signed = dict(payload, signature_hmac_sha256=hmac.new(secret, canonical(payload), hashlib.sha256).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8', newline='\n') as f:
        json.dump(signed, f, sort_keys=True, ensure_ascii=False, indent=2); f.write('\n')
    print('Signed authorization written; the key was not saved. Inject the same key through the PI-controlled launch environment.')

if __name__ == '__main__':
    main()
