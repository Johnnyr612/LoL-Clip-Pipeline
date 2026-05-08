from __future__ import annotations

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from backend.auth import TokenCipher


def test_token_encrypt_decrypt():
    cipher = TokenCipher(AESGCM.generate_key(bit_length=256))
    encrypted, iv = cipher.encrypt('{"access_token":"fake"}')
    assert encrypted != b'{"access_token":"fake"}'
    assert cipher.decrypt(encrypted, iv) == '{"access_token":"fake"}'


def test_token_wrong_key():
    cipher = TokenCipher(AESGCM.generate_key(bit_length=256))
    wrong = TokenCipher(AESGCM.generate_key(bit_length=256))
    encrypted, iv = cipher.encrypt("secret")
    with pytest.raises(Exception):
        wrong.decrypt(encrypted, iv)
