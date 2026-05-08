from __future__ import annotations

import base64
import hashlib
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from . import models


def generate_pkce_pair() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")
    return verifier, challenge


def load_or_create_key(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path.read_bytes()
    key = AESGCM.generate_key(bit_length=256)
    path.write_bytes(key)
    return key


@dataclass
class TokenCipher:
    key: bytes

    def encrypt(self, plaintext: str) -> tuple[bytes, bytes]:
        iv = os.urandom(12)
        encrypted = AESGCM(self.key).encrypt(iv, plaintext.encode("utf-8"), None)
        return encrypted, iv

    def decrypt(self, encrypted: bytes, iv: bytes) -> str:
        return AESGCM(self.key).decrypt(iv, encrypted, None).decode("utf-8")


class OAuthTokenStore:
    def __init__(self, db_path: Path, key_path: Path):
        self.db_path = db_path
        self.cipher = TokenCipher(load_or_create_key(key_path))

    async def save_token(self, platform: str, token_json: str, expires_at: int | None) -> None:
        await models.init_db(self.db_path)
        encrypted, iv = self.cipher.encrypt(token_json)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO oauth_tokens (platform, encrypted_token, token_iv, expires_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(platform) DO UPDATE SET
                  encrypted_token=excluded.encrypted_token,
                  token_iv=excluded.token_iv,
                  expires_at=excluded.expires_at,
                  updated_at=CURRENT_TIMESTAMP
                """,
                (platform, encrypted, iv, expires_at),
            )
            await db.commit()

    async def get_token(self, platform: str) -> str | None:
        await models.init_db(self.db_path)
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT encrypted_token, token_iv FROM oauth_tokens WHERE platform=?", (platform,))
            row = await cursor.fetchone()
        if not row:
            return None
        return self.cipher.decrypt(row[0], row[1])

    async def clear_token(self, platform: str) -> None:
        await models.init_db(self.db_path)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM oauth_tokens WHERE platform=?", (platform,))
            await db.commit()
