"""Ed25519 oracle identity for world receipt signing."""

import hashlib
from dataclasses import dataclass, field

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature


@dataclass
class OracleIdentity:
    world_public_key: str  # hex-encoded 32-byte Ed25519 public key
    _private_key: Ed25519PrivateKey = field(default=None, repr=False)

    def sign(self, data: bytes) -> str:
        """Sign data and return hex-encoded signature."""
        sig_bytes = self._private_key.sign(data)
        return sig_bytes.hex()

    def verify(self, data: bytes, signature: str) -> bool:
        """Verify a hex-encoded signature against this oracle's public key."""
        return self.verify_with_public_key(self.world_public_key, data, signature)

    @staticmethod
    def verify_with_public_key(public_key_hex: str, data: bytes, signature: str) -> bool:
        """Verify using only the public key (for remote verification)."""
        try:
            pub_bytes = bytes.fromhex(public_key_hex)
            pub_key = Ed25519PublicKey.from_public_bytes(pub_bytes)
            sig_bytes = bytes.fromhex(signature)
            pub_key.verify(sig_bytes, data)
            return True
        except (InvalidSignature, ValueError):
            return False

    @classmethod
    def generate(cls) -> "OracleIdentity":
        """Generate a new random Ed25519 keypair."""
        private_key = Ed25519PrivateKey.generate()
        pub_bytes = private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        return cls(world_public_key=pub_bytes.hex(), _private_key=private_key)

    @classmethod
    def from_seed(cls, seed: int) -> "OracleIdentity":
        """Deterministic keypair from seed."""
        seed_bytes = hashlib.sha256(f"oracle-{seed}".encode()).digest()
        private_key = Ed25519PrivateKey.from_private_bytes(seed_bytes)
        pub_bytes = private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        return cls(world_public_key=pub_bytes.hex(), _private_key=private_key)
