from gary.api.auth import hash_password, new_session_token, verify_password


def test_password_hash_round_trip():
    password = "VeryStrongPassword123!"
    hashed = hash_password(password)
    assert hashed.startswith("scrypt$")
    assert verify_password(password, hashed) is True
    assert verify_password("wrong-password", hashed) is False


def test_new_session_token_shape():
    token = new_session_token()
    assert isinstance(token, str)
    assert len(token) >= 32
