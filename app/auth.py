"""管理写接口的 HMAC-SHA256 防重放认证。

受保护接口（策略新增/更新、回滚、定时变更的创建/取消）必须携带：

    X-Key-Id     服务端 ADMIN_KEY_ID
    X-Timestamp  请求时刻（Unix 秒，整数）
    X-Nonce      每次请求唯一的随机串
    X-Signature  小写十六进制 HMAC-SHA256

签名原文（换行符为 ``\\n``）::

    <METHOD>
    <原始路径及查询串>
    <请求体 SHA-256 十六进制>
    <timestamp>
    <nonce>

安全要点：

- 密钥与 key id 取自环境变量 ``ADMIN_KEY_ID`` / ``ADMIN_HMAC_SECRET``；
  签名与 key id 均用 ``hmac.compare_digest`` 常量时间比较；
- ``abs(服务端当前时间 - X-Timestamp) > 300`` 秒一律 401；
- 认证通过的 nonce 在 SQLite 中保留 300 秒，同一 key 重放返回 409；
- nonce 登记与业务写入在**同一个 BEGIN IMMEDIATE 事务**内提交
  （``admin_write_tx``），业务失败整体回滚，不会占用 nonce。

GET、``/decide`` 裁决与 ``/decide/{request_id}/revoke`` 撤销接口不鉴权。
"""

from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from fastapi import HTTPException, Request

# 时间戳允许偏离当前时间的秒数；同时也是 nonce 在库中的保留时长
SKEW_SECONDS = 300


@dataclass(frozen=True)
class AdminAuth:
    """鉴权通过后携带给业务层的调用方信息。"""

    key_id: str
    nonce: str
    timestamp: int


def check_admin_config() -> None:
    """启动时确认管理凭证已配置；缺失则拒绝启动，避免管理接口裸奔。"""
    missing = [
        name
        for name in ("ADMIN_KEY_ID", "ADMIN_HMAC_SECRET")
        if not os.environ.get(name)
    ]
    if missing:
        raise RuntimeError(
            "管理写接口需要配置环境变量 "
            + "、".join(missing)
            + "（HMAC-SHA256 防重放认证）"
        )


def _admin_credentials() -> tuple[str, bytes]:
    # 每个请求即时读取，便于测试 reload；正常启动已由 check_admin_config 兜底
    key_id = os.environ.get("ADMIN_KEY_ID", "")
    secret = os.environ.get("ADMIN_HMAC_SECRET", "")
    if not key_id or not secret:
        raise HTTPException(503, "管理接口凭证未配置")
    return key_id, secret.encode("utf-8")


def _unauthorized() -> HTTPException:
    # 统一文案，不区分缺失头 / key 不符 / 签名错，避免泄露判定细节
    return HTTPException(401, "认证无效：X-Key-Id / X-Timestamp / X-Nonce / X-Signature 缺失或不合法")


async def require_admin(request: Request) -> AdminAuth:
    """FastAPI 依赖：校验时间戳窗口与 HMAC-SHA256 签名。

    仅做认证（401）；nonce 占用与重放（409）在 ``admin_write_tx`` 中处理，
    以便与业务写入共用同一事务。
    """
    key_id_h = request.headers.get("x-key-id", "")
    timestamp_h = request.headers.get("x-timestamp", "")
    nonce = request.headers.get("x-nonce", "")
    signature_h = request.headers.get("x-signature", "")

    # 任一头缺失或为空 -> 401（不用 Header(...)，避免缺头被 FastAPI 回成 422）
    if not key_id_h or not timestamp_h or not nonce or not signature_h:
        raise _unauthorized()

    # X-Timestamp：Unix 秒整数
    try:
        timestamp = int(timestamp_h)
    except ValueError:
        raise _unauthorized() from None

    expected_key_id, secret = _admin_credentials()
    # key id 常量时间比较；按字节比较，非 ASCII（latin-1 解码出的畸形头值）
    # 也只会安全地返回不匹配，而不会让 compare_digest 抛 TypeError（->500）
    if not hmac.compare_digest(
        key_id_h.encode("latin-1", "replace"), expected_key_id.encode("utf-8")
    ):
        raise _unauthorized()

    if abs(time.time() - timestamp) > SKEW_SECONDS:
        raise HTTPException(401, "时间戳已过期或偏离当前时间超过 300 秒")

    body = await request.body()
    body_sha = hashlib.sha256(body).hexdigest()
    # 签名绑定「原始」请求目标：raw_path / query_string 均为客户端实际发送的
    # 原始字节（scope["path"] 已做百分号解码，不能用于验签，否则存在
    # 编码规范化歧义）。latin-1 可无损还原字节。
    raw_path = request.scope.get("raw_path")
    if raw_path is None:
        raw_path = request.scope["path"].encode("latin-1")
    raw_query = request.scope.get("query_string", b"")
    path_and_query = raw_path.decode("latin-1")
    if raw_query:
        path_and_query += "?" + raw_query.decode("latin-1")
    signing_text = "\n".join(
        [request.method, path_and_query, body_sha, timestamp_h, nonce]
    )
    expected_sig = hmac.new(
        secret, signing_text.encode("utf-8"), hashlib.sha256
    ).hexdigest()  # 小写十六进制
    if not hmac.compare_digest(
        signature_h.encode("latin-1", "replace"), expected_sig.encode("ascii")
    ):
        raise _unauthorized()

    return AdminAuth(key_id=key_id_h, nonce=nonce, timestamp=timestamp)


def purge_expired_nonces(conn: sqlite3.Connection, now: int | None = None) -> None:
    """删除保留期已过的 nonce；由每个管理写事务在登记前顺带执行。"""
    current = int(time.time()) if now is None else now
    conn.execute("DELETE FROM admin_nonces WHERE expires_at <= ?", (current,))


def count_nonces(
    conn: sqlite3.Connection, key_id: str | None = None
) -> int:
    """统计在保留期内的 nonce（测试/排障用）。"""
    if key_id is None:
        row = conn.execute("SELECT COUNT(*) AS n FROM admin_nonces").fetchone()
    else:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM admin_nonces WHERE key_id=?", (key_id,)
        ).fetchone()
    return int(row["n"])


@contextmanager
def admin_write_tx(
    conn: sqlite3.Connection, auth: AdminAuth
) -> Iterator[sqlite3.Connection]:
    """管理写接口的统一事务。

    在单个 ``BEGIN IMMEDIATE`` 事务内：
      1. 清除保留期已过（> 300 秒）的 nonce；
      2. 登记本次 nonce（同一 key_id 重放 -> 唯一约束冲突 -> 409）；
      3. 执行业务写入并 COMMIT。

    业务抛异常（含 404/409/422 等 HTTPException）时整体 ROLLBACK，
    nonce 不会留下，客户端可用同一 nonce 修正后重试。
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        now = int(time.time())
        conn.execute("DELETE FROM admin_nonces WHERE expires_at <= ?", (now,))
        try:
            conn.execute(
                "INSERT INTO admin_nonces (key_id, nonce, ts, seen_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    auth.key_id,
                    auth.nonce,
                    auth.timestamp,
                    now,
                    now + SKEW_SECONDS,
                ),
            )
        except sqlite3.IntegrityError:
            # (key_id, nonce) 已被仍在保留期内的成功请求占用
            raise HTTPException(409, "nonce 已被使用，请为每个请求生成新的 nonce")
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
