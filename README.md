# WAL Store

A dependency-free Python reference implementation for storage, crash-recovery, key-value-store.

Run with: python3 demo.py
Tests: python3 -m unittest discover -s tests -v

## Scope

实现单写者预写日志键值存储：每次修改先写入日志，显式提交推进持久序号；进程在任意时刻被终止后，恢复出的状态必须恰好等于最后一次提交的状态。恢复输出、序号单调性、未提交记录和损坏日志拒绝行为都要稳定可测，不引入复制、压缩或外部服务。

## 写入中断恢复

进程在单条记录写入中途被终止时，文件末尾会留下残片（截断的 UTF-8 字节，或仅在输入结束处才发现缺少闭合结构的 JSON 前缀）。恢复（构造、显式 `recover`，以及 `get`/`contains` 的内部读取使用同一套判定）丢弃这一残片：不计入 `pending_count`，不推进序号；此前完整的未提交记录仍正常统计。`recover` 成功后仅从文件末尾移除残片字节（合法内容逐字节保留），后续 `set`/`delete`/`commit` 从最后成功提交的序号继续形成连续序列。末尾空白、空记录、含换行的坏记录、可解析但字段或序号不合法的记录、非标准 JSON 常量、重复字段，以及末尾之外的任何无效 UTF-8 或 JSON，仍抛出 `WalCorruptionError`，且抛错时保留原有内存 state 和 commit_seq。
