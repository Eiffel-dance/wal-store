# WAL Store

A dependency-free Python reference implementation for storage, crash-recovery, key-value-store.

Run with: python3 demo.py
Tests: python3 -m unittest discover -s tests -v

## Scope

实现单写者预写日志键值存储：每次修改先写入日志，显式提交推进持久序号；进程在任意时刻被终止后，恢复出的状态必须恰好等于最后一次提交的状态。恢复输出、序号单调性、未提交记录和损坏日志拒绝行为都要稳定可测，不引入复制、压缩或外部服务。

进程在单条记录写入中途被终止时，日志末尾会留下残片（截断的 UTF-8 字节、缺少闭合结构的 JSON 前缀，或字节完整但终止换行从未落盘的整条记录）。记录终止符是一条记录可被恢复采用的边界：恢复时丢弃该残片——不计入 pending_count、不推进序号、不改变状态；末尾之外的无效 UTF-8/JSON、空记录、字段或序号不合法的记录（即使缺少终止符）、非标准 JSON 常量和重复字段仍一律抛出 WalCorruptionError。恢复后的追加会先截掉残片，合法日志的字节内容不被改写。
