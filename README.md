# WAL Store

A dependency-free Python reference implementation for storage, crash-recovery, key-value-store.

Run with: python3 demo.py
Tests: python3 -m unittest discover -s tests -v

## Scope

实现单写者预写日志键值存储：每次修改先写入日志，显式提交推进持久序号；进程在任意时刻被终止后，恢复出的状态必须恰好等于最后一次提交的状态。恢复输出、序号单调性、未提交记录和损坏日志拒绝行为都要稳定可测，不引入复制、压缩或外部服务。

进程在单条记录写入中途被终止时，日志末尾会留下残片（截断的 UTF-8 字节、缺少闭合结构的 JSON 前缀，或缺少记录终止符的完整记录字节）。记录终止符是一条记录可被恢复采用的边界：末尾缺少终止符的片段一律视为未完成写入被丢弃——不计入 pending_count、不推进序号、不影响查询。末尾之外的无效 UTF-8/JSON、空记录、字段或序号不合法的记录、非标准 JSON 常量和重复字段仍一律抛出 WalCorruptionError；无终止符但无法识别为一条完整合法记录的片段同样如此。恢复后的追加会先截掉残片，合法日志的字节内容不被改写。

`audit()` 是只读日志审计入口：严格按 recover 的 UTF-8、JSON、字段集合、序号连续性和记录终止符规则解析日志，把边界和片段暴露给调用方，但不改变任何入口的结果。返回值只含 `state`、`commit_seq`、`pending_count`、`valid_bytes`、`committed_bytes`、`tail_bytes` 六个字段：`state` 是独立深拷贝，前三个字段与 recover 完全一致；`valid_bytes` 是从文件开头到最后一个带终止符且被接受的记录末尾的字节数（已提交前缀加未提交 set/delete）；`committed_bytes` 是最后一个 commit 记录末尾的字节偏移，没有 commit 记录时为 0；`tail_bytes` 为文件字节数减 `valid_bytes`，只表示末尾未完成写入片段（未闭合的 JSON 前缀、缺少终止符但完整合法的一条记录，或截断的 UTF-8 字节），该片段不被应用、不计入 pending_count、不推进 commit_seq。空文件或不存在的日志返回空 state、序号和待处理数均为零、三个字节字段均为零。完整记录之后出现空行、尾随空白、非法 UTF-8、非标准 JSON 常量、重复字段、错误字段集合、序号断裂或无法识别为未完成写入的内容时，audit 统一抛出 WalCorruptionError，不返回部分结果，也不替换当前内存状态。audit 不截断、重写或追加日志，不改变 state、commit_seq、文件大小或后续 append 行为；同一日志前缀重复调用以及重新打开实例后结果一致。

`snapshot(target_seq=None)` 提供按已提交序号读取历史快照的只读入口：先按恢复规则完整解析当前日志（任何损坏统一抛出 WalCorruptionError，不返回部分快照），再重放截至目标提交的批次。省略 target_seq 时取最新提交，传 0 返回空状态；其他值必须是非布尔的非负整数且不大于最新提交序号，否则抛出 ValueError。目标之后的已提交批次和未提交尾部记录都不会出现在结果中。返回值只含 `state` 和 `commit_seq` 两个字段，state 是独立深拷贝，调用方修改不影响存储；同一日志和目标反复调用结果一致。snapshot 不追加、截断或重排日志，也不改变公开的 state、commit_seq。
