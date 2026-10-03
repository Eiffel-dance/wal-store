# WAL Store

A dependency-free Python reference implementation for storage, crash-recovery, key-value-store.

Run with: python3 demo.py
Tests: python3 -m unittest discover -s tests -v

## Scope

实现单写者预写日志键值存储：每次修改先写入日志，显式提交推进持久序号；进程在任意时刻被终止后，恢复出的状态必须恰好等于最后一次提交的状态。恢复输出、序号单调性、未提交记录和损坏日志拒绝行为都要稳定可测，不引入复制、压缩或外部服务。

进程在单条记录写入中途被终止时，日志末尾会留下残片（截断的 UTF-8 字节、缺少闭合结构的 JSON 前缀，或缺少记录终止符的完整记录字节）。记录终止符是一条记录可被恢复采用的边界：末尾缺少终止符的片段一律视为未完成写入被丢弃——不计入 pending_count、不推进序号、不影响查询。末尾之外的无效 UTF-8/JSON、空记录、字段或序号不合法的记录、非标准 JSON 常量和重复字段仍一律抛出 WalCorruptionError；无终止符但无法识别为一条完整合法记录的片段同样如此。恢复后的追加会先截掉残片，合法日志的字节内容不被改写。

`audit()` 提供只读日志审计入口，把可接受的边界和末尾片段暴露给调用方，不改变任何入口的结果。它严格按照 recover 的 UTF-8、JSON、字段集合、序号连续性和记录终止符规则解析日志，但不采纳重放结果：只返回 `state`、`commit_seq`、`pending_count`（三者与 recover 完全一致，state 为独立深拷贝）以及 `valid_bytes`、`committed_bytes`、`tail_bytes` 三个字节字段。`valid_bytes` 是从文件开头到最后一个带终止符且被接受的记录末尾的字节数，包含已提交前缀和未提交的 set/delete 记录；`committed_bytes` 是最后一个 commit 记录末尾的字节偏移，没有 commit 记录时为零；`tail_bytes` 等于文件字节数减 `valid_bytes`，只表示末尾未完成的写入片段。空文件或不存在的日志返回空 state、序号与待处理数均为零、三个字节字段均为零。末尾缺少终止符的片段，只有在可判定为未完成 JSON 前缀，或为通过结构、字段和序号校验的完整记录时才计入 `tail_bytes`；它不会被应用、计入 `pending_count` 或推进 `commit_seq`，截断的 UTF-8 字节也仅作为片段统计。完整记录之后出现空行、尾随空白、非法 UTF-8、非标准 JSON 常量、重复字段、错误字段集合、序号断裂或其他无法识别的非未完成内容时，audit 统一抛出 WalCorruptionError，不返回部分结果，也不替换当前内存状态。audit 不截断、不重写、不追加日志，不改变 state、commit_seq、文件大小或后续 append 行为；对同一日志前缀重复调用以及重新打开实例后结果一致。

`repair_tail()` 提供运维主动清理日志末尾未完成写入片段的入口：先按 recover/audit 完全相同的 UTF-8、JSON、记录字段、序号连续性、终止符与重复字段规则校验整个日志（任何无法证明为单个末尾未完成片段的内容——空行、尾随空白、非法 UTF-8、非标准 JSON 常量、重复字段、字段集合错误、序号跳变、未知操作等——一律抛出 WalCorruptionError，文件、内存 state 与 commit_seq 均不变），只有末尾确实存在未完成片段时才将日志就地截断到 audit 报告的 `valid_bytes`：此前任何字节不被改写，已经完整写入（带终止符）的 set/delete 记录——包括未提交的——不会被删除，仍保持 pending 并继续遵循 rollback 语义。截断经 flush 与 fsync 持久化后才采纳重放结果；打开、截断或持久化失败传播 OSError 且不改变内存提交状态。返回 `state`（清理后可重放状态的独立深拷贝）、`commit_seq`、`pending_count` 和 `removed_bytes`（本次从文件末尾移除的原始 tail 字节数）。清理成功后重新打开同一路径，恢复结果与清理前 recover() 对可接受前缀的结果完全一致，尾部片段不再出现；没有文件、空文件或没有尾部片段时 `removed_bytes` 为零且不创建文件。

`snapshot(target_seq=None)` 提供按已提交序号读取历史快照的只读入口：先按恢复规则完整解析当前日志（任何损坏统一抛出 WalCorruptionError，不返回部分快照），再重放截至目标提交的批次。省略 target_seq 时取最新提交，传 0 返回空状态；其他值必须是非布尔的非负整数且不大于最新提交序号，否则抛出 ValueError。目标之后的已提交批次和未提交尾部记录都不会出现在结果中。返回值只含 `state` 和 `commit_seq` 两个字段，state 是独立深拷贝，调用方修改不影响存储；同一日志和目标反复调用结果一致。snapshot 不追加、截断或重排日志，也不改变公开的 state、commit_seq。

`history(since_seq=0, until_seq=None)` 提供按持久序号导出已提交修改批次的只读入口，作为恢复与快照之外的历史变更视图。先校验参数形式：since_seq 必须是非布尔的非负整数，until_seq 提供时（None 表示最新提交序号）同样必须是非布尔的非负整数且满足 since_seq <= until_seq；类型错误、负数或区间倒置统一抛出 ValueError。随后按 recover 完全相同的规则完整校验日志（UTF-8、JSON、字段集合、重复字段、序号连续性、终止符与 JSON 值），末尾可判定为中断写入的片段按 recover 规则忽略，已写入但未提交的 set/delete 只留在日志中；其他损坏统一抛出 WalCorruptionError，不返回部分结果。解析得出最新提交序号后，since_seq 大于最新序号或 until_seq 超过最新序号也统一抛出 ValueError；不存在或空日志的默认调用返回空数组。返回按提交顺序排列的数组，仅纳入 since_seq < commit_seq <= until_seq 的提交，每项为 `{"commit_seq": seq, "changes": [...]}`：changes 保留该批次日志中的写入顺序，set 项含 op、key、value，delete 项含 op、key，空提交返回空 changes。返回对象及嵌套值都是独立深拷贝，调用方修改不影响存储；history 不追加、截断、重排或改写日志，不改变公开的 state、commit_seq、pending_count 或现有入口行为，同一日志前缀反复调用和重新打开实例后结果一致。

`pending_changes()` 提供只读入口，查看最后一次完整提交之后、已完整写入日志但尚未进入持久状态的待处理批次，便于调用方在决定提交或回滚前检查。按 recover 的全部规则把日志解析到可接受边界：校验 UTF-8、JSON 对象、重复字段、各操作的字段集合、正整数序号、序号连续性、键和值类型以及记录终止符。末尾能够证明是中断写入的 JSON 前缀、无终止符的完整记录或截断 UTF-8 字节，沿用 recover 的丢弃规则，不计入结果；已经完整写入（带终止符）的 set 和 delete 记录即使没有 commit 也按日志顺序保留。任何超出可丢弃末尾片段范围的非法 UTF-8、非标准 JSON 常量、空记录、重复键、错误字段、未知操作、序号跳变或类型不合法，都统一抛出 WalCorruptionError，不返回部分结果，也不改变 state、commit_seq，不截断或追加文件。返回 RecoveryResult，含 `commit_seq`（最后一次完整 commit 的序号，没有时为 0）、`pending_count`（恒等于 changes 长度）和 `changes`；没有待处理记录时分别返回当前序号、零和空数组。changes 中 set 项保留 op、key、value，delete 项保留 op、key（不暴露 seq），所有嵌套值都是独立深拷贝。该入口不追加、截断或重写日志，不让查询看到未提交修改，不消耗序号；调用方修改返回对象后，存储、后续 commit、rollback、重新打开及重复调用的结果仍保持一致，audit、history、snapshot 与 repair_tail 的可观察行为不变。

`WalStore(path, exclusive=True)` 在打开时为日志路径申请独占写入租约，让部署可以明确落实单写者约束：租约在实例存活期间覆盖 set、delete、commit、rollback 和 repair_tail。租约参数必须是布尔值，否则抛出 ValueError；`exclusive=False` 或省略时行为与现有默认构造完全一致，查询类方法保持只读。租约通过对日志文件本身加 `flock(LOCK_EX|LOCK_NB)` 实现，在读取日志之前获取：相同规范化路径（同一 inode，含 `.`/`..` 或符号链接别名）已被其他独占实例持有时，新实例抛出唯一的 WalBusyError——不创建、不截断、不追加日志，也不改变持有方实例的 state 与 commit_seq，冲突期间每次失败结果都可重复观察。锁隶属于打开的文件描述符，实例关闭或进程异常终止时由内核自动释放，不会留下阻塞后续打开的失效状态；另一个进程随后以 `exclusive=True` 打开同一路径时按原有规则恢复最后一次提交，完整未提交记录仍只计入 pending，未完成尾片与损坏日志的判断及 WalCorruptionError 结果保持不变。`close()` 释放租约，可重复调用；实例也支持上下文管理器（`with WalStore(path, exclusive=True) as s:`）自动释放。关闭后再调用除 close 外的任一公开方法（recover、set、delete、commit、rollback、get、contains、snapshot、history、pending_changes、audit、repair_tail）都抛出唯一的 WalClosedError。独占模式不改写 WAL 记录结构或提交顺序，写入与 fsync 失败仍沿用现有 OSError 语义且内存状态不变；成功关闭后再次获取租约，既有 state、commit_seq、snapshot、history、audit 和 repair_tail 的结果均不改变。
