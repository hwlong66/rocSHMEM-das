# XDP 读写分离

本实现参考旧提交 `d3788dd1a96c7fb0635409efc053ff8d806bc810`，面向最新 GDA/IPC 数据路径，将地址转换和 MR key 选择集中处理。无需为普通、wave、wg、NBI、`_dp` 各自维护一套 HDP 接口。

## 使用

```cpp
// 所有 PE 以相同顺序、大小集体分配。
auto* send = static_cast<int*>(rocshmem_malloc_hdp(bytes));
auto* recv = static_cast<int*>(rocshmem_malloc(bytes));
// 将数据生产到 send；GPU 上调用任意支持的 put/get/集合通信接口。
// put(recv, send, ...) 或 get(recv, send, ...)：NIC 从 HDP 读、向 XDP 写。
rocshmem_free(send);
rocshmem_free(recv);
```

新增 `rocshmem_malloc_hdp(size)` 和 `rocshmem_align_hdp(alignment, size)`，统一通过现有 `rocshmem_free` 释放。原 `rocshmem_malloc/align/calloc/free` 的签名不变。关闭 XDP 或普通非 XDP allocator 下，HDP 分配退回原堆；RO 后端的新分配入口也退回原堆，本次的读写分离范围为 GDA/IPC。

沿用既有 XDP 检测：fine-grained allocator、DTK 对 XDP 分配标志的版本支持、`/sys/module/hycu/parameters/xdp_size`，并遵守 `ROCSHMEM_GDR_DISABLE_XDP`。只有实际选择 XDP 分配策略时才增设 HDP 堆。两个堆各自使用 `ROCSHMEM_HEAP_SIZE`，因此该模式的堆显存占用约为原来的两倍，另有两套 MR/IPC/window 管理开销。所有 PE 的分配配置应保持一致。

**两块堆是独立的数据存储，不是同一物理内存的别名。** `send` 和 `recv` 不会自动保持一致；要转发收到的数据，应显式从接收缓冲区复制/计算到 HDP 发送缓冲区，再满足原接口所需的同步。不能只把 XDP 指针减去堆基址、加上 HDP 堆基址，就把它当作相同数据的另一个视图。也不应为 GET/原子操作按操作类型强制切换另一块堆：操作必须访问调用者指定的对象。

## 适配边界

- 公共 `memory/heap_view.hpp` 根据对象所属堆转换对称地址，支持子指针、任意 PE/IPC 映射差值；无需假定不同 PE 的 XDP/HDP 地址差相同。
- GDA 的 typed/byte put/get、普通/wave/wg、blocking/NBI、显式/默认 context 共用地址转换和 QueuePair key 查询。原子与信号访问保留目标对象所在堆。wait/test 等本地访问直接使用原对象地址。
- 集合通信中的多 rail broadcast/reduce、alltoall/alltoallv 手写用户缓冲区偏移已改为公共转换；内部 pSync/pWrk 仍在其原堆，不为它们隐式创建镜像。集合算法的内部中间缓冲区或 in-place 操作可能继续从 XDP 读取，本实现不通过隐藏拷贝承诺所有内部读取都经过 HDP。
- HDP MR 注册和 rkey 交换复用最新多 NIC 流程，每个 QP 保存对应 PD 的 HDP lkey、对应远端 PE/rail 的 rkey；保留用户注册缓冲区和显式 key 接口的语义。
- MLX5/SHCA 的 `_dp` 快捷路径也按地址查询 key，MLX5 DP WQE 复用普通单 lane 实现的队列空间、CQ completion 和 key 字节序处理。
- MPI/TCP IPC 初始化、host/device 指针查询都覆盖双堆。SDMA 复用这些已转换的指针，其队列逻辑不变。
- 主机 MPI 路径每个 context 有两套 window，按远端对象选择；quiet/fence/barrier 完成两套 window 中的操作。已有不支持的接口不在此次实现中新增，例如 GDA 跨节点 scalar `g` 原有的未实现限制仍保留。

## 验证

独立地址转换测试无需 GPU：

```sh
c++ -std=c++17 -Wall -Wextra -Werror -D__host__= -D__device__= -Isrc \
  tests/standalone/heap_view_test.cpp -o heap_view_test
./heap_view_test
```

启用 `BUILD_EXAMPLES` 后，新增 `rocshmem_xdp_split_test`。它覆盖四种源/目标堆组合、子指针、默认/用户 context、普通/wave/wg put/get 和 NBI、六类 put-signal、fetch-add、broadcast/reduce/alltoall，并逐元素检查结果。`--dp` 增加 GDA MLX5/SHCA 的 DP put/atomic；`--host` 增加主机 MPI put/get NBI 和 quiet。默认运行不要求 MPI host RMA。`--mode N --verbose` 可定位单项操作；标量 DP 原子由单线程调用，显式 DP QP 的并发使用仍遵循原接口约束。

```sh
# 至少 2 个 PE。用实际项目的 launcher、设备与网络参数。
ROCSHMEM_MAX_NUM_CONTEXTS=2 mpirun -np 2 ./rocshmem_xdp_split_test
# GDA MLX5/SHCA，开启 DP 与 MPI host RMA 验证；本节点还需导出 ROCSHMEM_IB_GID_INDEX=1：
ROCSHMEM_MAX_NUM_CONTEXTS=2 mpirun -np 2 ./rocshmem_xdp_split_test --dp --host
# 非 XDP 回退：
ROCSHMEM_GDR_DISABLE_XDP=1 ROCSHMEM_MAX_NUM_CONTEXTS=2 \
  mpirun -np 2 ./rocshmem_xdp_split_test
```

实机矩阵还应包括：同节点 IPC、禁用 mixed IPC 后的 RDMA、跨节点、多 NIC/多 QP、MPI/TCP bootstrap、SDMA 开/关，以及原有 user-buffer/collective 功能测试。运行前选择空闲设备；NBI 源缓冲区的生命周期和所有排序要求与原接口一致。

本次正确基线为 `f56dda42070f42e9958334bb98b8217e7e09d070`，对应 D 盘 `rocshmem5/rocshmem` 与节点 `10.16.1.61` 的原始代码。顶层 CMake、版本配置、DAS/RCCL 开关不在本次修改范围内。普通接口及 Tile 接口复用当前的 `get_raddr_info/get_lkey/ipcPeerPtr`，保留对称注册表语义。

独立 CPU 地址转换测试已通过。节点 `hewl@10.16.1.61` / 容器 `hwl6` 中，原 `build` 目录的 GDA MLX5 完整构建及安装成功（gfx936）。四 PE 的原 IPC alltoall 参数 `-a 19 -w 16 -z 256`、IPC 双堆矩阵、关闭 XDP 的分配回退、GDA mixed IPC 双堆矩阵、MPI host RMA、alltoallv GET/COPY 均已通过。

测试将 `ROCSHMEM_HEAP_SIZE` 改为 `1073741824`（每堆 1 GiB）：节点存在其他显存占用，原 22949672960 字节配置开启双堆后无法容纳。此次不据此宣称性能提升。

GDA RoCE 测试须额外传入 `-x ROCSHMEM_IB_GID_INDEX=1`；`UCX_IB_GID_INDEX=1` 仅配置 UCX。缺少前者时，普通纯 RDMA 与 DP put 均出现 PE 0/1 数据为零；关闭 XDP 对照同样失败，另一设备对还出现 AH 建立超时。补齐 GDA GID 后，普通及 DP put 的四种堆组合全部通过。修正测试中的标量 DP 原子调用后，GDA mixed IPC 与禁用 mixed IPC 的纯 RDMA 全矩阵均通过（四 PE、四种源/目标堆组合、默认/用户 context、模式 0–23，包含 DP put/atomic）。

测试还发现并修复已有 WG put-signal 仅 leader 进入 WG 拷贝的问题，现由整个 WG 参与并在 signal 前完成 IPC release 和同步；blocking/NBI 两项均已通过。

跨节点、多 NIC/多 QP 专项、TCP bootstrap、SDMA 开/关和 SHCA 硬件尚未逐项实测；不能将 MLX5 单节点结果视为这些场景的通过证据。

## 原生 alltoall 用例

`tests/functional_tests/team_alltoall_tester.cpp`（`-a 19`）在默认 `USER_BUF_TYPE_HEAP` 模式下，通过公共 `alloc_test_source_buffer` 分配 HDP 发送缓冲区，接收缓冲区继续从默认堆分配。开启 XDP 时为 HDP 读、XDP 写；关闭 XDP 时 HDP 分配自动回退。显式选择的已注册用户缓冲区类型保持原行为，已有初始化、逐元素校验和释放逻辑不变。

容器 `/home/code5/rocshmem/build` 中可运行：

```sh
mpirun --allow-run-as-root --map-by numa --mca coll_hcoll_enable 0 -np 4 \
  -x ROCSHMEM_BACKEND=ipc -x LD_LIBRARY_PATH -x PATH \
  -x ROCSHMEM_TEST_UUID=1 -x ROCSHMEM_HEAP_SIZE=1073741824 \
  -x ROCSHMEM_MAX_NUM_CONTEXTS=32 -x UCX_IB_GID_INDEX=1 \
  tests/functional_tests/rocshmem_functional_tests -a 19 -w 16 -z 256
```

测试 GDA 时改为 `ROCSHMEM_BACKEND=gda` 并增加 `-x ROCSHMEM_IB_GID_INDEX=1`；关闭 XDP 的回退验证增加 `-x ROCSHMEM_GDR_DISABLE_XDP=1`。

实机原生 `-a 19 -w 16 -z 256` 的 IPC、GDA 和关闭 XDP 的 IPC 回退三项均已通过，消息大小 8 B–1 MiB，四 PE、每堆 1 GiB。最大消息处本次报告的带宽分别为 66.58 GB/s 和 57.97 GB/s；这是忙节点单次运行记录，不构成读写分离的性能提升结论。

## 全部 functional tests 的公共分配策略

公共策略位于 `tests/functional_tests/tester.hpp/.cpp`。用例仅声明缓冲区角色，不自行判断 XDP：

- `alloc_test_source_buffer(size, type)`：纯发送/读取源；对称堆类型选择 HDP。
- `alloc_test_buffer(size, type, role)`：目标或读写对象保持默认堆，默认角色为 `ReadWrite`。
- `alloc_test_rma_buffer(size, type, local)`：集中处理普通、默认 context、team context、wave、wg、flood、Tile 的 GET/PUT 本地/远端角色交换。
- 显式 host/device/fine/uncached/managed 用户缓冲区保留原分配和注册方式，不被角色覆盖。

默认 `ROCSHMEM_TEST_HEAP_MODE=split`；导出 `ROCSHMEM_TEST_HEAP_MODE=default` 可恢复全部角色使用原堆的对照模式。该选项与库的 `ROCSHMEM_GDR_DISABLE_XDP` 独立：前者仅控制测试缓冲区落点，不取消库的第二堆；后者控制库是否实际启用 XDP。两种堆均由现有释放入口处理。

已审计全部 functional test 分配点：源/目标分离的 collective、stream、host RMA、普通/DP/QP/SDMA、ping-pong、fence、device bitcode、Tile 用例统一标记源角色。原子目标、信号、wait/test、同步对象和 DP 往返标志属于读写对象，不创建镜像；只含这些对象的用例无需更改分配。初始化及结果校验仍访问各自真实存储。原地操作不会被隐式复制到另一块堆。

新增社区用例时，可继续使用原目标分配入口；发送源使用公共源入口，兼具源/目标用途的对象使用 `ReadWrite`。避免按变量名、分配顺序或接口名称自动猜测角色。

Tile RMA 同时修正原测试的缓冲区尺寸与跨步偏移：按实际 block 线程数统一分配、初始化和校验；row-major / arbitrary-stride 的线程间偏移使用实际物理跨度。原默认堆对照也可复现这些问题，修正不改变库的 Tile API。

2026-11-04 在 `10.16.1.61` / `hwl6` 验证：功能测试编译成功，159 项常规回归及 30 项 Tile 专项全部通过。配置为两 PE、每堆 1 GiB、一个 WG / 64 线程、最大消息 256 B、3 次计时迭代及 1 次 warmup；Tile 本身的固定矩阵负载仍为原测试尺寸。常规回归使用默认逻辑线程参数，专项额外指定 `-t 64`，两者现在均按真实 block 大小安全分配和校验。

覆盖 IPC、GDA、DP/QP、普通/wave/wg、stream、host RMA、collective、device bitcode，以及禁用 mixed IPC 的纯 RDMA、默认堆模式、关闭 XDP 回退和显式 fine-grained 用户缓冲区对照。测试覆盖发生源缓冲区变更的用例类别，未遍历所有尺寸和参数组合；SDMA 用例已接入公共策略，但当前构建未启用 SDMA，未实测该硬件路径。

节点明细位于 `build/functional-buffer-roles-results/results.json` 和 `build/functional-buffer-roles-tile-results/results.json`，各项日志与 JSON 同目录。早期 Tile 诊断失败已由上述修正后的完整重跑结果替代。

构建时若缓存的 LLVM optimizer 已失效，可通过构建目录的 CMake 缓存指定当前 DTK 的 `LLVM_OPT`；本次还清除了过期 ROCm 主/次/补丁版本缓存，使其与当前 HIP 7.2.26333 一致，并重建了不兼容的旧 bitcode/对象文件；未改动项目构建脚本。

## 后续同步社区代码

优先保留 `heap_view.hpp`、第二堆的生命周期、QueuePair key 查询和公共地址转换入口；不要把 HDP/XDP 标志逐层加入所有接口签名。社区新增手写对称地址偏移的接口时，应调用 `get_remote_ptr/get_local_ptr` 或 `ipcImpl_.remote_ptr`，不要再默认它只属于主堆。
