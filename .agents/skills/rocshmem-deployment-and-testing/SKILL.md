---
name: rocshmem-deployment-and-testing
description: 用于部署、配置、验证、性能测试和排查面向 Hygon HCU 适配的 rocSHMEM。适用于 DTK/HCU 环境检查、IPC/GDA/RO 后端选择、HCU-NIC 拓扑、运行配置、功能验证、RMA/AMO/COLL 性能测试、日志解析与 Excel 报告导出及已确认问题排查；不用于通用的社区 ROCm 硬件指导。
---

# rocSHMEM 部署与测试

本 Skill 用于面向 Hygon HCU 适配的 rocSHMEM。平台部署说明使用 **HCU** 和 DTK；引用源码接口、工具名称及 `README-driver.md` 的通用 GPU 表述时保留原称谓。`GPU_TARGETS`、`rocminfo` 和 `ROCM_PATH` 是构建系统或工具链保留的字面名称。

## 事实来源

在给出版本、配置、命令、性能测试、验收或故障排查建议前，先定位并阅读 [`docs/rocshmem_deployment_and_testing_guide.md`](../../../docs/rocshmem_deployment_and_testing_guide.md) 的相关章节。查询 `driver.sh` 的命令行接口、套件、默认值、日志、重试或退出码时，读取 [`scripts/functional_tests/README-driver.md`](../../../scripts/functional_tests/README-driver.md)。解析测试日志或导出 Excel 报告时，读取 [`scripts/functional_tests/README-heatmap-parse-rocshmem-general.md`](../../../scripts/functional_tests/README-heatmap-parse-rocshmem-general.md)，并以配套脚本源码为准。不要在本 Skill 中复制这些文档的参数表。

按以下顺序判定信息权威性：

1. 某个后端、参数或接口是否由当前版本实现，以当前源码和构建脚本为准。
2. `driver.sh` 的参数解析、默认行为和输出，以脚本源码为准，`README-driver.md` 作为其使用说明。
3. Hygon HCU 环境下经验证的推荐值、操作流程和验收口径，以部署与测试指南为准。
4. 当前节点的软硬件版本、路径、NIC、PCI BDF 和拓扑，以现场检查结果为准。

根据任务读取对应章节：

- 后端选择、环境、编译、拓扑、内存、Context 和 QP 的通用原则：阅读“后端选择”至“与部署相关的编程约束”。
- 正确性验证、性能测试执行和结果对比：阅读“验证与性能测试”。
- `driver.sh` 的命令语法、套件名、默认行为、日志和退出码：阅读 `README-driver.md`，必要时对照脚本源码。
- 测试日志解析、Excel 热力图导出：阅读 `README-heatmap-parse-rocshmem-general.md`，核对输入日志格式和解析脚本；不必为单纯导出报告重读部署配置章节。
- 部署或运行故障：阅读“故障排查”，优先返回用户可执行的检查和处理方式；源码缺陷交由维护人员定位。
- 发布或生产上线准备：阅读“生产部署清单”。
- 全文评审或文档更新：阅读完整指南，并与当前源码树核对。

## 工作流程

1. 将请求分类为部署、配置、验证、性能测试、测试结果导出、故障排查或文档维护。
2. 检查当前代码库；当目标运行环境可用且在任务范围内时，同时检查环境。不得默认文档示例就是当前节点的实际配置。
3. 只读取相关章节；生成命令前，先确定后端、拓扑、进程数和测试类型。
4. 返回可直接执行的命令或明确配置。主机名、NIC 名称、PCI BDF、安装路径和输出路径等现场参数，必须标明为已检查值或清晰的占位符。
5. 说明当前使用的配置档位和验证前提。若指南、源码和实际环境存在冲突，应显式报告，不得默认选择其中一个。

## 输出要求

交付部署、测试或排查方案时，按任务需要说明：

- 适用的后端、provider、PE 数；跨节点 GDA 场景还需说明 HCU–NIC 拓扑。
- 哪些值已经检查，哪些仍是必须替换的现场值。
- 可直接执行的命令或配置，以及执行前提。
- 预期结果和明确的通过标准。
- 失败时应匹配的故障现象、检查项和处理方式。

## 必须保持的区分

- 生产配置与性能测试配置必须分开；除非明确说明原因，不得混用它们的对称堆、Context 或 QP 参数。
- 在同一后端和拓扑下通过正确性验证后，才能使用 `-noverif` 进行性能计时。
- 多节点 MPI 任务必须将所有必需运行时变量显式传递到远端 rank。
- 每个 PE 绑定一张 HCU；跨节点 GDA 部署须核验代码自动选择的 HCU–NIC 映射，不得仅根据设备编号推断物理亲和性。只有自动选择不符合实际拓扑或任务明确要求固定映射时，才使用强制拓扑文件覆盖。
- 优先给出已验证、可直接执行的建议，不给出无边界的试探性调参步骤。
- 环境变量和命令行选项的字面名称必须原样保留。
