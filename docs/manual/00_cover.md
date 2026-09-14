@TITLE: Deep Research Agent 项目代码级技术说明
@SUBTITLE: 多智能体研究助手架构、检索、工具调用、评测与 GPU 实现
@META: 项目名称：deep-research-agent
@META: 代码目录：C:\Users\29716\Documents\demo1\deep-research-agent
@META: 远程仓库：https://github.com/llsosu-zxy/deep-research-agent
@META: 文档日期：2026-09-14
@META: 目标读者：项目作者、面试官、希望复现或二次开发的工程师

本文档解释 deep-research-agent 的完整实现，而不是只介绍概念。内容覆盖目录结构、配置系统、数据模型、混合检索、工具注册表、Guardrails、LangGraph 编排、LLM 客户端、FastAPI 与 Gradio、评测体系、GPU 路径、测试体系、运行手册以及当前实现的边界与限制。每个章节都会指出对应文件和关键类或函数，并给出必要代码片段。

阅读本文档时，建议先看第一章的总体调用链，再按目录结构跳到具体模块。只想运行项目时，直接看第十章运行手册；准备面试讲解时，重点看第一章、第六章、第九章和第十二章。

@PAGEBREAK

@TOC
