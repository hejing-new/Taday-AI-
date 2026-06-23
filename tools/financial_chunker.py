"""
财报专用智能分块器 (Financial Report Chunker)

针对上市公司财报的结构特点，采用三层分块策略：

1. 章节切割 → 按财报目录结构（"一、二、三、..."）切分大段
2. 表格保护 → 检测表格区域，表格整体作为一个 chunk，不拆分
3. 语义细分 → 对非表格区域按语义边界做二次切分

相比 SemanticSplitterNodeParser 的优势：
- 中文优化：支持中文标点分句
- 表格感知：表格不拆分，保留完整数据
- 章节感知：章节边界天然就是 chunk 边界
- 自适应：叙述段 chunk 大（~1024 tokens），数据段 chunk 小（~512 tokens）
"""
import re
import io
import sys
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

# 确保 UTF-8
if hasattr(sys.stdout, 'buffer') and not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


@dataclass
class ChunkCandidate:
    """候选 chunk"""
    text: str
    page_start: int
    page_end: int
    is_table: bool = False
    section_name: str = ""
    token_count: int = 0


# 财报章节关键词（按常见顺序）
_SECTION_PATTERNS = [
    # 中文括号数字: （1）动力业务 / (4) 技术研发
    r'[（(]\s*[一二三四五六七八九十\d]+\s*[）)]\s*[\u4e00-\u9fff]{2,10}',
    # 中文数字顿号: 一、公司基本情况 / 二、经营情况
    r'[一二三四五六七八九十]+、\s*[\u4e00-\u9fff]{2,10}',
    # 第X节: 第二节 经营情况讨论
    r'第[一二三四五六七八九十\d]+节\s*[\u4e00-\u9fff]{2,10}',
    # 数字分级: 3.1 公司简介 / 3.2.1 财务数据
    r'\d+\.\d+\.?\d*\s+[\u4e00-\u9fff]{2,10}',
    # 附注: 附注四 / 附注4
    r'附注[一二三四五六七八九十\d]+',
]

# 表格指示词
_TABLE_INDICATORS = [
    '单位：千元', '单位：元', '单位：万元', '单位：百万元',
    '万元', '千元', '百万元', '亿元',
    '项目', '科目', '金额', '本期', '上期',
    '营业收入', '营业成本', '净利润', '资产', '负债',
    '合并资产负债表', '合并利润表', '合并现金流量表',
    '基本情况', '主要会计数据', '非经常性损益',
]


def _is_section_header(line: str) -> bool:
    """判断一行是否是章节标题"""
    line = line.strip()
    if not line or len(line) > 80:
        return False
    for pattern in _SECTION_PATTERNS:
        if re.match(pattern, line):
            return True
    return False


def _is_table_line(line: str) -> bool:
    """
    判断一行是否属于表格/数据行。
    严格模式：只识别明确的表格行，避免把叙事中的数字行误判。

    财报中三种可靠模式：
    1. 纯数字行: "5,978,096" / "(1,234)" / "41.85%"
    2. 标签+数字: "营业收入  84,704,589"（含金融指示词+数字）
    3. 分隔符行: 含 │ | \t 等表格线
    """
    line = line.strip()
    if not line or len(line) < 2:
        return False

    has_separator = any(c in line for c in ['│', '|', '┆', '┊', '\t'])
    has_indicator = any(kw in line for kw in _TABLE_INDICATORS)

    # 模式 1: 纯数字行 — 宽松匹配
    # "5,978,096" / "-1,234" / "41.85%" / "(123)" / "12.5%"
    cleaned = line.replace(',', '').replace('，', '').strip()
    is_pure_numeric = bool(re.match(
        r'^[（(]?\s*-?\d+\.?\d*\s*[）)]?\s*%?\s*$',
        cleaned
    ))
    if is_pure_numeric:
        return True

    # 模式 2: 含表格分隔符 — 确定是表格
    if has_separator:
        return True

    # 模式 3: 含金融指示词 + 含数字 — "营业收入 84,704,589"
    if has_indicator:
        digit_count = sum(1 for c in line if c.isdigit())
        if digit_count >= 4:
            return True

    return False


def _is_table_block(lines: List[str], start: int, window: int = 5) -> Tuple[bool, int]:
    """
    判断从 start 开始的 lines 是否构成表格块。
    返回 (is_table, end_index)。
    """
    if start >= len(lines):
        return False, start

    # 看后续 window 行中有多少行像表格
    table_like = 0
    end = start
    for i in range(start, min(start + window, len(lines))):
        if _is_table_line(lines[i]):
            table_like += 1
            end = i + 1
        elif lines[i].strip():  # 非空但不像表格
            break

    # 至少 3 行表格数据才算表格块
    if table_like >= 3:
        # 继续向后延伸，直到表格结束
        for i in range(end, len(lines)):
            if _is_table_line(lines[i]):
                end = i + 1
            elif not lines[i].strip():
                # 空行可能是表格间隔，继续看
                if i + 1 < len(lines) and _is_table_line(lines[i + 1]):
                    continue
                break
            else:
                break
        return True, end

    return False, start


def _split_sentences(text: str) -> List[str]:
    """中英文混合分句"""
    # 按中英文标点分句，保留标点
    sentences = re.split(
        r'(?<=[。！？；\n\.!?;])\s*',
        text
    )
    # 过滤空句
    return [s for s in sentences if s.strip()]


def _estimate_tokens(text: str) -> int:
    """
    快速估算 token 数。

    中文: 约 1.0-1.2 char/token（GPT 系列中文 token 率）
    英文: 约 4 char/token
    数字/标点: 按 1.5 char/token
    """
    if not text:
        return 0
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    other_chars = len(text) - chinese_chars - english_chars
    return int(chinese_chars / 1.1 + english_chars / 4 + other_chars / 2)


def chunk_financial_report(
    page_texts: List[Tuple[int, str]],
    max_chunk_tokens: int = 800,
    min_chunk_tokens: int = 100,
    overlap_tokens: int = 80,
) -> List[ChunkCandidate]:
    """
    财报专用分块主函数。

    Args:
        page_texts: [(page_num, text), ...] 按页码排序
        max_chunk_tokens: 最大 chunk 大小
        min_chunk_tokens: 最小 chunk 大小（小于则合并到上一个）
        overlap_tokens: chunk 间重叠 token 数

    Returns:
        分块结果列表
    """
    chunks: List[ChunkCandidate] = []

    # 第一步：按页解析，章节切割 + 表格保护
    for page_num, text in page_texts:
        lines = text.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i]

            # 检测章节标题 → 作为 chunk 边界
            if _is_section_header(line):
                # 如果有累积的候选，先保存
                section_name = line.strip()

                # 章节标题本身作为独立 chunk（带上下文）
                i += 1
                # 收集章节下的一段文字
                section_lines = []
                while i < len(lines) and not _is_section_header(lines[i]):
                    section_lines.append(lines[i])
                    i += 1
                    if _estimate_tokens('\n'.join(section_lines)) > max_chunk_tokens:
                        break

                section_text = '\n'.join(section_lines)
                if _estimate_tokens(section_text) > min_chunk_tokens:
                    chunks.append(ChunkCandidate(
                        text=section_text,
                        page_start=page_num,
                        page_end=page_num,
                        is_table=False,
                        section_name=section_name,
                    ))
                continue

            # 检测表格块
            is_tbl, end_idx = _is_table_block(lines, i)
            if is_tbl:
                table_text = '\n'.join(lines[i:end_idx])
                chunks.append(ChunkCandidate(
                    text=table_text,
                    page_start=page_num,
                    page_end=page_num,
                    is_table=True,
                    section_name="",
                ))
                i = end_idx
                continue

            # 普通段落：收集到下一个章节/表格/空行
            para_lines = []
            while i < len(lines):
                l = lines[i]
                if _is_section_header(l):
                    break
                is_tbl, _ = _is_table_block(lines, i, window=3)
                if is_tbl:
                    break
                para_lines.append(l)
                i += 1
                # 空行可能是段落分隔
                if not l.strip() and para_lines:
                    break

            para_text = '\n'.join(para_lines).strip()
            if para_text and _estimate_tokens(para_text) > min_chunk_tokens:
                chunks.append(ChunkCandidate(
                    text=para_text,
                    page_start=page_num,
                    page_end=page_num,
                    is_table=False,
                    section_name="",
                ))

    # 第二步：对超长非表格 chunk 做语义细分
    final_chunks: List[ChunkCandidate] = []
    for chunk in chunks:
        if chunk.is_table:
            # 表格不拆分，但超大表格截断
            if _estimate_tokens(chunk.text) > max_chunk_tokens * 2:
                sub_chunks = _split_table_chunk(chunk, max_chunk_tokens)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        else:
            if _estimate_tokens(chunk.text) > max_chunk_tokens:
                sub_chunks = _split_semantic(chunk, max_chunk_tokens, overlap_tokens)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

    # 第三步：合并过小的 chunk
    merged = _merge_small_chunks(final_chunks, min_chunk_tokens)

    # 计算 token 数
    for c in merged:
        c.token_count = _estimate_tokens(c.text)

    return merged


def _split_semantic(chunk: ChunkCandidate, max_tokens: int, overlap: int) -> List[ChunkCandidate]:
    """对长段落做语义细分（按句子边界）"""
    sentences = _split_sentences(chunk.text)
    if len(sentences) <= 1:
        return [chunk]

    results: List[ChunkCandidate] = []
    current_lines: List[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _estimate_tokens(sent)

        if current_tokens + sent_tokens > max_tokens and current_lines:
            # 保存当前 chunk
            results.append(ChunkCandidate(
                text='\n'.join(current_lines),
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                is_table=False,
                section_name=chunk.section_name,
            ))
            # 保留 overlap
            if overlap > 0:
                overlap_lines = []
                overlap_tokens = 0
                for l in reversed(current_lines):
                    overlap_lines.insert(0, l)
                    overlap_tokens += _estimate_tokens(l)
                    if overlap_tokens >= overlap:
                        break
                current_lines = overlap_lines
                current_tokens = overlap_tokens
            else:
                current_lines = []
                current_tokens = 0

        current_lines.append(sent)
        current_tokens += sent_tokens

    # 最后一段
    if current_lines:
        results.append(ChunkCandidate(
            text='\n'.join(current_lines),
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            is_table=False,
            section_name=chunk.section_name,
        ))

    return results


def _split_table_chunk(chunk: ChunkCandidate, max_tokens: int) -> List[ChunkCandidate]:
    """大表格按行切分，保留表头"""
    lines = chunk.text.split('\n')
    if len(lines) <= 4:
        return [chunk]

    # 第一行通常是表头
    header = lines[0]
    data_lines = lines[1:]

    results = []
    current_lines = [header]
    current_tokens = _estimate_tokens(header)

    for line in data_lines:
        line_tokens = _estimate_tokens(line)
        if current_tokens + line_tokens > max_tokens and len(current_lines) > 1:
            results.append(ChunkCandidate(
                text='\n'.join(current_lines),
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                is_table=True,
                section_name=chunk.section_name,
            ))
            current_lines = [header]  # 新 chunk 也带表头
            current_tokens = _estimate_tokens(header)

        current_lines.append(line)
        current_tokens += line_tokens

    if len(current_lines) > 1:
        results.append(ChunkCandidate(
            text='\n'.join(current_lines),
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            is_table=True,
            section_name=chunk.section_name,
        ))

    return results


def _merge_small_chunks(chunks: List[ChunkCandidate], min_tokens: int) -> List[ChunkCandidate]:
    """
    合并过小的 chunk 到相邻 chunk。

    规则：
    - 表格 chunk 绝不和段落 chunk 合并（保持表格完整性）
    - 同类型的小 chunk 合并到最近的较大邻居
    - 只合并 < min_tokens 的 chunk，不触发连锁反应
    - 使用 group-by 避免重复：每个 target 只合并一次
    """
    if not chunks:
        return chunks

    # 按类型分组，分别处理
    def merge_group(group: List[tuple]) -> List[ChunkCandidate]:
        """
        合并一组同类型的 (original_index, chunk)。

        策略：贪心合并相邻 chunk，直到达到 min_tokens。
        优先合并相邻的小 chunk，形成 ~min_tokens 的 chunk。
        """
        if len(group) <= 1:
            return [c for _, c in group]

        # 按原始顺序排列
        group_sorted = sorted(group, key=lambda x: x[0])

        result: List[ChunkCandidate] = []
        accumulator: Optional[ChunkCandidate] = None

        for _, chunk in group_sorted:
            if accumulator is None:
                accumulator = ChunkCandidate(
                    text=chunk.text,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    is_table=chunk.is_table,
                    section_name=chunk.section_name,
                )
                accumulator.token_count = _estimate_tokens(accumulator.text)
            else:
                # 合并到 accumulator
                accumulator.text += '\n\n' + chunk.text
                accumulator.page_start = min(accumulator.page_start, chunk.page_start)
                accumulator.page_end = max(accumulator.page_end, chunk.page_end)
                accumulator.token_count = _estimate_tokens(accumulator.text)

            # 如果 accumulator 达到了 min_tokens，保存并重置
            if accumulator.token_count >= min_tokens:
                result.append(accumulator)
                accumulator = None

        # 处理剩余的 accumulator
        if accumulator is not None:
            if result:
                # 合并到最后一个 chunk
                result[-1].text += '\n\n' + accumulator.text
                result[-1].page_end = max(result[-1].page_end, accumulator.page_end)
                result[-1].token_count = _estimate_tokens(result[-1].text)
            else:
                # 整个组都 < min_tokens，保留为一个 chunk
                result.append(accumulator)

        return result

    # 按类型分组（保持原始顺序）
    text_group = [(i, c) for i, c in enumerate(chunks) if not c.is_table]
    table_group = [(i, c) for i, c in enumerate(chunks) if c.is_table]

    merged_text = merge_group(text_group)
    merged_tables = merge_group(table_group)

    # 合并并排序
    all_chunks = merged_text + merged_tables
    # 按 page_start 排序保持文档顺序
    all_chunks.sort(key=lambda c: c.page_start)

    return all_chunks
