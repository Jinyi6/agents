import arxiv
import csv
import time
import os
import asyncio
import logging
import re
from datetime import datetime, date
from pathlib import Path
import shutil
from openai import AsyncOpenAI

# 假设您的 config.py 文件在同一个目录下
from config import (
    background_tasks, outputs_dir, workspace_dir,
    OPENAI_API_KEY, OPENAI_API_BASE, MODEL_NAME,
    MAX_CONCURRENT_TRANSLATIONS
)

def build_arxiv_query(keyword_phrase: str) -> str:
    """
    为给定的关键词短语构建一个高级查询字符串，同时搜索摘要(abs)和标题(ti)。
    - 对于特殊配置的短语，使用预设的 AND/OR 组合。
    - 对于其他短语，将所有单词用 AND 连接，每个单词都同时匹配标题或摘要。
    """
    _ABS_OR_TI_LLM = '((abs:LLM OR ti:LLM) OR (abs:"Large Language Model" OR ti:"Large Language Model"))'
    _ABS_OR_TI_RL = '((abs:RL OR ti:RL) OR (abs:"Reinforcement Learning" OR ti:"Reinforcement Learning"))'
    
    _SPECIAL_PHRASE_CONFIG = {
        "large language model agent rl": [_ABS_OR_TI_LLM, '(abs:agent OR ti:agent)', _ABS_OR_TI_RL],
        "llm rft": [_ABS_OR_TI_LLM, '(abs:RFT OR ti:RFT)'],
        "llm reinforcement learning finetuning": [_ABS_OR_TI_LLM, _ABS_OR_TI_RL, '(abs:Finetuning OR ti:Finetuning)'],
        "large language model rl": [_ABS_OR_TI_LLM, _ABS_OR_TI_RL]
    }
    
    phrase_lower = keyword_phrase.lower().strip()
    if phrase_lower in _SPECIAL_PHRASE_CONFIG:
        query_parts = _SPECIAL_PHRASE_CONFIG[phrase_lower]
        return f"({' AND '.join(query_parts)})"
    else:
        words = [word for word in keyword_phrase.split() if word]
        if not words:
            return ""
        
        query_parts = [f'(abs:{word} OR ti:{word})' for word in words]
        return f"({' AND '.join(query_parts)})"

def search_arxiv_by_date_range(keywords, start_date_str, end_date_str, max_results, process_log):
    """根据日期范围从 arXiv 检索论文，并为长关键词组执行补充搜索。"""
    unique_papers = {}
    try:
        filter_start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        filter_end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        raise ValueError(f"日期格式无效，请使用 YYYY-MM-DD。")

    if filter_start_date > filter_end_date:
        raise ValueError("起始日期不能晚于结束日期。")

    process_log.append(f"INFO: 开始检索，日期范围: {start_date_str} 到 {end_date_str}")
    logging.info(f"开始检索，日期范围: {start_date_str} 到 {end_date_str}")

    def _perform_search(query_keyword, original_keyword_for_result):
        advanced_query = build_arxiv_query(query_keyword)
        if not advanced_query:
            process_log.append(f"INFO: 查询关键词 '{query_keyword}' 为空，已跳过。")
            return
            
        log_message_prefix = "补充" if query_keyword != original_keyword_for_result else ""
        process_log.append(f"INFO: 正在执行{log_message_prefix}搜索 '{query_keyword}' (查询: {advanced_query})")
        logging.info(f"正在执行{log_message_prefix}搜索 '{query_keyword}' (查询: {advanced_query})")
        
        try:
            search = arxiv.Search(
                query=advanced_query,
                max_results=max_results * 2,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            retrieved_count = 0
            for result in search.results():
                if retrieved_count >= max_results: break
                paper_date = result.published.date()
                if filter_start_date <= paper_date <= filter_end_date:
                    if result.entry_id not in unique_papers:
                        unique_papers[result.entry_id] = {
                            "title": result.title,
                            "published_date": paper_date.strftime("%Y-%m-%d"),
                            "summary_en": result.summary.replace("\n", " "),
                            "authors": [author.name for author in result.authors],
                            "arxiv_link": result.entry_id,
                            "pdf_link": result.pdf_url,
                            "original_keyword": original_keyword_for_result
                        }
                        retrieved_count += 1
            process_log.append(f"SUCCESS: {log_message_prefix}搜索 '{query_keyword}' 找到 {retrieved_count} 篇新论文。")
        except Exception as e:
            logging.error(f"搜索关键词 '{query_keyword}' 时出错: {e}")
            process_log.append(f"WARNING: 搜索关键词 '{query_keyword}' 时出错: {e}")

    for keyword in keywords:
        _perform_search(keyword, keyword)
        time.sleep(3)

        keyword_parts = keyword.split()
        if len(keyword_parts) > 3:
            supplementary_keyword = " ".join(keyword_parts[:-1])
            process_log.append(f"INFO: 关键词 '{keyword}' 包含超过3个词，将执行一次补充搜索。")
            
            _perform_search(supplementary_keyword, keyword)
            time.sleep(3)

    total_found = len(unique_papers)
    process_log.append(f"SUCCESS: 所有关键词检索完成，共找到 {total_found} 篇不重复的论文。")
    logging.info(f"所有关键词检索完成，共找到 {total_found} 篇不重复的论文。")
    return list(unique_papers.values())


async def translate_one_abstract(aclient, abstract_en, target_language, semaphore):
    """使用 LLM 异步翻译单个摘要。"""
    if not abstract_en or not abstract_en.strip(): return ""
    async with semaphore:
        try:
            prompt_content = f"Please translate the following academic abstract into {target_language}. Keep the original formatting and technical terms. Abstract: "
            response = await aclient.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": prompt_content},
                    {"role": "user", "content": abstract_en}
                ],
                temperature=0.2,
                max_tokens=4000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"翻译摘要时出错: {e}")
            return f"翻译错误: {e}"

def sanitize_filename_part(text: str) -> str:
    """清理字符串，使其可安全地用于文件名。"""
    text = re.sub(r'[\s,]+', '_', text)
    text = re.sub(r'[^\w\-_.]', '', text)
    return text[:50]

async def run_arxiv_search_and_process(run_id: str, request_params: dict):
    """后台任务的主执行函数：搜索、翻译、保存。"""
    process_log = background_tasks[run_id]['summary']
    work_dir = workspace_dir / f"work_dir_{run_id}"
    work_dir.mkdir(exist_ok=True)
    
    try:
        papers = search_arxiv_by_date_range(
            keywords=request_params['keywords'],
            start_date_str=request_params['start_date'],
            end_date_str=request_params['end_date'],
            max_results=request_params['max_results'],
            process_log=process_log
        )
        background_tasks[run_id]['summary'] = process_log

        if not papers:
            process_log.append("INFO: 未找到符合条件的论文，任务结束。")
            background_tasks[run_id].update({"status": "completed", "summary": process_log})
            return

        target_language = request_params.get('target_language')
        if target_language and target_language.strip():
            process_log.append(f"INFO: 开始将 {len(papers)} 篇论文摘要翻译为 {target_language}...")
            background_tasks[run_id].update({"status": "translating", "summary": process_log})
            
            aclient = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_TRANSLATIONS)
            
            translation_tasks = [
                translate_one_abstract(aclient, paper['summary_en'], target_language, semaphore)
                for paper in papers
            ]
            translated_summaries = await asyncio.gather(*translation_tasks, return_exceptions=True)
            
            for paper, translated in zip(papers, translated_summaries):
                paper['summary_translated'] = translated if not isinstance(translated, Exception) else f"翻译失败: {translated}"
            process_log.append("SUCCESS: 所有摘要翻译完成。")
        else:
            process_log.append("INFO: 无需翻译。")
            for paper in papers:
                paper['summary_translated'] = ""

        topic_str = sanitize_filename_part("_".join(request_params['keywords']))
        lang_str = sanitize_filename_part(target_language) if target_language else "en"
        start_str = request_params['start_date']
        end_str = request_params['end_date']
        count = len(papers)
        csv_filename = f"arxiv_papers_{topic_str}_{start_str}_to_{end_str}_{lang_str}_{count}.csv"
        output_path = outputs_dir / csv_filename
        
        process_log.append(f"INFO: 准备将结果写入文件: {csv_filename}")
        
        fieldnames = ["原始关键词", "论文标题", "发表日期", "英文摘要", f"翻译摘要 ({lang_str})", "作者列表", "arxiv链接", "PDF链接"]
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for paper in papers:
                writer.writerow({
                    "原始关键词": paper.get("original_keyword", "N/A"),
                    "论文标题": paper['title'],
                    "发表日期": paper['published_date'],
                    "英文摘要": paper['summary_en'],
                    f"翻译摘要 ({lang_str})": paper['summary_translated'],
                    "作者列表": ", ".join(paper['authors']),
                    "arxiv链接": paper['arxiv_link'],
                    "PDF链接": paper.get('pdf_link', 'N/A')
                })
        
        process_log.append("🎉 SUCCESS: 任务成功完成！")
        background_tasks[run_id].update({
            "status": "completed",
            "summary": process_log,
            "output_path": str(output_path)
        })

    except Exception as e:
        logging.error(f"Run ID {run_id}: 处理过程中发生致命错误: {e}", exc_info=True)
        process_log.append(f"❌ FATAL_ERROR: {e}")
        background_tasks[run_id].update({"status": "failed", "summary": process_log})
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir)
            logging.info(f"Run ID {run_id}: 已清理临时工作目录。")