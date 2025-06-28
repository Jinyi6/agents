import logging
import json
from typing import Dict, List, Optional

from openai import OpenAI

from config import (
    OPENAI_API_KEY, OPENAI_API_BASE, MODEL_NAME, style_transfer_tasks
)
from core.utils import retry_step

# --- 初始化 OpenAI 客户端 ---
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

def build_prompt(original_text: str, must_include_keywords: Optional[List[str]], reference_keywords: Optional[List[str]], style_requirements: Optional[List[str]], style_example: Optional[str], previous_results: Optional[List[str]] = None, mode = None) -> str:
    """构建用于文本润色的详细提示词 (版本 2.0)"""
    
    prompt = "你是一位顶级的学术写作专家和语言模型。你的任务是基于一系列极其严格和精确的指令，对一段初始文本进行深度、专业的重构和优化。\n\n"
    
    prompt += "【核心改写准则：必须严格遵守】\n"
    prompt += "1.  **关键词强制注入 (Keyword Injection Mandate)**: 在任何情况下，**[必须包含的关键词]** 列表中的每一个词，都必须 **一字不差 (verbatim)** 地出现在你最终生成的文本中。在生成后，你必须进行自我核查，确保没有任何遗漏。这是一个绝对的、不可协商的指令。\n"
    prompt += "2.  **风格深度复刻 (Style Replication Imperative)**: 你的首要目标是成为**[风格参考示例]**作者的“影子写手”。在动笔前，你必须进行深度的风格解构分析。你的最终输出在**阐述视角、句式复杂度、词汇选择和行文节奏**上，必须达到与参考范例难以区分的程度。单纯的模仿是不够的，你需要实现风格的完全复现。\n\n"

    prompt += "【改写执行流程】\n"
    prompt += "1.  **第一步：解构分析**\n"
    prompt += "    - **内容分析**: 彻底理解 **[待改写的表述]** 的所有核心信息点和逻辑关系，确保无遗漏、无曲解。\n"
    prompt += "    - **风格分析**: 系统性解构 **[风格参考示例]** 的句式结构（例如，主从复合句的比例、平均句长）、专业词组搭配 (collocations) 和整体的阐述视角（客观、主观、批判性等）。\n"
    prompt += "2.  **第二步：融合重构**\n"
    prompt += "    - 依据分析所得，用**复刻的风格**重新组织和表达**待改写的内容**。\n"
    prompt += "    - 在重构过程中，将**[必须包含的关键词]** 自然、无缝地植入文本，使其看起来像是原文固有的一部分。\n"
    prompt += "    - 同时，确保**[需表达含义的关键词]** 的核心语义被准确传达。\n"
    prompt += "3.  **第三步：最终校验**\n"
    prompt += "    - **检查关键词**: 回溯检查，确认所有**[必须包含的关键词]**都已一字不差地包含在内。\n"
    prompt += "    - **比对风格**: 将你的草稿与**[风格参考示例]**并排比对，评估风格的一致性。如果不达标，返回第二步进行修改。\n\n"
    
    prompt += "---【输入材料清单】---\n"
    prompt += f"1. **[待改写的表述]**:\n   - {original_text}\n\n"
    
    if must_include_keywords:
        prompt += f"2. **[必须包含的关键词]** (必须一字不差地、自然地嵌入到改写后的文本中):\n"
        for keyword in must_include_keywords:
            prompt += f"   - `{keyword}`\n"
        prompt += "\n"

    if reference_keywords:
        prompt += f"3. **[需表达含义的关键词]** (不必直接使用原词，但必须准确、完整地传达其核心语义):\n"
        for keyword in reference_keywords:
            prompt += f"   - {keyword}\n"
        prompt += "\n"

    if style_requirements:
        prompt += f"4. **[风格要求]**:\n"
        for style in style_requirements:
            prompt += f"   - {style}\n"
        prompt += "\n"

    if style_example:
        prompt += f"5. **[风格参考示例]** (请深度分析并模仿其阐述视角、句式结构和词组搭配):\n"
        prompt += f"   - \"{style_example}\"\n\n"
        
    prompt += "---【输出指令】---\n"
    
    if previous_results:
        prompt += "你之前已经生成了以下版本，请在本次生成中提供一个与之前版本**截然不同**的、全新的、创新的版本。\n"
        prompt += "[之前已生成的版本]:\n"
        for i, result in enumerate(previous_results):
            prompt += f"{i+1}. {result}\n"
        prompt += "\n"
        prompt += "请严格遵循上述所有要求，只输出一个**新**的、经过润色的文本版本。不要包含任何解释或代码块标记。"
    else:
        num_results = "3到5个"
        prompt += f"请严格遵循上述所有要求，生成 **{num_results}** 个经过润色的、风格各异的文本版本。请以JSON格式的列表形式返回，列表中每个元素都是一个完整的文本字符串。例如：[\"完整的改写结果1\", \"完整的改写结果2\", ...]。不要添加任何解释或代码块标记。"

    return prompt

@retry_step
def call_llm_for_style_transfer(prompt: str, is_json: bool = False) -> any:
    """调用LLM进行风格转换，并根据需要解析JSON"""
    logging.info("正在与 LLM 交互进行文本润色...")
    
    response_format = {"type": "json_object"} if is_json else {"type": "text"}

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        response_format=response_format
    )
    
    content = response.choices[0].message.content.strip()
    logging.info(f"LLM Response (raw): {content[:500]}...")

    if is_json:
        try:
            if content.startswith("```json"):
                content = content[7:-3].strip()
            return json.loads(content)
        except json.JSONDecodeError as e:
            logging.error(f"LLM did not return valid JSON: {e}")
            # 尝试从非标准格式中挽救数据，例如纯文本列表
            if '[' in content and ']' in content:
                logging.warning("Attempting to salvage list from malformed JSON.")
                try:
                    # 这是一个简单的挽救尝试，可能需要更复杂的解析
                    salvaged_content = content[content.find('['):content.rfind(']')+1]
                    return json.loads(salvaged_content)
                except json.JSONDecodeError:
                    logging.error("Salvage attempt failed.")
            raise ValueError("LLM 返回了无效的JSON格式。")
    else:
        return content


def run_style_transfer_logic(run_id: str, request_params: dict):
    """
    执行文本润色任务的主逻辑 (版本 2.0)。
    """
    process_log = style_transfer_tasks[run_id]['summary']
    mode = request_params.get("mode", "标准")
    
    try:
        if mode == "专业":
            process_log.append(f"INFO: 已启动【专业模式】，将执行 7+1 轮 LLM 调用。")
            style_transfer_tasks[run_id]['summary'] = process_log
            
            # 1. 迭代生成7个结果
            generated_results = []
            for i in range(7):
                process_log.append(f"INFO: 正在进行第 {i+1}/7 轮迭代生成...")
                style_transfer_tasks[run_id]['summary'] = process_log
                
                prompt = build_prompt(previous_results=generated_results, **request_params)
                new_result = call_llm_for_style_transfer(prompt, is_json=False)
                generated_results.append(new_result)
                process_log.append(f"SUCCESS: 第 {i+1} 轮生成完成。")
                style_transfer_tasks[run_id]['summary'] = process_log

            # 2. LLM挑选最佳4个
            process_log.append("INFO: 7轮迭代完成，正在调用 LLM 挑选最佳的4个结果...")
            style_transfer_tasks[run_id]['summary'] = process_log
            
            selection_prompt = f"""
你是一位资深的文本编辑和评论家。这里有7个基于相同要求润色后的文本版本。请仔细评估它们，并选出其中**最优秀、最符合要求、且风格差异最明显**的4个版本。

[原始要求]
- 原始表述: {request_params['original_text']}
- 必须包含的关键词: {request_params.get('must_include_keywords')}
- 风格要求: {request_params.get('style_requirements')}
- 风格参考示例: {request_params.get('style_example')}

[7个候选版本]
{chr(10).join(f'--- 版本 {i+1} ---\n"{res}"' for i, res in enumerate(generated_results))}

请严格按照以下格式返回一个JSON列表，列表中包含你选出的4个版本的**完整原始文本**。
**输出格式示例**:
["这里是版本A的完整文本内容...", "这里是版本B的完整文本内容...", "这里是版本C的完整文本内容...", "这里是版本D的完整文本内容..."]
不要添加任何解释、序号或代码块标记。只输出纯粹的JSON列表。
"""
            final_results_raw = call_llm_for_style_transfer(selection_prompt, is_json=True)
            
            # --- 鲁棒性处理逻辑 ---
            final_results = []
            if isinstance(final_results_raw, list) and all(isinstance(item, str) for item in final_results_raw) and len(final_results_raw) >= 4:
                # 理想情况：返回的是一个包含4个或更多字符串的列表
                final_results = final_results_raw[:4]
                logging.info("LLM成功挑选并返回了4个文本结果。")
                process_log.append("SUCCESS: LLM 已成功挑选出4个最佳结果。")
            else:
                # 备用方案：如果返回的不是文本列表（例如是索引列表 [1, 2, 5, 7]）
                logging.warning(f"LLM 未按要求返回文本列表，返回内容: {final_results_raw}。正在尝试从索引恢复。")
                process_log.append("WARNING: LLM 未按预期格式返回，尝试从索引恢复。")
                
                indices_to_use = []
                if isinstance(final_results_raw, list) and all(isinstance(item, int) for item in final_results_raw):
                    # 确认返回的是一个整数列表
                    indices_to_use = [i - 1 for i in final_results_raw if 0 < i <= len(generated_results)] # 转换为0-based索引
                
                if len(indices_to_use) >= 4:
                    final_results = [generated_results[i] for i in indices_to_use[:4]]
                    logging.info(f"成功从LLM返回的索引 {indices_to_use[:4]} 恢复了4个结果。")
                    process_log.append("SUCCESS: 已从LLM返回的索引成功恢复结果。")
                else:
                    # 终极备用方案：如果所有方法都失败，默认选择前4个
                    logging.error("无法从LLM的输出中恢复选择，将默认使用前4个生成的结果。")
                    process_log.append("ERROR: 无法解析LLM的选择，默认选用前4个结果。")
                    final_results = generated_results[:4]

            style_transfer_tasks[run_id]['summary'] = process_log

        else: # 标准模式
            process_log.append(f"INFO: 已启动【标准模式】，将执行 1 轮 LLM 调用。")
            style_transfer_tasks[run_id]['summary'] = process_log
            
            prompt = build_prompt(**request_params)
            final_results = call_llm_for_style_transfer(prompt, is_json=True)
            if not isinstance(final_results, list):
                 raise ValueError("LLM未能返回结果列表。")

            process_log.append(f"SUCCESS: LLM 已生成 {len(final_results)} 条润色结果。")
            style_transfer_tasks[run_id]['summary'] = process_log
        
        # 3. LLM生成修改建议 (保持不变)
        process_log.append("INFO: 正在调用 LLM 为最终结果生成修改建议...")
        style_transfer_tasks[run_id]['summary'] = process_log
        
        suggestions_prompt = f"""
你是一位乐于助人的写作助理。这里有几条由AI润色后的文本。请你站在用户的角度，检查这些结果是否完全符合原始要求，并提供一小段精炼的、可操作的修改建议。

[原始要求]
- 原始表述: {request_params['original_text']}
- 必须包含的关键词: {request_params.get('must_include_keywords')}
- 参考关键词: {request_params.get('reference_keywords')}
- 风格要求: {request_params.get('style_requirements')}
- 风格参考示例: {request_params.get('style_example')}

[最终润色结果]
{chr(10).join(f'- {res}' for res in final_results)}

请根据以上信息，生成一小段文本提示，指出这些结果中可能存在的、需要用户手动微调的问题（例如：某个必须包含的关键词是否自然融入？风格是否完全对齐？），并给出修改建议。你的回答应该是直接面向用户的、友好的文本。
"""
        suggestions = call_llm_for_style_transfer(suggestions_prompt, is_json=False)
        process_log.append("SUCCESS: LLM 已生成修改建议。")
        
        # 4. 任务完成 (保持不变)
        process_log.append("🎉 SUCCESS: 文本润色任务成功完成！")
        style_transfer_tasks[run_id].update({
            "status": "completed",
            "summary": process_log,
            "result": {
                "results": final_results,
                "suggestions": suggestions
            }
        })

    except Exception as e:
        logging.error(f"Run ID {run_id}: 处理过程中发生致命错误: {e}", exc_info=True)
        process_log.append(f"❌ FATAL_ERROR: {e}")
        style_transfer_tasks[run_id].update({"status": "failed", "summary": process_log})