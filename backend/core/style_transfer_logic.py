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

def build_prompt(original_text: str, must_include_keywords: Optional[List[str]], reference_keywords: Optional[List[str]], style_requirements: Optional[List[str]], style_example: Optional[str], previous_results: Optional[List[str]] = None) -> str:
    """构建用于文本润色的详细提示词"""
    
    prompt = "你是一位精通特定专业领域的学术写作专家和高级文本编辑。你的任务是基于一系列精确的指令，对一段初始文本进行深度、专业的重构和优化。\n\n"
    prompt += "【核心任务】\n"
    prompt += "根据下述的输入材料和具体要求，将**[待改写的表述]**改写为一段在内容上完整、准确，在风格上高度专业、正式、严谨的文本。\n\n"

    prompt += """
【改写执行指令】
1.  **忠实于原意**: 确保改写后的文本全面、无遗漏地覆盖**[待改写的表述]**中的所有核心信息点和逻辑关系。
2.  **深度风格分析**: 在动笔前，请对**[风格参考示例]**进行系统性分析，重点关注以下维度：
    * **阐述视角 (Perspective)**: 分析范例是从何种立场进行阐述的（例如：客观研究者、技术报告撰写者、领域评论员等）。
    * **句式结构 (Sentence Structure)**: 解构范例中句子的典型长度、复杂性（如主从复合句的使用）和语法范式。
    * **词组与搭配 (Phrasing & Collocation)**: 识别范例中具有标志性的专业术语、固定短语搭配和行文节奏。
    
    **系统性风格复现**: 在改写过程中，必须主动、系统地运用你在上一步中分析出的风格特征。你的最终输出在**阐述视角、句式结构、词组搭配**上，都应与**[风格参考示例]**保持高度一致性。
"""
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
        
    prompt += "---【改写执行指令】---\n"
    
    if previous_results:
        prompt += "你之前已经生成了以下版本，请在本次生成中提供一个与之前版本**截然不同**的、全新的、创新的版本。\n"
        prompt += "[之前已生成的版本]:\n"
        for i, result in enumerate(previous_results):
            prompt += f"{i+1}. {result}\n"
        prompt += "\n"
        prompt += "请严格遵循上述所有要求，只输出一个**新**的、经过润色的文本版本，不要包含任何解释或代码块标记。"
    else:
        num_results = "3到5个"
        prompt += f"请严格遵循上述所有要求，生成 **{num_results}** 个经过润色的、风格各异的文本版本。请以JSON格式的列表形式返回，例如：[\"结果1\", \"结果2\", ...]。不要添加任何解释或代码块标记。"

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
        temperature=0.4, # 稍微提高温度以获得更多样化的结果
        response_format=response_format
    )
    
    content = response.choices[0].message.content.strip()
    logging.info(f"LLM Response (raw): {content[:500]}...")

    if is_json:
        try:
            # 兼容处理LLM可能返回的被```json ... ```包裹的响应
            if content.startswith("```json"):
                content = content[7:-3].strip()
            return json.loads(content)
        except json.JSONDecodeError as e:
            logging.error(f"LLM did not return valid JSON: {e}")
            raise ValueError("LLM 返回了无效的JSON格式。")
    else:
        return content


def run_style_transfer_logic(run_id: str, request_params: dict):
    """
    执行文本润色任务的主逻辑。
    """
    process_log = style_transfer_tasks[run_id]['summary']
    mode = request_params.get("mode", "标准")
    
    try:
        if mode == "专业":
            process_log.append(f"INFO: 已启动【专业模式】，将执行 7+2 轮 LLM 调用。")
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
{chr(10).join(f'{i+1}. {res}' for i, res in enumerate(generated_results))}

请以JSON格式返回一个包含你选出的4个文本的列表。例如：["选择的版本A", "选择的版本B", ...]。不要添加任何解释或代码块标记。
"""
            final_results = call_llm_for_style_transfer(selection_prompt, is_json=True)
            if not isinstance(final_results, list) or len(final_results) != 4:
                # 如果LLM返回格式不正确，则手动选择前4个作为备用方案
                logging.warning("LLM未能正确挑选4个结果，将默认使用前4个。")
                final_results = generated_results[:4]

            process_log.append("SUCCESS: LLM 已挑选出4个最佳结果。")
            style_transfer_tasks[run_id]['summary'] = process_log

        else: # 标准模式
            process_log.append(f"INFO: 已启动【标准模式】，将执行 1+1 轮 LLM 调用。")
            style_transfer_tasks[run_id]['summary'] = process_log
            
            prompt = build_prompt(**request_params)
            final_results = call_llm_for_style_transfer(prompt, is_json=True)
            if not isinstance(final_results, list):
                 raise ValueError("LLM未能返回结果列表。")

            process_log.append(f"SUCCESS: LLM 已生成 {len(final_results)} 条润色结果。")
            style_transfer_tasks[run_id]['summary'] = process_log
        
        # 3. LLM生成修改建议
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
        
        # 4. 任务完成
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

