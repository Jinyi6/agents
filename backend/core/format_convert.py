import os
import shutil
import zipfile
import tarfile
import re
import logging
from pathlib import Path

from openai import OpenAI

from config import (
    OPENAI_API_KEY, OPENAI_API_BASE, MODEL_NAME, conversion_tasks,
    workspace_dir, outputs_dir
)
from core.utils import retry_step

# --- 初始化模块 ---
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)


@retry_step
def call_llm(prompt: str) -> str:
    """
    调用大语言模型 (LLM) 并处理重试逻辑。

    Args:
        prompt (str): 发送给 LLM 的提示词。
        max_retries (int): 最大重试次数。

    Returns:
        str: LLM 返回的响应内容。
    
    Raises:
        Exception: 在多次重试后仍然失败时抛出异常。
    """
    logging.info("正在与 LLM 交互...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "你是一个精通 LaTeX 的助手，严格按照指令完成格式转换任务。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
    )
    logging.info(f"Asking: {prompt[:400]}...") # 避免日志过长
    if response and response.choices and response.choices[0].message and response.choices[0].message.content:
        content = response.choices[0].message.content
        content = re.sub(r'^```(latex)?\n', '', content, flags=re.MULTILINE)
        content = re.sub(r'\n```$', '', content, flags=re.MULTILINE)
        response = content.strip()
        logging.info(f"Response: {response}") 
        return response
    raise ValueError("LLM 返回了无效或空的响应。")

def extract_archive(archive_path: str, extract_to: str):
    """
    解压 .zip 或 .tar.gz 文件，并处理解压后可能出现的单层冗余目录。

    如果解压后 `extract_to` 目录中只包含一个子目录，
    则会将该子目录中的所有内容移动到 `extract_to`，并删除该空子目录。

    Args:
        archive_path (str): 压缩文件路径。
        extract_to (str): 解压目标目录。

    Returns:
        str: 压缩文件的后缀名 ('.zip' 或 '.tar.gz')。
    
    Raises:
        ValueError: 如果文件格式不受支持。
    """
    path = Path(archive_path)
    suffix = ''.join(path.suffixes)
    logging.info(f"正在解压文件: {archive_path} 到 {extract_to}")
    os.makedirs(extract_to, exist_ok=True)

    if suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif suffix in ['.tar.gz', '.tgz']:
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(path=extract_to)
    else:
        raise ValueError(f"不支持的压缩文件格式: {suffix}。")

    logging.info(f"检查解压目录 '{extract_to}' 是否存在单一冗余子目录。")
    items = os.listdir(extract_to)
    if len(items) == 1:
        single_item_path = os.path.join(extract_to, items[0])
        if os.path.isdir(single_item_path):
            logging.info(f"发现单一子目录: '{single_item_path}'，准备提升其内容。")
            for item_in_subdir in os.listdir(single_item_path):
                shutil.move(os.path.join(single_item_path, item_in_subdir), os.path.join(extract_to, item_in_subdir))
            os.rmdir(single_item_path)
            logging.info(f"内容提升完成，已删除空目录: '{single_item_path}'")
    return suffix

@retry_step
def find_main_tex_file(directory: str, process_log: list) -> str:
    """
    在指定目录中找到主 .tex 文件。

    逻辑:
    1. 如果只有一个 .tex 文件，则它就是主文件。
    2. 如果有多个，则使用 LLM 判断哪个是主文件。

    Args:
        directory (str): 要搜索的目录。

    Returns:
        str: 主 .tex 文件的绝对路径。

    Raises:
        FileNotFoundError: 如果找不到 .tex 文件或 LLM 无法确定主文件。
    """
    tex_files = [p for p in Path(directory).rglob('*.tex') if '__MACOSX' not in p.parts]
    if not tex_files:
        raise FileNotFoundError(f"在目录 '{directory}' 中没有找到任何 .tex 文件。")

    if len(tex_files) == 1:
        main_file = str(tex_files[0])
        logging.info(f"发现唯一 .tex 文件，认定为主文件: {main_file}")
        return main_file

    process_log.append("WARNING: 发现多个 .tex 文件，将使用 LLM 确定主文件。")
    logging.info("发现多个 .tex 文件，将使用 LLM 确定主文件。")

    file_tree = []
    for root, _, files in os.walk(directory):
        if '__MACOSX' in root: continue
        level = root.replace(directory, '').count(os.sep)
        indent = ' ' * 4 * level
        file_tree.append(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for f in files: file_tree.append(f"{sub_indent}{f}")
    file_tree_str = "\n".join(file_tree)
    tex_content_snippets = [f"--- 文件: {os.path.relpath(p, directory)} ---\n{p.read_text(encoding='utf-8', errors='ignore')[:300]}\n" for p in tex_files]
    prompt = f"""
你是一个LaTeX专家。请根据下面提供的文件结构和 .tex 文件内容片段，判断哪个是主 .tex 文件。
主文件通常是包含 `\\documentclass` 命令并且是整个文档编译入口的文件。
注意，有些tex可能看起来是入口文件，但内容上其实是format instruction或者example之类的。所以，请你看看，如果有更贴合一篇学术论文本身的tex文件，那么应该选这个文件。如果没有的话，那么format instruction文件应该也是可以的。

[文件结构]
````

{file_tree_str}

````

[.tex 文件内容预览]
{"".join(tex_content_snippets)}

请分析以上信息，并仅返回你认为是主文件的那个文件的相对路径 (例如: xx.tex 或 src/xx.tex)。不要添加任何解释。
"""
    main_file_relative_path = call_llm(prompt)
    main_file_path = Path(directory) / main_file_relative_path.strip()
    if not main_file_path.exists():
        raise FileNotFoundError(f"LLM 返回了不存在的文件路径: {main_file_relative_path}")
    logging.info(f"LLM 确定主文件为: {main_file_path}")
    return str(main_file_path)

def convert_paper_format_logic(run_id: str, content_archive_path: str, format_archive_path: str):
    """
    实现论文格式转换的主要流程。

    Args:
        content_archive_path (str): 原始论文压缩包路径。
        format_archive_path (str): 新格式压缩包路径。
    """
    process_log = ["INFO: 开始格式转换流程。"]
    conversion_tasks[run_id]['summary'] = process_log
    
    work_dir = workspace_dir / f"work_dir_{run_id}"
    if work_dir.exists(): shutil.rmtree(work_dir)
    work_dir.mkdir()

    content_dir = work_dir / "content"
    format_dir = work_dir / "format"
    
    try:
        # 步骤 1 & 2 & 3
        extract_archive(content_archive_path, str(content_dir))
        process_log.append("SUCCESS: 原始论文压缩包已解压。")
        extract_archive(format_archive_path, str(format_dir))
        process_log.append("SUCCESS: 新格式压缩包已解压。")
        conversion_tasks[run_id]['summary'] = process_log

        content_main_path = Path(find_main_tex_file(str(content_dir), process_log))
        process_log.append(f"SUCCESS: 找到原始论文主文件: '{content_main_path.name}'")
        format_main_path = Path(find_main_tex_file(str(format_dir), process_log))
        process_log.append(f"SUCCESS: 找到格式模板主文件: '{format_main_path.name}'")
        
        backup_path = format_main_path.with_suffix(".bak.tex")
        shutil.copy2(format_main_path, backup_path)
        process_log.append(f"INFO: 格式模板主文件已备份到 '{backup_path.name}'")
        conversion_tasks[run_id]['summary'] = process_log

        # 步骤 4: 合并文件目录
        shutil.copytree(str(content_dir), str(format_dir), dirs_exist_ok=True)
        process_log.append("INFO: 原始论文文件已全部复制到新格式目录。")
        content_main_path_in_format_dir = format_dir / content_main_path.name
        
        # 步骤 5: LLM 核心合并逻辑
        content_text = content_main_path_in_format_dir.read_text(encoding='utf-8', errors='ignore')
        format_text = format_main_path.read_text(encoding='utf-8', errors='ignore')

        # [修改点 3]: 先找 \section，再找 \input
        content_section_match = re.search(r'\\section', content_text)
        if not content_section_match:
            logging.info("未找到 `\\section`，正在尝试查找 `\\input` 作为正文起点...")
            content_section_match = re.search(r'\\input', content_text)

        if not content_section_match:
            raise ValueError("在原始论文主文件中未找到 `\\section` 或 `\\input` 作为正文起点。")
        
        content_split_index = content_section_match.start()
        content_header = content_text[:content_split_index]
        content_end_doc_match = re.search(r'\\end{document}', content_text)
        content_end_index = content_end_doc_match.start() if content_end_doc_match else len(content_text)
        content_body = content_text[content_split_index:content_end_index]

        format_section_match = re.search(r'\\section', format_text)
        format_split_index = format_section_match.start() if format_section_match else len(format_text)
        format_header = format_text[:format_split_index]

        header_prompt = f"""
你是一个LaTeX格式转换专家。请根据下面提供的“原始论文头部”和“新格式模板头部”，生成一个合并后的新头部。

[任务要求]
1.  **保留格式**: 新头部的基础结构和格式定义必须完全来自“新格式模板头部”。这包括 `\\documentclass`, 页面设置, 宏包(`\\usepackage`)等所有与样式相关的命令。
2.  **提取内容**: 从“原始论文头部”中，精准提取出以下与论文内容强相关的信息：（包括但不限于下面这些，请不要遗漏论文内容）
    -   必要的、内容相关的包 (例如 `\\usepackage{{amsmath}}`, `\\usepackage{{graphicx}}`)
    -   自定义命令 (例如 `\\newcommand`, `\\def`)
    -   论文标题 (`\\title{{...}}`)
    -   作者信息 (`\\author{{...}}`)
    -   摘要 (`\\abstract{{...}}` 或 `\\begin{{abstract}}...\\end{{abstract}}`)
3.  **合并**: 将提取出的内容信息，无缝地、正确地整合到新格式模板的结构中。如果新格式模板已有标题、作者等占位符，请用原始论文的实际内容替换它们。如果新格式没有某个部分(如摘要)，请根据上下文合理地添加。
4.  **输出**: 只输出合并后得到的、完整的、可以直接使用的LaTeX代码。代码应该从 `\\documentclass` 开始，到第一个 `\\section` 命令之前结束。不要包含任何解释或代码块标记。

---
[原始论文头部]
```latex
{content_header}
````

-----

[新格式模板头部]

```latex
{format_header}
```
"""
        merged_header = call_llm(header_prompt)
        process_log.append("SUCCESS: LLM 已成功合并文件头部信息。")

        # [修改点 4]: 宽松地匹配 bib 相关行
        content_bib_lines = []
        for line in content_text.splitlines():
            if line.strip().startswith(r'\bibliography') or line.strip().startswith(r'\bibliographystyle'):
                content_bib_lines.append(line)

        format_bib_lines = []
        for line in format_text.splitlines():
            if line.strip().startswith(r'\bibliography') or line.strip().startswith(r'\bibliographystyle'):
                format_bib_lines.append(line)

        final_bib_section = ""
        if format_bib_lines:
            original_format_bib = "\n".join(format_bib_lines)
            final_bib_section += f"% ===== Automatically commented out by the conversion script =====\n% {original_format_bib.replace(chr(10),chr(10)+'% ')}\n\n"
        
        if content_bib_lines:
            content_bib_str = "\n".join(content_bib_lines)
            # 从正文中移除旧的 bib 命令
            content_body = content_body.replace(content_bib_str, "")
            
            format_bib_str = "\n".join(format_bib_lines)
            bib_prompt = f"你是一个LaTeX文献管理专家。请根据下面的原始文献命令和新格式模板的文献命令，生成合并后的新命令。\n\n[任务要求]\n1. 使用新格式的样式和原始内容的 .bib 文件名，生成新的 `\\bibliographystyle` 和 `\\bibliography` 命令。\n2. 如果新格式命令为空，使用通用样式 `unsrt`。\n3. 只输出新命令，不要解释。\n\n[原始文献命令]:\n{content_bib_lines}\n\n[新格式文献命令]:\n{format_bib_lines}"
            merged_bib_lines = call_llm(bib_prompt)
            final_bib_section += merged_bib_lines
            process_log.append("SUCCESS: Bibliography 信息已成功合并。")
        else:
            process_log.append("INFO: 在原始论文中未找到 bib 命令。")
        
        # 组装和写入
        final_tex_content = f"{merged_header.strip()}\n\n{content_body.strip()}\n\n{final_bib_section.strip()}\n\n\\end{{document}}\n"
        format_main_path.write_text(final_tex_content, encoding='utf-8')
        process_log.append(f"SUCCESS: 新的主文件 '{format_main_path.name}' 已生成。")
        
        # 步骤 6: 打包输出
        output_zip_base_name = f"{run_id}_converted_paper"
        output_zip_path = outputs_dir / output_zip_base_name
        shutil.make_archive(str(output_zip_path), 'zip', str(format_dir))
        
        final_path = f"{output_zip_path}.zip"
        process_log.append("🎉 SUCCESS: 格式转换成功！")
        conversion_tasks[run_id].update({
            "status": "completed",
            "summary": process_log,
            "output_path": str(final_path)
        })

    except Exception as e:
        logging.error(f"Run ID {run_id}: 处理过程中发生致命错误: {e}", exc_info=True)
        process_log.append(f"❌ FATAL_ERROR: {e}")
        conversion_tasks[run_id].update({ "status": "failed", "summary": process_log })
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir)
            logging.info(f"Run ID {run_id}: 已清理临时工作目录。")