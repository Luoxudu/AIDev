"""文档加载器：支持 .md / .txt / .pdf / .docx 格式。"""

import os
from pathlib import Path

from langchain_core.documents import Document

_SUPPORTED_EXT = {".md", ".txt", ".pdf", ".docx"}
_ENCODINGS = ("utf-8", "gbk", "gb18030", "big5", "shift_jis", "latin-1")


def _load_text_file(path: Path, data_path: Path) -> Document | None:
    content = None
    for enc in _ENCODINGS:
        try:
            content = path.read_text(encoding=enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if content is None:
        print(f"[Loader] 跳过无法解码的文件 {path}（尝试编码: {', '.join(_ENCODINGS)}）")
        return None
    if not content.strip():
        return None
    return Document(
        page_content=content,
        metadata={"source": str(path.relative_to(data_path))},
    )


def _load_pdf_file(path: Path, data_path: Path) -> Document | None:
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(str(path))
    pages = loader.load()
    if not pages:
        return None
    content = "\n\n".join(p.page_content for p in pages if p.page_content.strip())
    if not content.strip():
        return None
    return Document(
        page_content=content,
        metadata={"source": str(path.relative_to(data_path))},
    )


def _load_docx_file(path: Path, data_path: Path) -> Document | None:
    from langchain_community.document_loaders import Docx2txtLoader
    loader = Docx2txtLoader(str(path))
    docs = loader.load()
    if not docs or not docs[0].page_content.strip():
        return None
    docs[0].metadata["source"] = str(path.relative_to(data_path))
    return docs[0]


_LOADERS = {
    ".md": _load_text_file,
    ".txt": _load_text_file,
    ".pdf": _load_pdf_file,
    ".docx": _load_docx_file,
}


def load_documents(data_dir: str) -> list[Document]:
    """递归加载目录下所有支持格式的文档。

    Args:
        data_dir: 包含文档的目录路径。

    Returns:
        Document 对象列表。
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录 {data_dir} 不存在。")

    docs: list[Document] = []
    data_path = Path(data_dir)

    for root, dirs, files in os.walk(data_dir, followlinks=False):
        dirs.sort()
        root_path = Path(root)
        for fname in sorted(files):
            f = root_path / fname
            if f.suffix.lower() not in _SUPPORTED_EXT:
                continue
            loader_fn = _LOADERS[f.suffix.lower()]
            try:
                doc = loader_fn(f, data_path)
            except Exception as e:
                print(f"[Loader] 跳过损坏文件 {f}: {e}")
                continue
            if doc is not None:
                docs.append(doc)

    if not docs:
        raise ValueError(
            f"在 {data_dir} 中没有找到任何非空文档（支持: {', '.join(sorted(_SUPPORTED_EXT))}）。"
        )

    print(f"[Loader] 已加载 {len(docs)} 个文档。")
    for doc in docs:
        print(f"  - {doc.metadata['source']} ({len(doc.page_content)} 字符)")
    return docs


# 旧接口兼容
load_markdown_files = load_documents
