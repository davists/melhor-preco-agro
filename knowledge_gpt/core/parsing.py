from io import BytesIO
from typing import List, Any, Optional
import re

import docx2txt
from langchain.docstore.document import Document
import fitz
from hashlib import md5
from PIL import Image
import pytesseract
from pdf2image import convert_from_path, convert_from_bytes


from abc import abstractmethod, ABC
from copy import deepcopy


class File(ABC):
    """Represents an uploaded file comprised of Documents"""

    def __init__(
        self,
        name: str,
        id: str,
        metadata: Optional[dict[str, Any]] = None,
        docs: Optional[List[Document]] = None,
    ):
        self.name = name
        self.id = id
        self.metadata = metadata or {}
        self.docs = docs or []

    @classmethod
    @abstractmethod
    def from_bytes(cls, file: BytesIO) -> "File":
        """Creates a File from a BytesIO object"""

    def __repr__(self) -> str:
        return (
            f"File(name={self.name}, id={self.id},"
            " metadata={self.metadata}, docs={self.docs})"
        )

    def __str__(self) -> str:
        return f"File(name={self.name}, id={self.id}, metadata={self.metadata})"

    def copy(self) -> "File":
        """Create a deep copy of this File"""
        return self.__class__(
            name=self.name,
            id=self.id,
            metadata=deepcopy(self.metadata),
            docs=deepcopy(self.docs),
        )


def strip_consecutive_newlines(text: str) -> str:
    """Strips consecutive newlines from a string
    possibly with whitespace in between
    """
    return re.sub(r"\s*\n\s*", "\n", text)

class ImageFile(File):
    @classmethod
    def from_bytes(cls, file: BytesIO) -> "ImageFile":
        img = Image.open(file)
        text = pytesseract.image_to_string(img, lang='por')
        text = strip_consecutive_newlines(text)
        doc = Document(page_content=text.strip())
        doc.metadata["source"] = "p-1"
        return cls(name=file.name, id=md5(file.read()).hexdigest(), docs=[doc])

class DocxFile(File):
    @classmethod
    def from_bytes(cls, file: BytesIO) -> "DocxFile":
        text = docx2txt.process(file)
        text = strip_consecutive_newlines(text)
        doc = Document(page_content=text.strip())
        doc.metadata["source"] = "p-1"
        return cls(name=file.name, id=md5(file.read()).hexdigest(), docs=[doc])


class PdfFile(File):
    @classmethod
    def from_bytes(cls, file: BytesIO) -> "PdfFile":
        pdf = fitz.open(stream=file.read(), filetype="pdf")  # type: ignore
        docs = []
        for i, page in enumerate(pdf):
            text = page.get_text(sort=True)
            text = strip_consecutive_newlines(text)
            doc = Document(page_content=text.strip())
            doc.metadata["page"] = i + 1
            doc.metadata["source"] = f"p-{i+1}"
            docs.append(doc)
        # file.read() mutates the file object, which can affect caching
        # so we need to reset the file pointer to the beginning
        file.seek(0)

        if len(docs[0].page_content) == 0:
            images = pdf_to_img(file.read())
            text = ocr_core(images)

            doc = Document(page_content=text.strip())
            doc.metadata["page"] = 1
            doc.metadata["source"] = f"p-{1}"
            docs.append(doc)
            
        # Replace 'path_to_your_pdf.pdf' with the path to the PDF file you want to read
        # images = pdf_to_img('/workspaces/knowledge_gpt/knowledge_gpt/template_prompt/1546-contrato-escaneado-1599741721.pdf')
        # extracted_text = ocr_core(images)
        # docs.append(extracted_text)
        # print(extracted_text)
        return cls(name=file.name, id=md5(file.read()).hexdigest(), docs=docs)


class TxtFile(File):
    @classmethod
    def from_bytes(cls, file: BytesIO) -> "TxtFile":
        text = file.read().decode("utf-8", errors="replace")
        text = strip_consecutive_newlines(text)
        file.seek(0)
        doc = Document(page_content=text.strip())
        doc.metadata["source"] = "p-1"
        return cls(name=file.name, id=md5(file.read()).hexdigest(), docs=[doc])


def read_file(file: BytesIO) -> File:
    """Reads an uploaded file and returns a File object"""
    if file.name.lower().endswith(".docx"):
        return DocxFile.from_bytes(file)
    elif file.name.lower().endswith(".pdf"):
        return PdfFile.from_bytes(file)
    elif file.name.lower().endswith(".txt"):
        return TxtFile.from_bytes(file)
    elif file.name.lower().endswith(".jpg") or file.name.lower().endswith(".jpeg") or file.name.lower().endswith(".png"):
        return ImageFile.from_bytes(file)
    else:
        raise NotImplementedError(f"File type {file.name.split('.')[-1]} not supported")

def open_local_file(file_path: str) -> File:
    file_content = open(file_path, 'rb')
    return TxtFile.from_bytes(file_content)

def docs_as_text(_docs: Document):
    text = ''
    
    for _doc in _docs:
        text += f"{_doc.page_content}\n"
    
    return text


def pdf_to_img(pdf_file):
    return convert_from_bytes(pdf_file)

# Function to apply OCR on images
def ocr_core(images):
    text = ''
    for img in images:
        text += pytesseract.image_to_string(img, lang='por')  # You can specify language by adding lang='eng'
    return text
