import os
import streamlit as st
from timeit import default_timer as timer
from datetime import timedelta
import pickle

from knowledge_gpt.components.sidebar import sidebar

from knowledge_gpt.ui import (
    wrap_doc_in_html,
    is_query_valid,
    is_open_ai_key_valid,
    display_file_read_error,
)

from knowledge_gpt.core.caching import bootstrap_caching

from knowledge_gpt.core.parsing import open_local_file, read_file, docs_as_text
from knowledge_gpt.core.chunking import chunk_file
from knowledge_gpt.core.embedding import embed_files
from knowledge_gpt.core.qa import query_folder
from knowledge_gpt.core.utils import get_llm
from pathlib import Path
import glob

EMBEDDING = "openai"
VECTOR_STORE = "faiss"
MODEL_LIST = ["gpt-4-turbo"]
PERSPECTIVA_SOU_LOCADOR = 'Quero Vender a Produção de Soja'
PERSPECTIVA_SOU_LOCATARIO = 'Quero Comprar Insumos'
PROJECT_ROOT = Path(__file__).parent.resolve()

# Uncomment to enable debug mode
# MODEL_LIST.insert(0, "debug")

def on_button_locador_clicked():
    st.session_state['prompt_perspective'] = PERSPECTIVA_SOU_LOCADOR
    # Open the file in read mode
    with open(f"{PROJECT_ROOT}/template_prompt/venda.txt", 'r') as file:
        # Read the contents of the file
        st.session_state['prompt'] =  file.read()

def on_button_locatario_clicked():
    st.session_state['prompt_perspective'] = PERSPECTIVA_SOU_LOCATARIO
    with open(f"{PROJECT_ROOT}/template_prompt/compra.txt", 'r') as file:
        # Read the contents of the file
        st.session_state['prompt'] =  file.read()


st.set_page_config(page_title="Melhor Preço Agro", page_icon="💵", layout="wide")
st.header("Melhor Preço Agro")

# Enable caching for expensive functions
bootstrap_caching()

st.session_state['prompt_perspective'] = None if 'prompt_perspective' not in st.session_state else st.session_state['prompt_perspective']
st.session_state['prompt'] = None if 'prompt' not in st.session_state else st.session_state['prompt']
openai_api_key = os.environ.get('OPENAI_API_KEY')

uploaded_file = st.file_uploader(
    "Upload a pdf, docx, or txt file",
    type=["pdf", "docx", "doc","txt", "jpg", "png", "jpeg"],
    # help="Scanned documents are not supported yet!",
)

model: str = st.selectbox("Model", options=MODEL_LIST)  # type: ignore

return_all_chunks = True
show_full_doc = False

if not uploaded_file:
    st.stop()

try:
    upload_file_content = read_file(uploaded_file)
    chunked_files = []

    chunck_pkl = f"{PROJECT_ROOT}/chunked_files.pkl"
    if os.path.exists(chunck_pkl):
        # Open the file in binary read mode
        with open(chunck_pkl, 'rb') as file:
            # Unpickle the data
            chunked_files = pickle.load(file)
        # print(chunked_files)
    else:
        base_knowledge = glob.glob(f"{PROJECT_ROOT}/knowledge_base/*.txt")

        for _file in base_knowledge:
            file_base = open_local_file(_file)
            chunked_files.append(chunk_file(file_base, chunk_size=300, chunk_overlap=0))
            break

        # Open a file and use pickle.dump()
        with open(chunck_pkl, 'wb') as file:
            pickle.dump(chunked_files, file)
        
except Exception as e:
    display_file_read_error(e, file_name=uploaded_file.name)

if not is_open_ai_key_valid(openai_api_key, model):
    st.stop()

with st.spinner("Indexando o Documento ...⏳"):
    folder_index = embed_files(
        files=chunked_files,
        embedding=EMBEDDING if model != "debug" else "debug",
        vector_store=VECTOR_STORE if model != "debug" else "debug",
        openai_api_key=openai_api_key,
    )

if show_full_doc:
    with st.expander("Documento Enviado Para Analise"):
        # Hack to get around st.markdown rendering LaTeX
        print()
        st.markdown(f"<p>{wrap_doc_in_html(docs_as_text(upload_file_content.docs))}</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1,2])

with col1:
    st.button(PERSPECTIVA_SOU_LOCADOR, on_click=on_button_locador_clicked)

with col2:
    st.button(PERSPECTIVA_SOU_LOCATARIO, on_click=on_button_locatario_clicked)

if st.session_state['prompt_perspective'] is not None:
    query = st.session_state['prompt'].format(context=docs_as_text(upload_file_content.docs))

if st.session_state['prompt_perspective'] is not None:
    if not is_query_valid(query):
        st.stop()
    
    with st.form(key="qa_form"):
        query = st.text_area("Prompt Sobre o Documento Enviado", value=query)
        submit = st.form_submit_button("Gerar Resposta")

    if submit:
        with st.spinner("Gerando a Resposta ...⏳"):
            # Output Columns
            answer_col, sources_col = st.columns(2)
            start = timer()
            
            llm = get_llm(model=model, openai_api_key=openai_api_key, temperature=0)
            result = query_folder(
                folder_index=folder_index,
                query=query,
                return_all=return_all_chunks,
                llm=llm,
            )

            end = timer()
            time_elapsed = timedelta(seconds=end-start)

            with answer_col:
                st.markdown("#### Resposta (GPT4 + Fontes Internas)")
                st.markdown(result.answer)
                st.markdown(f"Tempo de Processamento: {time_elapsed}")

            with sources_col:
                st.markdown("#### Fontes Internas Utilizadas")
                for source in result.sources:
                    st.markdown(source.page_content)
                    st.markdown(source.metadata["source"])
                    st.markdown("---")
