#!/usr/bin/python
# -*- coding: utf-8 -*-
import gradio as gr
import click
import torch
import logging
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.vectorstores import Chroma
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, \
    PERSIST_DIRECTORY
from transformers import GenerationConfig


def load_model(device_type, model_id, model_basename=None):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models. 
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """

    # logging.info(f'Loading Model: {model_id}, on: {device_type}')
    # logging.info('This action can take a few minutes!')

    if model_basename is not None:
        # The code supports all huggingface models that ends with GPTQ and have some variation of .no-act.order or .safetensors in their HF repo.
        if '.safetensors' in model_basename:
            # Remove the ".safetensors" ending if present
            model_basename = model_basename.replace('.safetensors', '')

        model = AutoGPTQForCausalLM.from_quantized(model_id,
                model_basename=model_basename, device='cuda',
                use_safetensors=True, use_triton=False)

    # Create a pipeline for text generation

    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
        )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info('Local LLM Loaded')

    return local_llm


model_id = 'TheBloke/WizardLM-7B-uncensored-GPTQ'
model_basename = \
    'WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors'
llm = load_model('cuda', model_id=model_id,
                 model_basename=model_basename)


def greet(query):
    '''
    This function implements the information retreival task.


    1. Loads an embedding model, can be HuggingFaceInstructEmbeddings or HuggingFaceEmbeddings
    2. Loads the existing vectorestore that was created by inget.py
    3. Loads the local LLM using load_model function - You can now set different LLMs.
    4. Setup the Question Answer retreival chain.
    5. Question answers.
    '''

    embeddings = \
        HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'})

    # load the vectorstore

    db = Chroma(persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings,
                client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff',
            retriever=retriever, return_source_documents=False)
    res = qa(query)
    answer = res['result']
    return answer


demo = gr.Interface(fn=greet, inputs='text', outputs='text')
demo.queue().launch(share=True)
