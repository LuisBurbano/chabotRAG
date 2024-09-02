# Proyecto de Preguntas y Respuestas con LLMs y Base de Datos Vectorial 

Este proyecto es una aplicación web que utiliza modelos de lenguaje de última generación (LLMs) y una base de datos vectorial para proporcionar respuestas a preguntas sobre la Universidad de las Fuerzas Armadas "ESPE". La aplicación está desarrollada con Flask y hace uso de tecnologías modernas para ofrecer un chatbot interactivo.

## Descripción General

La aplicación utiliza un archivo CSV con preguntas frecuentes para construir una base de conocimiento. Los datos se procesan y dividen en fragmentos que se convierten en embeddings utilizando el modelo `sentence-transformers/all-mpnet-base-v2`. Estos embeddings se almacenan en una base de datos vectorial utilizando Chroma.

Para las respuestas, se utilizan dos modelos de lenguaje de la biblioteca LangChain: Gemma2 y Llama3. Dependiendo de la longitud de la pregunta, la aplicación selecciona el modelo adecuado para proporcionar una respuesta más precisa.

## Requisitos

- Python 3.8 o superior
- Flask
- pandas
- torch
- sentence-transformers
- chromadb
- langchain

#Autores
Luis Burbano, Sebastian Torres, Cesar Loor

