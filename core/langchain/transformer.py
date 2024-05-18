from langchain_text_splitters import RecursiveCharacterTextSplitter
def transform(dado):
    # quebrar o dado em pedaços menores
    # delimitadores que pegam contextos espalhados, quando quebramos ele não analisa tudo
    # categoriza
    text_splitter = RecursiveCharacterTextSplitter(
        # ao quebrar usando os separators, 
        # se for maior q 100, ele quebra de novo com outro tipo de separator,
        # enquanto for maior que o limite de 100
        # por exemplo (recursividade ate chegar no final, no final ele quebra em palavras)
        #  separators=[
        #     "\n\n",
        #     "\n",
        #     " ",
        #     ".",
        #     ",",
        #     "\u200b",  # Zero-width space
        #     "\uff0c",  # Fullwidth comma
        #     "\u3001",  # Ideographic comma
        #     "\uff0e",  # Fullwidth full stop
        #     "\u3002",  # Ideographic full stop
        #     "",
        # ],
        chunk_size=100, 
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_text(dado)